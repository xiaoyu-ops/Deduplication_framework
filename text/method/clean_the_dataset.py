# 数据集清理工具 - 使用Jaccard相似度去重
"""
这个工具利用jaccard_deduplication.py中的算法来清理和去重数据集
支持多种数据集格式和自定义去重参数
"""

import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm
from jaccard_deduplication import (
    deduplicate_dataset_jaccard, 
    deduplicate_cross_splits_jaccard,
    quick_jaccard_deduplicate,
    jaccard_similarity,
    get_ngrams,
    normalize_text
)
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) #os.path.dirname(__file__)当前文件的绝对路径 然后dirname取目录
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)  # 添加这行，获取根目录

# 将根目录添加到Python路径（这样可以导入env_manager包）
sys.path.insert(0, root_dir)
print(f"已添加根目录到路径: {root_dir}")

# 现在可以用完整包路径导入
from env_manager.manager import EnvManager

# 全局函数用于支持multiprocessing
def process_chunk_worker(chunk_data): # 将大数据集分成多个小块，每个块再独立的进程中处理。
    """
    全局工作函数，用于multiprocessing
    必须在模块级别定义以支持pickle序列化
    """
    try:
        from jaccard_deduplication import jaccard_similarity, normalize_text, get_ngrams
        from datasets import Dataset
        
        text_field, threshold, ngram_size = chunk_data['params']
        chunk_items = chunk_data['chunk_items']
        chunk_id = chunk_data.get('chunk_id', 0)
        original_range = chunk_data.get('original_range', 'unknown')
        
        print(f"块 {chunk_id} ({original_range}): 开始处理...")
        
        if not chunk_items:
            print(f"块 {chunk_id}: 数据为空")
            return None
        
        print(f"块 {chunk_id}: 接收到 {len(chunk_items)} 条数据")
        
        # 验证数据格式
        valid_items = []
        for i, item in enumerate(chunk_items):
            if isinstance(item, dict) and text_field in item:
                text = item[text_field]
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    valid_items.append(item) # 检查后文本内容有效就可以加入已验证数据集中
                else:
                    print(f"块 {chunk_id}: 项目 {i} 文本内容无效")
            else:
                print(f"块 {chunk_id}: 项目 {i} 格式无效")
        
        if not valid_items:
            print(f"块 {chunk_id}: 没有有效数据")
            return None
        
        print(f"块 {chunk_id}: 有效数据 {len(valid_items)} 条")
        
        # 重建Dataset对象
        try:
            chunk = Dataset.from_list(valid_items) # 只用有效数据重建Dataset
            print(f"块 {chunk_id}: Dataset重建成功，包含 {len(chunk)} 条数据")
        except Exception as e:
            print(f"块 {chunk_id}: Dataset重建失败: {e}")
            return None
        
        # 优化的去重逻辑 - 使用哈希和预过滤
        kept_indices = []
        seen_ngram_sets = []  # 存储n-gram集合而不是原始文本
        seen_text_hashes = set()  # 快速查找重复文本
        processed_count = 0
        start_time = time.time()
        
        # 早期终止检测变量
        check_interval = min(500, len(chunk) // 10)  # 检查间隔
        early_stop_threshold = 0.95  # 如果保留率超过95%，可能去重效果有限
        
        # 创建进度条
        pbar = tqdm(
            total=len(chunk), 
            desc=f"块 {chunk_id}", 
            position=chunk_id, 
            leave=False, 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] 保留:{postfix}'
        )
        
        for i, item in enumerate(chunk):
            processed_count += 1
            
            if not isinstance(item, dict) or text_field not in item:
                pbar.update(1) # 手动给进度条加一
                continue
                
            text = item[text_field]
            if not text or not isinstance(text, str):
                pbar.update(1)
                continue
                
            # 快速哈希检查 - 完全相同的文本
            text_hash = hash(text.strip())
            if text_hash in seen_text_hashes:
                pbar.update(1)
                continue
            
            # 计算当前文本的n-gram集合
            current_ngrams = set(get_ngrams(normalize_text(text), ngram_size))
            
            # 长度和集合大小预过滤
            text_len = len(text)
            ngram_count = len(current_ngrams)
            
            is_duplicate = False
            similarity_checks = 0
            
            # 与已保留的文本进行相似度比较
            for j, seen_ngrams in enumerate(seen_ngram_sets):
                # 快速预过滤检查
                if ngram_count == 0 or len(seen_ngrams) == 0:
                    continue
                    
                # 长度比率预过滤
                size_ratio = ngram_count / len(seen_ngrams)
                if size_ratio > 2.0 or size_ratio < 0.5:
                    continue
                
                # 快速交集估算
                intersection_size = len(current_ngrams & seen_ngrams)
                min_size = min(ngram_count, len(seen_ngrams))
                
                if min_size > 0:
                    quick_similarity = intersection_size / min_size
                    if quick_similarity < 0.3:  # 不可能达到0.8相似度
                        continue
                
                # 计算精确Jaccard相似度
                try:
                    similarity_checks += 1
                    union_size = len(current_ngrams | seen_ngrams)
                    if union_size > 0:
                        similarity = intersection_size / union_size
                        if similarity >= threshold:
                            is_duplicate = True
                            break
                except Exception as e:
                    continue
            
            if not is_duplicate:
                kept_indices.append(i)
                seen_ngram_sets.append(current_ngrams)
                seen_text_hashes.add(text_hash)
            
            # 更新进度条，显示去重效果和处理速度
            if i % 100 == 0 and i > 0:  # 每100条更新一次性能信息
                elapsed = time.time() - start_time
                speed = (i+1) / elapsed if elapsed > 0 else 0
                retention_rate = len(kept_indices) / (i+1) * 100 if i > 0 else 100 # 数据保存率
                pbar.set_postfix_str(f"保留:{len(kept_indices)} ({retention_rate:.1f}%) 速度:{speed:.1f}/s")
                
                # 早期终止检测
                if i >= check_interval and retention_rate > early_stop_threshold:
                    print(f"\n块 {chunk_id}: 检测到去重效果有限 (保留率 {retention_rate:.1f}%)，建议检查数据质量")
            else:
                pbar.set_postfix_str(f"{len(kept_indices)}")
            pbar.update(1)
        
        pbar.close()
        
        processing_time = time.time() - start_time
        final_speed = len(chunk) / processing_time if processing_time > 0 else 0
        
        print(f"块 {chunk_id}: 处理完成，保留 {len(kept_indices)}/{len(chunk)} 条 [总耗时: {processing_time:.1f}秒, 平均速度: {final_speed:.1f} 条/秒]")
        
        if kept_indices:
            result = chunk.select(kept_indices)
            print(f"块 {chunk_id}: 成功创建结果Dataset，包含 {len(result)} 条数据")
            return result
        else:
            print(f"块 {chunk_id}: 所有数据被去重，返回空结果")
            return None
            
    except Exception as e:
        chunk_id = chunk_data.get('chunk_id', 'unknown')
        print(f"块 {chunk_id} 处理失败: {str(e)}")
        import traceback
        print(f"块 {chunk_id} 错误详情: {traceback.format_exc()}")
        return None

def load_local_dataset(file_path, text_field='text'):
    """从本地文件加载数据集"""
    from datasets import Dataset
    import json
    import pandas as pd
    import os
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return None
    
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            print(f"正在加载JSON文件: {file_path}")
            
            # 尝试不同的JSON格式
            try:
                # 尝试加载行式JSON (每行一个JSON对象)
                dataset = Dataset.from_json(file_path)
                print(f"成功以行式JSON格式加载")
                return dataset
            except Exception as e:
                print(f"行式JSON加载失败，尝试标准JSON格式: {e}")
                
                try:
                    # 尝试加载标准JSON (整个文件是一个JSON对象/数组)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 处理不同的JSON结构
                    if isinstance(data, list):
                        dataset = Dataset.from_list(data)
                        print(f"成功加载JSON数组，包含{len(dataset)}条记录")
                    elif isinstance(data, dict):
                        # 检查是否有数据字段
                        if 'data' in data and isinstance(data['data'], list):
                            dataset = Dataset.from_list(data['data'])
                            print(f"成功加载带data字段的JSON，包含{len(dataset)}条记录")
                        else:
                            # 将单个字典转换为列表
                            dataset = Dataset.from_list([data])
                            print("成功加载单个JSON对象")
                    else:
                        print(f"不支持的JSON格式")
                        return None
                        
                    return dataset
                except Exception as e2:
                    print(f"JSON加载失败: {e2}")
                    return None
        
        elif file_ext == '.csv':
            print(f"正在加载CSV文件: {file_path}")
            df = pd.read_csv(file_path)
            dataset = Dataset.from_pandas(df)
            print(f"成功加载CSV，包含{len(dataset)}条记录")
            return dataset
            
        elif file_ext == '.parquet':
            print(f"正在加载Parquet文件: {file_path}")
            dataset = Dataset.from_parquet(file_path)
            print(f"成功加载Parquet，包含{len(dataset)}条记录")
            return dataset
        
        else:
            print(f"不支持的文件格式: {file_ext}")
            return None
            
    except Exception as e:
        print(f"加载本地数据集失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return None

class DatasetCleaner:
    """数据集清理器，提供多种去重和清理功能"""
    
    def __init__(self, threshold=0.8, ngram_size=3, enable_parallel=False, num_workers=None, enable_prefilter=True):
        """
        初始化清理器
        
        Args:
            threshold: Jaccard相似度阈值 (0.0-1.0)
            ngram_size: n-gram大小，默认3
            enable_parallel: 是否启用并行处理
            num_workers: 并行工作进程数，None则自动检测CPU核心数
            enable_prefilter: 是否启用n-gram预过滤优化
        """
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.enable_parallel = enable_parallel
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.enable_prefilter = enable_prefilter
        self.stats = {}
        
        print(f"清理器初始化:")
        print(f"  并行处理: {'启用' if enable_parallel else '禁用'}")
        if enable_parallel:
            print(f"  工作进程数: {self.num_workers}")
        print(f"  预过滤优化: {'启用' if enable_prefilter else '禁用'}")
        print(f"  相似度阈值: {threshold}")
        print(f"  N-gram大小: {ngram_size}")
        print()
    
    def load_dataset_safe(self, dataset_name, config=None, split=None):
        """
        安全加载数据集
        
        Args:
            dataset_name: 数据集名称
            config: 数据集配置（可选）
            split: 数据集分割（可选）
        """
        try:
            if config:
                print(f"正在加载数据集: {dataset_name} (配置: {config})")
                if split:
                    dataset = load_dataset(dataset_name, config, split=split)
                else:
                    dataset = load_dataset(dataset_name, config)
            else:
                print(f"正在加载数据集: {dataset_name}")
                if split:
                    dataset = load_dataset(dataset_name, split=split)
                else:
                    dataset = load_dataset(dataset_name)
            print(f"数据集加载成功")
            return dataset
        except Exception as e:
            print(f"数据集加载失败: {e}")
            return None
    
    def analyze_dataset(self, dataset, text_field='text'):
        """分析数据集基本信息"""
        print("\n数据集分析")
        print("=" * 50)
        
        if isinstance(dataset, DatasetDict):
            # 多分割数据集
            total_samples = 0
            for split_name, split_data in dataset.items():
                samples = len(split_data)
                total_samples += samples
                print(f"  {split_name}: {samples:,} 条")
                
                # 分析文本长度
                if samples > 0:
                    sample_texts = [split_data[i][text_field] for i in range(min(100, samples))]
                    avg_length = sum(len(text) for text in sample_texts) / len(sample_texts)
                    print(f"    平均文本长度: {avg_length:.0f} 字符")
                    
                    # 数据格式检查
                    self._check_data_format(split_data, text_field, 5)
            
            print(f"  总计: {total_samples:,} 条")
            
        else:
            # 单分割数据集
            samples = len(dataset)
            print(f"  样本数量: {samples:,} 条")
            
            # 分析文本长度
            if samples > 0:
                sample_texts = [dataset[i][text_field] for i in range(min(100, samples))]
                avg_length = sum(len(text) for text in sample_texts) / len(sample_texts)
                print(f"  平均文本长度: {avg_length:.0f} 字符")
                
                # 数据格式检查
                self._check_data_format(dataset, text_field, 5)
        
        print()
        return dataset
    
    def _check_data_format(self, dataset, text_field, num_samples=5):
        """检查数据格式"""
        print(f"  数据格式检查 (前{num_samples}条):")
        
        for i in range(min(num_samples, len(dataset))):
            try:
                item = dataset[i]
                print(f"    [{i}] 类型: {type(item)}")
                
                if isinstance(item, dict):
                    print(f"        字段: {list(item.keys())}")
                    if text_field in item:
                        text = item[text_field]
                        print(f"        文本类型: {type(text)}")
                        if isinstance(text, str):
                            print(f"        文本长度: {len(text)}")
                            print(f"        文本预览: {repr(text[:50])}...")
                        else:
                            print(f"        文本内容: {text}")
                    else:
                        print(f"        缺少字段 '{text_field}'")
                else:
                    print(f"        数据: {item}")
                    
            except Exception as e:
                print(f"    [{i}] 访问失败: {e}")
    
    def quick_filter(self, text1, text2):
        """快速预过滤，避免不必要的详细计算"""
        if not self.enable_prefilter:
            return True
            
        # 策略1：长度差异过大直接跳过
        len_ratio = len(text1) / len(text2) if len(text2) > 0 else float('inf') #inf表示正无穷大
        if len_ratio > 2.0 or len_ratio < 0.5:
            return False  # 长度差异超过2倍，不太可能相似
        
        # 策略2：n-gram集合大小预判
        ngrams1 = set(get_ngrams(normalize_text(text1), self.ngram_size))
        ngrams2 = set(get_ngrams(normalize_text(text2), self.ngram_size))
        
        size_ratio = len(ngrams1) / len(ngrams2) if len(ngrams2) > 0 else float('inf')
        if size_ratio > 1.5 or size_ratio < 0.67:
            return False  # n-gram集合大小差异过大
        
        # 策略3：快速交集估算
        intersection_size = len(ngrams1 & ngrams2)
        min_size = min(len(ngrams1), len(ngrams2))
        if min_size > 0 and intersection_size < min_size * 0.3:
            return False  # 交集太小，不可能达到高相似度
        
        return True  # 通过预过滤，值得详细计算
    
    def enhanced_jaccard_similarity(self, text1, text2):
        """带预过滤的Jaccard相似度计算"""
        # 快速预过滤
        if not self.quick_filter(text1, text2):
            return 0.0  # 直接返回低相似度
        
        # 执行完整计算
        return jaccard_similarity(text1, text2, self.ngram_size)
    
    def parallel_deduplicate_chunks(self, dataset, text_field):
        """并行分块去重 - 优化版本"""
        original_size = len(dataset)
        print(f"开始并行分块去重，原始数据: {original_size} 条")
        
        # 动态调整分块大小 - 优化为更小的块以提高响应速度
        if original_size < 1000:
            chunk_size = max(100, original_size // max(2, self.num_workers))
        elif original_size < 10000:
            chunk_size = max(200, original_size // self.num_workers)
        else:
            # 大数据集使用更小的块，但不少于500条
            chunk_size = max(500, min(2000, original_size // (self.num_workers * 2)))
        
        print(f"使用分块大小: {chunk_size} 条/块")
        
        chunks = []
        
        for i in range(0, original_size, chunk_size):
            end_idx = min(i + chunk_size, original_size)
            
            # 安全地转换数据 - 修复版本
            chunk_items = []
            try:
                print(f"准备块 {i}-{end_idx}...")
                
                # 直接按索引获取数据
                for j in range(i, end_idx):
                    try:
                        item = dataset[j]
                        # 确保item是字典且包含所需字段
                        if isinstance(item, dict) and text_field in item:
                            # 验证文本内容
                            text_content = item[text_field]
                            if text_content and isinstance(text_content, str) and len(text_content.strip()) > 0:
                                chunk_items.append(item)
                            else:
                                print(f"警告: 索引 {j} 文本内容为空或无效")
                        else:
                            print(f"警告: 索引 {j} 数据格式无效或缺少字段 '{text_field}'")
                    except Exception as e:
                        print(f"警告: 索引 {j} 数据访问失败: {e}")
                        continue
                        
                print(f"块 {i}-{end_idx}: 有效数据 {len(chunk_items)} 条")
                        
                if chunk_items:  # 只添加非空块
                    chunk_data = {
                        'chunk_items': chunk_items,
                        'params': (text_field, self.threshold, self.ngram_size),
                        'chunk_id': len(chunks),
                        'original_range': f"{i}-{end_idx}"
                    }
                    chunks.append(chunk_data)
                else:
                    print(f"警告: 块 {i}-{end_idx} 没有有效数据")
                    
            except Exception as e:
                print(f"块 {i}-{end_idx} 数据准备失败: {e}")
                # 尝试更简单的方法
                try:
                    simple_items = []
                    chunk_slice = dataset[i:end_idx]
                    for idx, item in enumerate(chunk_slice):
                        if hasattr(item, 'get') and item.get(text_field):
                            simple_items.append(dict(item))
                        elif isinstance(item, dict) and text_field in item:
                            simple_items.append(item)
                    
                    if simple_items:
                        chunk_data = {
                            'chunk_items': simple_items,
                            'params': (text_field, self.threshold, self.ngram_size),
                            'chunk_id': len(chunks),
                            'original_range': f"{i}-{end_idx}"
                        }
                        chunks.append(chunk_data)
                        print(f"块 {i}-{end_idx}: 使用简单方法获得 {len(simple_items)} 条数据")
                except Exception as e2:
                    print(f"块 {i}-{end_idx} 简单方法也失败: {e2}")
                    continue
        
        if not chunks:
            print("没有有效的数据块，回退到单线程处理")
            return deduplicate_dataset_jaccard(dataset, text_field, self.threshold, self.ngram_size)
        
        print(f"成功创建 {len(chunks)} 个有效数据块")
        
        try:
            # 使用线程池而不是进程池（避免序列化问题）
            cleaned_chunks = []
            
            if self.num_workers == 1 or len(chunks) == 1:
                # 单线程顺序处理
                for i, chunk_data in enumerate(chunks):
                    print(f"处理块 {i+1}/{len(chunks)} (范围: {chunk_data['original_range']})")
                    result = process_chunk_worker(chunk_data)
                    if result and len(result) > 0:
                        cleaned_chunks.append(result)
                        print(f"块 {i+1} 完成: 保留 {len(result)} 条")
                    else:
                        print(f"块 {i+1} 结果为空")
            else:
                # 多线程并行处理
                print(f"启动 {min(self.num_workers, len(chunks))} 个线程进行并行处理...")
                
                # 创建总体进度条
                overall_pbar = tqdm(
                    total=len(chunks),
                    desc="总体进度",
                    position=len(chunks),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] 完成块数'
                )
                
                with ThreadPoolExecutor(max_workers=min(self.num_workers, len(chunks))) as executor: # 创建一个线程池
                    '''
                    executor.submit() 将函数提交给线程池异步执行
                    返回一个 Future 对象，代表正在执行或将要执行的任务
                    process_chunk_worker 是要执行的工作函数
                    chunk_data 是传递给工作函数的参数
                    '''
                    future_to_chunk = {executor.submit(process_chunk_worker, chunk_data): i 
                                     for i, chunk_data in enumerate(chunks)}
                    
                    for future in future_to_chunk:
                        try:
                            result = future.result(timeout=600)  # 10分钟超时如果没获取到结果的话
                            chunk_idx = future_to_chunk[future]
                            if result and len(result) > 0:
                                cleaned_chunks.append(result)
                                print(f"完成块 {chunk_idx+1}/{len(chunks)}: 保留 {len(result)} 条")
                            else:
                                print(f"块 {chunk_idx+1} 结果为空")
                            overall_pbar.update(1)
                        except Exception as e:
                            chunk_idx = future_to_chunk[future]
                            print(f"块 {chunk_idx+1} 处理异常: {e}")
                            overall_pbar.update(1)
                
                overall_pbar.close()
            
            # 检查结果
            if not cleaned_chunks:
                print("所有并行块都失败了，回退到单线程处理")
                return deduplicate_dataset_jaccard(dataset, text_field, self.threshold, self.ngram_size)
            
            print(f"成功处理 {len(cleaned_chunks)}/{len(chunks)} 个块")
            
            # 合并结果
            merged_dataset = concatenate_datasets(cleaned_chunks)
            merged_size = len(merged_dataset)
            print(f"合并后数据: {merged_size} 条")
            
            if merged_size == 0:
                print("合并结果为空，回退到单线程处理")
                return deduplicate_dataset_jaccard(dataset, text_field, self.threshold, self.ngram_size)
            
            # 跨块去重（只对相对较小的合并数据进行）
            reduction_ratio = merged_size / original_size
            print(f"数据减少比例: {(1-reduction_ratio)*100:.1f}%")
            
            if reduction_ratio < 0.2:  # 如果已经去重了很多，进行跨块去重
                print("执行跨块去重...")
                final_dataset = deduplicate_dataset_jaccard(
                    merged_dataset, text_field, self.threshold, self.ngram_size
                )
                print(f"最终数据: {len(final_dataset)} 条")
                return final_dataset
            else:
                print("跳过跨块去重（数据减少不明显）")
                return merged_dataset
                
        except Exception as e:
            print(f"并行处理总体失败: {e}")
            print("回退到单线程处理...")
            return deduplicate_dataset_jaccard(dataset, text_field, self.threshold, self.ngram_size)
    
    def clean_single_split(self, dataset, text_field='text', mode='standard'):
        """清理单个数据集分割"""
        print(f"开始清理数据集 (模式: {mode})")
        print(f"参数: 阈值={self.threshold}, n-gram={self.ngram_size}")
        if self.enable_parallel:
            print(f"并行处理: {self.num_workers} 个工作进程")
        
        start_time = time.time()
        original_size = len(dataset)
        
        # 根据数据集大小智能选择处理方式
        if original_size > 200000 and mode == 'standard':
            print(f"数据集较大 ({original_size} 条)，建议使用快速模式")
            suggest_fast = input("是否切换到快速模式? (y/n, 默认y): ").strip().lower() != 'n'
            if suggest_fast:
                mode = 'fast'
        
        if mode == 'standard':
            if self.enable_parallel and original_size > 1000:
                try:
                    # 大数据集使用并行处理
                    cleaned_dataset = self.parallel_deduplicate_chunks(dataset, text_field)
                except Exception as e:
                    print(f"并行处理失败: {e}")
                    print("回退到单线程标准处理...")
                    cleaned_dataset = deduplicate_dataset_jaccard(
                        dataset, text_field, self.threshold, self.ngram_size
                    )
            else:
                # 标准Jaccard去重
                cleaned_dataset = deduplicate_dataset_jaccard(
                    dataset, text_field, self.threshold, self.ngram_size
                )
        elif mode == 'fast':
            # 快速采样去重 - 增大样本量以提高准确性
            sample_size = min(1000, original_size // 10)  # 10%的数据或最多1000条
            print(f"快速模式：使用 {sample_size} 条样本进行去重")
            cleaned_dataset = quick_jaccard_deduplicate(
                dataset, text_field, self.threshold, self.ngram_size, sample_size=sample_size
            )
        elif mode == 'parallel':
            try:
                # 强制并行模式
                cleaned_dataset = self.parallel_deduplicate_chunks(dataset, text_field)
            except Exception as e:
                print(f"强制并行模式失败: {e}")
                print("回退到单线程处理...")
                cleaned_dataset = deduplicate_dataset_jaccard(
                    dataset, text_field, self.threshold, self.ngram_size
                )
        else:
            raise ValueError(f"未知的清理模式: {mode}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 统计结果
        cleaned_size = len(cleaned_dataset)
        removed_count = original_size - cleaned_size
        removal_rate = (removed_count / original_size) * 100 if original_size > 0 else 0
        
        # 计算处理速度
        speed = original_size / processing_time if processing_time > 0 else 0
        
        self.stats = {
            'original_size': original_size,
            'cleaned_size': cleaned_size,
            'removed_count': removed_count,
            'removal_rate': removal_rate,
            'processing_time': processing_time,
            'processing_speed': speed
        }
        
        print(f"\n清理完成")
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"处理速度: {speed:.0f} 条/秒")
        print(f"原始数据: {original_size:,} 条")
        print(f"清理后: {cleaned_size:,} 条")
        print(f"移除数据: {removed_count:,} 条 ({removal_rate:.1f}%)")
        
        return cleaned_dataset
    
    def clean_multi_split(self, dataset_dict, text_field='text'):
        """清理多分割数据集"""
        print("开始跨分割清理")
        print(f"参数: 阈值={self.threshold}, n-gram={self.ngram_size}")
        
        start_time = time.time()
        
        # 使用跨分割去重
        cleaned_dataset = deduplicate_cross_splits_jaccard(
            dataset_dict, text_field, self.threshold, self.ngram_size
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 统计结果
        print(f"\n跨分割清理完成")
        print(f"处理时间: {processing_time:.2f} 秒")
        
        total_original = 0
        total_cleaned = 0
        
        for split_name in dataset_dict.keys():
            if split_name in cleaned_dataset:
                original = len(dataset_dict[split_name])
                cleaned = len(cleaned_dataset[split_name])
                removed = original - cleaned
                removal_rate = (removed / original) * 100 if original > 0 else 0
                
                total_original += original
                total_cleaned += cleaned
                
                print(f"  {split_name}: {original:,} -> {cleaned:,} 条 (减少 {removal_rate:.1f}%)")
        
        overall_removal_rate = ((total_original - total_cleaned) / total_original) * 100 if total_original > 0 else 0
        print(f"  总计: {total_original:,} -> {total_cleaned:,} 条 (减少 {overall_removal_rate:.1f}%)")
        
        return cleaned_dataset
    
    def save_cleaned_dataset(self, dataset, output_path, format='json'):
        """保存清理后的数据集"""
        try:
            print(f"保存数据集到: {os.path.abspath(output_path)}")
            print(f"数据集大小: {len(dataset)} 条")
            
            if format == 'json':
                dataset.to_json(output_path)
            elif format == 'csv':
                dataset.to_csv(output_path)
            elif format == 'parquet':
                dataset.to_parquet(output_path)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            # 验证文件是否成功创建
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"数据集保存成功")
                print(f"   文件大小: {file_size:,} 字节")
                print(f"   文件路径: {os.path.abspath(output_path)}")
            else:
                print("警告: 文件可能未成功创建")
            
        except Exception as e:
            print(f"保存失败: {e}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
    
    def sample_comparison(self, original_dataset, cleaned_dataset, text_field='text', num_samples=5):
        """展示清理前后的样本对比"""
        print(f"\n样本对比 (展示前{num_samples}条)")
        print("=" * 80)
        
        # 找出被移除的样本索引
        original_texts = [original_dataset[i][text_field] for i in range(len(original_dataset))]
        cleaned_texts = [cleaned_dataset[i][text_field] for i in range(len(cleaned_dataset))]
        
        print("保留的样本:")
        for i, text in enumerate(cleaned_texts[:num_samples]):
            print(f"{i+1}. {text[:100]}...")
        
        # 尝试找到一些被移除的样本
        removed_samples = []
        for text in original_texts[:50]:  # 检查前50个
            if text not in cleaned_texts:
                removed_samples.append(text)
                if len(removed_samples) >= num_samples:
                    break
        
        if removed_samples:
            print(f"\n被移除的样本:")
            for i, text in enumerate(removed_samples):
                print(f"{i+1}. {text[:100]}...")
    
    def get_cleaning_stats(self):
        """获取清理统计信息"""
        return self.stats

def show_dataset_info():
    """显示常见数据集的配置信息"""
    print("\n常见数据集配置信息:")
    print("-" * 40)
    print("分类任务推荐数据集 (原始未处理):")
    print("=" * 40)
    print("AG News (ag_news):")
    print("  - 用途: 新闻主题分类 (4类)")
    print("  - 特点: 原始新闻文本，120K训练样本")
    print("  - 字段: 'text' (新闻内容), 'label' (0-3)")
    print("  - 类别: World, Sports, Business, Sci/Tech")
    print("  - 优势: 未经预处理，适合去重测试")
    print()
    print("Yelp Reviews Full (yelp_review_full):")
    print("  - 用途: 情感分析 (5星评级)")
    print("  - 特点: 原始用户评论，650K训练样本")
    print("  - 字段: 'text' (评论内容), 'label' (0-4对应1-5星)")
    print("  - 优势: 包含口语化表达，适合去重测试")
    print()
    print("其他数据集:")
    print("=" * 40)
    print("Go Emotions (google-research-datasets/go_emotions):")
    print("  - 'simplified': 简化版本，包含基本情感标签")
    print("  - 'raw': 原始版本，包含详细情感分类")
    print("\nAmazon Reviews (amazon_reviews_multi):")
    print("  - 'en': 英语评论")
    print("  - 'fr': 法语评论")
    print("  - 'de': 德语评论")
    print("\nCommon Voice (mozilla-foundation/common_voice_11_0):")
    print("  - 'en': 英语")
    print("  - 'zh-CN': 中文")
    print("  - 'fr': 法语")
    print("-" * 40)

def interactive_cleaning():
    """交互式数据集清理"""
    print("交互式数据集清理工具")
    print("=" * 50)
    
    # 询问是否查看数据集信息
    show_info = input("是否查看常见数据集配置信息? (y/n, 默认n): ").strip().lower() == 'y'
    if show_info:
        show_dataset_info()
    
    # 选择数据集
    print("\n选择要清理的数据集:")
    print("分类任务推荐 (原始数据):")
    print("1. AG News - 新闻主题分类 (4类, 120K样本)")
    print("2. Yelp Reviews - 情感分析 (5星, 650K样本)")
    print()
    print("其他常用数据集:")
    print("3. IMDB电影评论数据集")
    print("4. Go Emotions情感数据集 (简化版)")
    print("5. Go Emotions情感数据集 (原始版)")
    print("6. 自定义数据集 (Hugging Face)")
    print("7. 本地数据集文件 (JSON/CSV/Parquet)")
    
    choice = input("请选择 (1-7): ").strip()
    
    if choice == '1':
        dataset_name = "ag_news"
        dataset_config = None
        text_field = 'text'
        print("AG News - 新闻主题分类数据集")
        print("包含4个类别: World, Sports, Business, Sci/Tech")
        print("120,000条训练数据 + 7,600条测试数据")
    elif choice == '2':
        dataset_name = "yelp_review_full"
        dataset_config = None
        text_field = 'text'
        print("Yelp Reviews - 5星情感分析数据集")
        print("包含1-5星用户评论，适合情感强度分析")
        print("650,000条训练数据 + 50,000条测试数据")
    elif choice == '3':
        dataset_name = "stanfordnlp/imdb"
        dataset_config = None
        text_field = 'text'
    elif choice == '4':
        dataset_name = "google-research-datasets/go_emotions"
        dataset_config = "simplified"
        text_field = 'text'
    elif choice == '5':
        dataset_name = "google-research-datasets/go_emotions"
        dataset_config = "raw"
        text_field = 'text'
    elif choice == '6':
        dataset_name = input("请输入数据集名称: ").strip()
        
        # 询问是否有配置
        has_config = input("数据集是否有配置参数? (y/n, 默认n): ").strip().lower() == 'y'
        if has_config:
            dataset_config = input("请输入配置名称 (如 'simplified', 'original' 等): ").strip()
            if not dataset_config:  # 如果输入为空
                dataset_config = None
        else:
            dataset_config = None
            
        text_field = input("请输入文本字段名 (默认'text'): ").strip() or 'text'
        
        # 设置加载模式为Hugging Face
        load_mode = 'huggingface'
    elif choice == '7':
        # 本地数据集文件
        print("\n本地数据集加载")
        print("-" * 40)
        
        # 默认目录
        default_dir = r"D:\桌面\Deduplication_framework\text\dataset"
        print(f"默认数据集目录: {default_dir}")
        
        # 询问文件路径
        use_default = input("使用默认目录? (y/n, 默认y): ").strip().lower() != 'n'
        
        if use_default:
            # 列出默认目录中的文件
            import os
            try:
                files = [f for f in os.listdir(default_dir) 
                         if os.path.isfile(os.path.join(default_dir, f)) and 
                         f.endswith(('.json', '.csv', '.parquet'))]
                
                if files:
                    print("\n可用数据集文件:")
                    for i, file in enumerate(files, 1):
                        print(f"{i}. {file}")
                    
                    file_choice = input("请选择文件编号 (或输入完整文件名): ").strip()
                    
                    try:
                        file_idx = int(file_choice) - 1
                        if 0 <= file_idx < len(files):
                            file_path = os.path.join(default_dir, files[file_idx])
                        else:
                            print("无效的文件编号，请输入文件路径")
                            file_path = input("请输入文件完整路径: ").strip()
                    except ValueError:
                        # 用户输入了文件名而不是编号
                        if file_choice in files:
                            file_path = os.path.join(default_dir, file_choice)
                        else:
                            file_path = input("未找到文件，请输入完整路径: ").strip()
                else:
                    print(f"默认目录中没有找到JSON/CSV/Parquet文件")
                    file_path = input("请输入文件完整路径: ").strip()
            except Exception as e:
                print(f"读取默认目录失败: {e}")
                file_path = input("请输入文件完整路径: ").strip()
        else:
            file_path = input("请输入文件完整路径: ").strip()
        
        # 文本字段
        text_field = input("请输入文本字段名 (默认'text'): ").strip() or 'text'
        
        # 设置加载模式为本地文件
        load_mode = 'local'
        dataset_name = file_path
        dataset_config = None
    else:
        print("无效选择")
        return
    
    # 设置参数
    print(f"\n设置清理参数:")
    threshold = float(input("相似度阈值 (0.0-1.0, 默认0.8): ") or "0.8")
    ngram_size = int(input("n-gram大小 (默认3): ") or "3")
    
    # 设置优化选项
    print(f"\n优化选项:")
    enable_parallel = input("启用并行处理? (y/n, 默认y): ").strip().lower() != 'n'
    enable_prefilter = input("启用预过滤优化? (y/n, 默认y): ").strip().lower() != 'n'
    
    # 设置输出路径
    print(f"\n保存文件设置:")
    
    # 根据数据集类型生成默认输出文件名
    if choice == '7':  # 本地文件
        import os
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        default_output = f"{base_filename}_clean_t{threshold}.json"
    else:  # Hugging Face数据集
        default_output = f"{dataset_name.split('/')[-1]}_clean_t{threshold}.json"
    
    output_path = input(f"输出文件路径 (默认'{default_output}'): ").strip() or default_output
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {os.path.abspath(output_dir)}")
    
    # 确保文件有扩展名
    if not output_path.endswith('.json'):
        output_path += '.json'
    print(f"完整输出路径: {os.path.abspath(output_path)}")
    
    if enable_parallel:
        max_workers = mp.cpu_count()
        num_workers = int(input(f"并行工作进程数 (1-{max_workers}, 默认{max_workers-1}): ") or str(max_workers-1))
        num_workers = max(1, min(num_workers, max_workers))
    else:
        num_workers = 1
    
    # 选择清理模式
    print(f"\n选择清理模式:")
    print("1. 标准模式 (自动选择串行/并行)")
    print("2. 快速模式 (采样优化)")
    print("3. 跨分割模式 (推荐)")
    print("4. 强制并行模式 (大数据集)")
    
    mode_choice = input("请选择 (1/2/3/4): ").strip()
    
    # 创建清理器
    cleaner = DatasetCleaner(
        threshold=threshold, 
        ngram_size=ngram_size,
        enable_parallel=enable_parallel,
        num_workers=num_workers,
        enable_prefilter=enable_prefilter
    )
    
    # 加载数据集
    if choice == '7' or load_mode == 'local':
        print(f"\n正在加载本地数据集: {dataset_name}")
        dataset = load_local_dataset(dataset_name, text_field)
    else:
        # 使用Hugging Face加载
        dataset = None
        retry_count = 0
        max_retries = 3
        
        while dataset is None and retry_count < max_retries:
            dataset = cleaner.load_dataset_safe(dataset_name, config=dataset_config)
            
            if dataset is None:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\n加载失败，第 {retry_count}/{max_retries} 次重试")
                    print("请检查数据集名称和配置是否正确")
                    
                    # 询问是否修改参数或尝试本地文件
                    action = input("1. 修改参数  2. 尝试加载本地文件  3. 退出  选择(1/2/3): ").strip()
                    
                    if action == '1':
                        new_dataset_name = input(f"数据集名称 (当前: {dataset_name}): ").strip()
                        if new_dataset_name:
                            dataset_name = new_dataset_name
                        
                        if dataset_config:
                            new_config = input(f"配置名称 (当前: {dataset_config}, 留空表示无配置): ").strip()
                            dataset_config = new_config if new_config else None
                        else:
                            has_config = input("是否添加配置参数? (y/n): ").strip().lower() == 'y'
                            if has_config:
                                dataset_config = input("请输入配置名称: ").strip()
                    
                    elif action == '2':
                        # 切换到本地文件加载
                        print("\n尝试加载本地文件...")
                        file_path = input("请输入本地文件路径: ").strip()
                        dataset = load_local_dataset(file_path, text_field)
                        break
                    else:
                        print("退出程序")
                        return
                else:
                    print("达到最大重试次数，尝试加载本地文件...")
                    file_path = input("请输入本地文件路径 (直接回车退出): ").strip()
                    if file_path:
                        dataset = load_local_dataset(file_path, text_field)
                    else:
                        return
    
    if dataset is None:
        print("数据集加载失败，程序退出")
        return
    
    # 分析数据集
    cleaner.analyze_dataset(dataset, text_field)
    
    # 执行清理
    if mode_choice == '1':
        # 标准模式 - 清理训练集
        if isinstance(dataset, DatasetDict) and 'train' in dataset:
            cleaned_data = cleaner.clean_single_split(dataset['train'], text_field, 'standard')
        else:
            cleaned_data = cleaner.clean_single_split(dataset, text_field, 'standard')
            
    elif mode_choice == '2':
        # 快速模式
        if isinstance(dataset, DatasetDict) and 'train' in dataset:
            cleaned_data = cleaner.clean_single_split(dataset['train'], text_field, 'fast')
        else:
            cleaned_data = cleaner.clean_single_split(dataset, text_field, 'fast')
            
    elif mode_choice == '3':
        # 跨分割模式
        if isinstance(dataset, DatasetDict):
            cleaned_data = cleaner.clean_multi_split(dataset, text_field)
        else:
            print("单分割数据集，使用标准模式")
            cleaned_data = cleaner.clean_single_split(dataset, text_field, 'standard')
            
    elif mode_choice == '4':
        # 强制并行模式
        if isinstance(dataset, DatasetDict) and 'train' in dataset:
            cleaned_data = cleaner.clean_single_split(dataset['train'], text_field, 'parallel')
        else:
            cleaned_data = cleaner.clean_single_split(dataset, text_field, 'parallel')
    else:
        print("无效选择")
        return
    
    # 样本对比
    if not isinstance(cleaned_data, DatasetDict):
        original_for_comparison = dataset['train'] if isinstance(dataset, DatasetDict) and 'train' in dataset else dataset
        cleaner.sample_comparison(original_for_comparison, cleaned_data, text_field)
    
    # 保存清理后的数据集
    print(f"\n开始保存清理结果...")
    if isinstance(cleaned_data, DatasetDict):
        # 保存多分割数据集
        print("检测到多分割数据集，分别保存各个分割...")
        for split_name, split_data in cleaned_data.items():
            # 生成分割特定的文件路径
            base_name = os.path.splitext(output_path)[0]  # 去掉扩展名
            extension = os.path.splitext(output_path)[1] or '.json'  # 获取扩展名，默认.json
            split_path = f"{base_name}_{split_name}{extension}"
            
            print(f"保存 {split_name} 分割到: {split_path}")
            cleaner.save_cleaned_dataset(split_data, split_path)
    else:
        # 保存单分割数据集
        print(f"保存数据集到: {output_path}")
        cleaner.save_cleaned_dataset(cleaned_data, output_path)
    
    print(f"\n数据集清理完成!")

def batch_cleaning_example():
    """批量清理示例 - 展示优化效果"""
    print("批量清理示例 - 性能对比")
    print("=" * 50)
    
    # 加载IMDB数据集
    print("正在加载IMDB数据集...")
    dataset = load_dataset("stanfordnlp/imdb")
    if dataset is None:
        return
    
    print("\n数据集信息:")
    print(f"  训练集: {len(dataset['train']):,} 条")
    print(f"  测试集: {len(dataset['test']):,} 条")
    
    # 性能对比测试
    print("\n=== 性能对比测试 ===")
    
    # 1. 传统方法 (小样本)
    print("\n1. 传统方法测试 (前1000条):")
    small_dataset = dataset['train'].select(range(1000))
    
    cleaner_traditional = DatasetCleaner(
        threshold=0.8, ngram_size=3, 
        enable_parallel=False, enable_prefilter=False
    )
    start_time = time.time()
    cleaned_traditional = cleaner_traditional.clean_single_split(small_dataset, 'text', 'standard')
    traditional_time = time.time() - start_time
    
    # 2. 优化方法 (小样本)
    print("\n2. 优化方法测试 (前1000条):")
    cleaner_optimized = DatasetCleaner(
        threshold=0.8, ngram_size=3, 
        enable_parallel=True, enable_prefilter=True
    )
    start_time = time.time()
    cleaned_optimized = cleaner_optimized.clean_single_split(small_dataset, 'text', 'standard')
    optimized_time = time.time() - start_time
    
    # 3. 完整数据集测试 (仅优化方法)
    print("\n3. 完整数据集测试 (仅优化方法):")
    cleaner_full = DatasetCleaner(
        threshold=0.8, ngram_size=3, 
        enable_parallel=True, enable_prefilter=True
    )
    
    # 执行跨分割清理
    cleaned_dataset = cleaner_full.clean_multi_split(dataset, 'text')
    
    # 性能总结
    print("\n" + "="*60)
    print("性能对比总结:")
    print(f"传统方法 (1K样本): {traditional_time:.2f}秒")
    print(f"优化方法 (1K样本): {optimized_time:.2f}秒")
    if traditional_time > 0:
        speedup = traditional_time / optimized_time
        print(f"性能提升: {speedup:.1f}x 倍")
    
    print(f"\n最终清理效果:")
    print(f"训练集: {len(dataset['train']):,} -> {len(cleaned_dataset['train']):,} 条")
    print(f"测试集: {len(dataset['test']):,} -> {len(cleaned_dataset['test']):,} 条")
    
    total_original = len(dataset['train']) + len(dataset['test'])
    total_cleaned = len(cleaned_dataset['train']) + len(cleaned_dataset['test'])
    reduction_rate = ((total_original - total_cleaned) / total_original) * 100
    print(f"总体数据减少: {reduction_rate:.1f}%")
    
    return cleaned_dataset

if __name__ == "__main__":
    # Windows系统需要这个保护以避免multiprocessing问题
    switcher = EnvManager()
    res = switcher.setup_text_env()
    if res:
        mp.set_start_method('spawn', force=True) # windows需要
        
        print("数据集清理工具")
        print("基于Jaccard相似度算法的智能去重")
        print("=" * 60)
        
        print("\n选择运行模式:")
        print("1. 交互式清理")
        print("2. 批量清理示例")
        print("3. 查看数据集配置信息")
        
        mode = input("请选择 (1/2/3): ").strip()
        
        if mode == '1':
            interactive_cleaning()
        elif mode == '2':
            batch_cleaning_example()
        elif mode == '3':
            show_dataset_info()
            print("\n")
            # 显示完信息后继续选择
            choice = input("是否继续运行交互式清理? (y/n): ").strip().lower()
            if choice == 'y':
                interactive_cleaning()
        else:
            print("无效选择，运行交互式清理...")
            interactive_cleaning()
    else:
        print("环境初始化失败，程序退出")