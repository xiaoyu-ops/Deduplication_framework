# 基于Jaccard相似度的数据集去重工具
import re
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import gc

# 全局变量
global_kept_ngrams = []

def normalize_text(text):
    """标准化文本用于比较"""
    if not isinstance(text, str):  # 确保输入的是字符串
        text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text) #删除所有不是英文字母和汉字的字符和下划线以及空白字符
    return text.strip() # 移除开头和结尾的空白字符

def clear_global_memory():
    """彻底清理全局内存变量"""
    global global_kept_ngrams
    global_kept_ngrams.clear()
    global_kept_ngrams = []
    gc.collect()
    print("全局内存变量已彻底清理")

def clear_all_memory():
    """清理所有可能的内存累积"""
    global global_kept_ngrams
    
    # 清理全局变量
    global_kept_ngrams.clear()
    global_kept_ngrams = []
    
    # 强制垃圾回收
    gc.collect()
    
    print("所有内存变量已彻底清理")

def get_ngrams(text, n=3):
    """生成文本的n-gram集合"""
    normalized = normalize_text(text)
    words = normalized.split() # 利用空白格来分词
    
    # 字符级n-gram
    char_ngrams = set()
    for i in range(len(normalized) - n + 1):
        char_ngrams.add(normalized[i:i+n]) #n是3，就是每3个字符这样加入
    
    # 词级n-gram
    word_ngrams = set()
    for i in range(len(words) - n + 1):
        word_ngrams.add(' '.join(words[i:i+n])) # 把三个词连接在一起再加入

    # 最后返回字符gram和词gram的并集
    return char_ngrams | word_ngrams

def jaccard_similarity(text1, text2, n=3):
    """计算两个文本的Jaccard相似度"""
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if len(ngrams1) == 0 and len(ngrams2) == 0:
        return 1.0
    
    intersection = len(ngrams1 & ngrams2) # 交集
    union = len(ngrams1 | ngrams2) # 并集
    # J(A,B) = |A ∩ B| / |A ∪ B| = intersection / union
    return intersection / union if union > 0 else 0.0

def deduplicate_dataset_jaccard(dataset, text_field='text', threshold=0.8, ngram_size=3):
    """
    使用Jaccard相似度对数据集进行去重
    
    Args:
        dataset: HuggingFace Dataset对象
        text_field: 包含文本的字段名
        threshold: 相似度阈值 (0.0-1.0)
        ngram_size: n-gram大小
    
    Returns:
        去重后的数据集
    """
    print(f"开始Jaccard去重，原始数据: {len(dataset)} 条")
    print(f"相似度阈值: {threshold:.2f}, n-gram大小: {ngram_size}")

    keep_indices = [] # 保持的索引
    kept_texts = [] # 保持的文本
    
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Jaccard去重进度"): # i接受索引 item接受数据
        text = item[text_field] #找到对应的文本列
        is_duplicate = False
        
        # 与已保留的文本比较
        for kept_text in kept_texts:
            similarity = jaccard_similarity(text, kept_text, ngram_size) # 将当前文本与已保留文本进行比较
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep_indices.append(i)
            kept_texts.append(text)
    
    undeduplicated = dataset.select(keep_indices)
    removed = len(dataset) - len(undeduplicated)
    
    print(f"去重完成，保留: {len(undeduplicated)} 条")
    print(f"移除重复: {removed} 条 ({removed/len(dataset)*100:.1f}%)")

    return undeduplicated #最后返回去重后的数据集
def deduplicate_cross_splits_jaccard(dataset_dict, text_field='text', threshold=0.8, ngram_size=3):
    """
    跨分割Jaccard去重（train/test/validation之间）
    优先保留train数据
    
    Args:
        dataset_dict: 包含多个分割的数据集字典
        text_field: 文本字段名
        threshold: 相似度阈值
        ngram_size: n-gram大小
    
    Returns:
        去重后的数据集字典
    """
    global global_kept_ngrams
    
    print("开始跨分割Jaccard去重...")
    print(f"相似度阈值: {threshold:.2f}, n-gram大小: {ngram_size}")
    print(f"当前global_kept_ngrams长度: {len(global_kept_ngrams)}")
    
    result = {}
    
    # 处理顺序：train -> test -> 其他
    splits = ['train', 'test'] + [k for k in dataset_dict.keys() if k not in ['train', 'test']]
    
    for split_name in splits:
        if split_name not in dataset_dict:
            continue
            
        dataset = dataset_dict[split_name]
        print(f"处理 {split_name} 分割 (原始: {len(dataset)} 条)")
        
        keep_indices = []
        
        # 添加进度显示
        from tqdm import tqdm
        
        for i, item in tqdm(enumerate(dataset), total=len(dataset), desc=f"处理{split_name}分割"):
            text = item[text_field]
            current_ngrams = get_ngrams(normalize_text(text), ngram_size)
            is_duplicate = False
            
            # 与已保留的n-gram集合比较（而不是原始文本）
            for kept_ngrams in global_kept_ngrams:
                similarity = len(current_ngrams & kept_ngrams) / len(current_ngrams | kept_ngrams) if (current_ngrams | kept_ngrams) else 0
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
                global_kept_ngrams.append(current_ngrams)
                
                # 内存管理：限制保留的n-gram数量
                if len(global_kept_ngrams) > 50000:  # 限制最大数量
                    # 保留最近的一半
                    global_kept_ngrams = global_kept_ngrams[-25000:]
                    print(f"内存优化：保留最近25000个n-gram集合")
        
        result[split_name] = dataset.select(keep_indices)
        print(f"{split_name} 去重后: {len(result[split_name])} 条")
        
        # 分割处理完成后进行垃圾回收
        gc.collect()
    
    return result

def quick_jaccard_deduplicate(dataset, text_field='text', threshold=0.8, ngram_size=3, sample_size=100):
    """
    快速Jaccard去重（采样优化版本）
    
    Args:
        dataset: HuggingFace Dataset对象
        text_field: 文本字段名
        threshold: 相似度阈值
        ngram_size: n-gram大小
        sample_size: 每轮比较的采样大小
    
    Returns:
        去重后的数据集
    """
    print(f"开始快速Jaccard去重，原始数据: {len(dataset)} 条")
    print(f"相似度阈值: {threshold:.2f}, 采样大小: {sample_size}")
    
    keep_indices = []
    kept_ngrams = []  # 存储已保留文本的n-gram集合
    
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="快速Jaccard去重"):
        text = item[text_field]
        current_ngrams = get_ngrams(text, ngram_size)
        is_duplicate = False
        
        '''
        这种设计基于一个重要的假设：最近添加的文本与当前文本更可能具有相似性。在许多实际场景中
        ，数据集往往具有某种时间或主题聚集性，相邻的文本内容更容易重复。因此，只与最近的 sample_size 个已保留文本
        进行比较，既能有效检测重复，又能显著减少计算量。
        '''
        # 与最近的sample_size个文本比较
        start_idx = max(0, len(kept_ngrams) - sample_size)
        for j in range(start_idx, len(kept_ngrams)):
            kept_ngrams_set = kept_ngrams[j]
            
            # 计算Jaccard相似度
            intersection = len(current_ngrams & kept_ngrams_set)
            union = len(current_ngrams | kept_ngrams_set)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep_indices.append(i)
            kept_ngrams.append(current_ngrams)
    
    deduplicated = dataset.select(keep_indices)
    removed = len(dataset) - len(deduplicated)
    
    print(f"去重完成，保留: {len(deduplicated)} 条")
    print(f"移除重复: {removed} 条 ({removed/len(dataset)*100:.1f}%)")
    
    return deduplicated

def main():
    """主函数 - 演示Jaccard相似度去重"""
    print("基于Jaccard相似度的数据集去重工具")
    print("=" * 50)
    
    try:
        # 加载IMDB数据集
        print("正在加载IMDB数据集...")
        ds = load_dataset("stanfordnlp/imdb")
        
        print(f"原始数据集信息:")
        print(f"  训练集: {len(ds['train'])} 条")
        print(f"  测试集: {len(ds['test'])} 条")
        
        # 参数设置
        print(f"\n参数设置:")
        threshold = float(input("请输入相似度阈值 (0.0-1.0, 默认0.8): ") or "0.8")
        ngram_size = int(input("请输入n-gram大小 (默认3): ") or "3")
        
        print(f"\n选择去重方式:")
        print("1. 对训练集进行Jaccard去重")
        print("2. 对测试集进行Jaccard去重")
        print("3. 跨分割Jaccard去重（推荐）")
        print("4. 快速Jaccard去重（采样优化）")
        print("5. 小样本测试（前100条）")
        
        choice = input("请输入选择 (1/2/3/4/5): ").strip()
        
        if choice == '1':
            print(f"\n对训练集进行Jaccard去重...")
            train_dedup = deduplicate_dataset_jaccard(ds['train'], 'text', threshold, ngram_size)
            
        elif choice == '2':
            print(f"\n对测试集进行Jaccard去重...")
            test_dedup = deduplicate_dataset_jaccard(ds['test'], 'text', threshold, ngram_size)
            
        elif choice == '3':
            print(f"\n进行跨分割Jaccard去重...")
            dedup_ds = deduplicate_cross_splits_jaccard(ds, 'text', threshold, ngram_size)
            
            print(f"\n跨分割去重结果:")
            for split, data in dedup_ds.items():
                original_size = len(ds[split])
                new_size = len(data)
                reduction = (original_size - new_size) / original_size * 100
                print(f"  {split}: {original_size} -> {new_size} 条 (减少 {reduction:.1f}%)")
                
        elif choice == '4':
            print(f"\n进行快速Jaccard去重...")
            sample_size = int(input("请输入采样大小 (默认100): ") or "100")
            train_dedup = quick_jaccard_deduplicate(ds['train'], 'text', threshold, ngram_size, sample_size)
            
        elif choice == '5':
            print(f"\n小样本测试...")
            # 取前100条进行测试
            small_dataset = ds['train'].select(range(100))
            dedup_result = deduplicate_dataset_jaccard(small_dataset, 'text', threshold, ngram_size)
            
            print(f"\n测试结果:")
            print(f"  原始: 100 条")
            print(f"  去重后: {len(dedup_result)} 条")
            print(f"  重复率: {(100 - len(dedup_result))}%")
            
        else:
            print("无效选择...")
    
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        print("请确保网络连接正常且已安装datasets库")

if __name__ == "__main__":
    main()
