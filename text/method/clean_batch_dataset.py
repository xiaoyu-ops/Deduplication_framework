from datasets import load_dataset
from text.method.dataset.clean_the_dataset import DatasetCleaner
from text.method.dataset.jaccard_deduplication import clear_global_memory, quick_jaccard_deduplicate
import time
import os
from tqdm import tqdm

def chunked_deduplication(dataset, text_field, threshold, ngram_size, chunk_size=5000, sample_size=1000):
    """
    分块+采样去重，避免O(n²)复杂度问题
    
    Args:
        dataset: 数据集
        text_field: 文本字段
        threshold: 相似度阈值
        ngram_size: n-gram大小
        chunk_size: 分块大小
        sample_size: 采样大小
    """
    print(f"开始分块去重: 总数据{len(dataset)}条, 分块大小{chunk_size}, 采样大小{sample_size}")
    
    all_kept_indices = []
    processed_count = 0
    
    # 分块处理
    for chunk_start in tqdm(range(0, len(dataset), chunk_size), desc="分块处理"):
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        chunk = dataset.select(range(chunk_start, chunk_end))
        
        print(f"处理分块 {chunk_start}-{chunk_end} ({len(chunk)} 条)")
        
        # 对每个分块使用快速去重（采样优化）
        deduplicated_chunk = quick_jaccard_deduplicate(
            chunk, text_field, threshold, ngram_size, sample_size
        )
        
        # 记录保留的全局索引
        if len(deduplicated_chunk) > 0:
            # 获取分块内的保留索引，然后转换为全局索引
            chunk_kept_indices = list(range(len(deduplicated_chunk)))
            global_indices = [chunk_start + i for i in chunk_kept_indices]
            all_kept_indices.extend(global_indices)
        
        processed_count += len(chunk)
        print(f"分块完成: 保留 {len(deduplicated_chunk)}/{len(chunk)} 条")
    
    # 构建最终结果
    if all_kept_indices:
        final_dataset = dataset.select(all_kept_indices)
        print(f"分块去重完成: {len(dataset)} -> {len(final_dataset)} 条")
        return final_dataset
    else:
        print("警告: 所有数据都被去重了")
        return dataset.select([])  # 返回空数据集

# 加载数据集
dataset = load_dataset("ag_news")

# 定义阈值
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 创建输出目录
os.makedirs("batch_cleaned_datasets", exist_ok=True)

for threshold in thresholds:
    print(f"\n{'='*60}")
    print(f"处理阈值: {threshold}")
    print(f"{'='*60}")
    
    # 清理全局变量（开始前先清理）
    clear_global_memory()
    
    start_time = time.time()
    
    # 分别对训练集和测试集进行分块去重
    print("处理训练集...")
    cleaned_train = chunked_deduplication(
        dataset['train'], 
        'text', 
        threshold, 
        ngram_size=3, 
        chunk_size=5000,   # 5K一块
        sample_size=1000   # 每块内最多比较1000个
    )
    
    print("\n处理测试集...")
    cleaned_test = chunked_deduplication(
        dataset['test'], 
        'text', 
        threshold, 
        ngram_size=3, 
        chunk_size=2000,   # 测试集较小，2K一块
        sample_size=500    # 每块内最多比较500个
    )
    
    processing_time = time.time() - start_time
    
    # 保存结果
    train_path = f"batch_cleaned_datasets/ag_news_train_threshold_{threshold}.json"
    test_path = f"batch_cleaned_datasets/ag_news_test_threshold_{threshold}.json"
    
    cleaned_train.to_json(train_path)
    cleaned_test.to_json(test_path)
    
    # 统计信息
    train_reduction = (len(dataset['train']) - len(cleaned_train)) / len(dataset['train']) * 100
    test_reduction = (len(dataset['test']) - len(cleaned_test)) / len(dataset['test']) * 100
    total_processed = len(dataset['train']) + len(dataset['test'])
    speed = total_processed / processing_time if processing_time > 0 else 0
    
    print(f"\n阈值 {threshold} 处理完成!")
    print(f"训练集: {len(dataset['train'])} -> {len(cleaned_train)} 条 (减少 {train_reduction:.1f}%)")
    print(f"测试集: {len(dataset['test'])} -> {len(cleaned_test)} 条 (减少 {test_reduction:.1f}%)")
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"处理速度: {speed:.0f} 条/秒")
    
    # 清理全局变量（结束后再清理）
    clear_global_memory()

print("\n🎉 所有阈值处理完成！")
