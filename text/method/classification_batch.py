from text.method.dataset.classification_comparison import load_cleaned_dataset, train_test_split, train_model, evaluate_model, save_threshold_results
from datasets import load_dataset
import re
import torch
import numpy as np
import random
import os


def main():
    np.random.seed(42)  # 确保结果可复现
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_results = []  # 存储所有阈值的结果
    
    # 确保结果目录存在
    os.makedirs('threshold_results', exist_ok=True)
    
    for threshold in thresholds:
        try:
            print(f"\n{'='*60}")
            print(f"处理阈值: {threshold}")
            print(f"{'='*60}")
            
            print("加载去重数据集...")
            cleaned_texts, cleaned_labels = load_cleaned_dataset(threshold)
            if not cleaned_texts:
                print(f"阈值 {threshold} 数据集加载失败或为空，跳过该阈值")
                continue
                
            dataset_size = len(cleaned_texts)
            print(f"数据集大小: {dataset_size} 条")
            
            # 分割去重数据集
            cleaned_train_texts, cleaned_test_texts, cleaned_train_labels, cleaned_test_labels = train_test_split(
                cleaned_texts, cleaned_labels, test_size=0.2, random_state=42, stratify=cleaned_labels
            )
            
            print(f"训练集: {len(cleaned_train_texts)} 条")
            print(f"测试集: {len(cleaned_test_texts)} 条")
            
            print("\n开始训练模型...")
            cleaned_trainer, cleaned_tokenizer, cleaned_time, cleaned_callback = train_model(
                cleaned_train_texts, cleaned_train_labels,
                cleaned_test_texts, cleaned_test_labels
            )
            
            print("\n评估模型性能...")
            cleaned_results, cleaned_report, _ = evaluate_model(
                cleaned_trainer, cleaned_test_texts, cleaned_test_labels, cleaned_tokenizer
            )
            
            print(f"\n阈值 {threshold} 训练完成:")
            print(f"准确率: {cleaned_results['eval_accuracy']:.4f}")
            print(f"F1分数: {cleaned_results['eval_f1']:.4f}")
            print(f"训练时间: {cleaned_time:.2f} 秒")
            
            # 保存单个阈值的详细结果
            threshold_data = save_threshold_results(
                threshold, cleaned_results, cleaned_time, cleaned_callback, dataset_size
            )
            all_results.append(threshold_data)
            
        except Exception as e:
            print(f"处理阈值 {threshold} 时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存所有阈值的汇总结果
    if all_results:
        summary_results = {
            'experiment_summary': {
                'total_thresholds_processed': len(all_results),
                'thresholds': [r['threshold'] for r in all_results],
                'best_threshold': max(all_results, key=lambda x: x['model_performance']['accuracy'])['threshold'],
                'best_accuracy': max(r['model_performance']['accuracy'] for r in all_results),
                'best_f1': max(r['model_performance']['f1_score'] for r in all_results)
            },
            'detailed_results': all_results
        }
        
        import json
        with open('threshold_results\\all_thresholds_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
        # 生成汇总报告
        with open('threshold_results\\summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("所有阈值实验汇总报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总共处理阈值数: {len(all_results)}\n")
            f.write(f"最佳阈值: {summary_results['experiment_summary']['best_threshold']}\n")
            f.write(f"最高准确率: {summary_results['experiment_summary']['best_accuracy']:.4f}\n")
            f.write(f"最高F1分数: {summary_results['experiment_summary']['best_f1']:.4f}\n\n")
            
            f.write("各阈值详细结果:\n")
            f.write("-" * 80 + "\n")
            for result in all_results:
                f.write(f"阈值 {result['threshold']:<4} | ")
                f.write(f"准确率: {result['model_performance']['accuracy']:.4f} | ")
                f.write(f"F1: {result['model_performance']['f1_score']:.4f} | ")
                f.write(f"数据量: {result['dataset_info']['size']:>6} | ")
                f.write(f"训练时间: {result['training_info']['training_time_seconds']:>6.1f}s\n")
        
        print(f"\n 所有阈值处理完成!")
        print(f"处理了 {len(all_results)} 个阈值")
        print(f"最佳阈值: {summary_results['experiment_summary']['best_threshold']}")
        print(f"汇总结果保存在: threshold_results\\all_thresholds_summary.json")
        print(f"汇总报告保存在: threshold_results\\summary_report.txt")
    else:
        print("没有成功处理任何阈值")


if __name__ == "__main__":
    main()
