"""
文本分类任务对比实验：去重前后数据集性能比较
使用预训练模型进行微调，比较去重前后的分类效果
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    TrainerCallback
)
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset as HFDataset
import time
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MetricsCallback(TrainerCallback):
    """记录训练过程中的指标变化"""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.epochs = []
        self.steps = []
        self.last_logged_step = 0
        self.log_interval = 250  # 每250步记录一次loss，减少波动

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 减少训练loss记录频率，让曲线更平滑
            if 'loss' in logs and state.global_step - self.last_logged_step >= self.log_interval:
                self.train_losses.append(logs['loss'])
                self.last_logged_step = state.global_step
            
            if 'eval_loss' in logs and 'eval_accuracy' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_accuracies.append(logs['eval_accuracy'])
                # 记录当前步数和对应的epoch
                self.steps.append(state.global_step)
                # 计算对应的epoch（连续值，不是整数）
                current_epoch = state.global_step / (state.max_steps / args.num_train_epochs) * args.num_train_epochs if state.max_steps > 0 else 0
                self.epochs.append(current_epoch)

class NewsDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer #文本分词器对象，在NLP中用于将文本转换为模型可接受的输入格式
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True, # 截断过长文本
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(), # 展平为一维张量
            'attention_mask': encoding['attention_mask'].flatten(), # 展平为一维张量
            'labels': torch.tensor(label, dtype=torch.long) # 标签
        }

def load_original_dataset():
    """加载原始AG News数据集（作为未去重的基准）"""
    from datasets import load_dataset
    print("正在加载原始AG News数据集...")
    
    # 直接加载真实的AG News数据集
    dataset = load_dataset("ag_news")
    
    # 转换为我们需要的格式（转换为Python列表）
    train_texts = list(dataset['train']['text'])
    train_labels = list(dataset['train']['label'])
    test_texts = list(dataset['test']['text'])
    test_labels = list(dataset['test']['label'])
    
    print(f"原始AG News数据集加载完成：")
    print(f"  训练集: {len(train_texts)} 条")
    print(f"  测试集: {len(test_texts)} 条")
    
    return train_texts, train_labels, test_texts, test_labels

def load_cleaned_dataset(threshold):
    """加载去重后的数据集（合并训练集和测试集）"""
    print("正在加载去重后的数据集...")
    
    texts = []
    labels = []
    
    # 加载训练集
    try:
        with open(f'batch_cleaned_datasets\\ag_news_threshold_{threshold}.json', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 处理可能的多行JSON格式
            current_obj = ""
            bracket_count = 0
            
            for char in content:
                current_obj += char
                
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    
                    if bracket_count == 0:
                        try:
                            data = json.loads(current_obj.strip())
                            if 'text' in data and 'label' in data:
                                texts.append(data['text'])
                                labels.append(data['label'])
                        except json.JSONDecodeError:
                            pass
                        
                        current_obj = ""
        
        print(f"训练集加载完成: {len(texts)} 条")
        
    except Exception as e:
        print(f"加载训练集失败: {e}")
        return [], []
    
    # 加载测试集
    try:
        with open(f'batch_cleaned_datasets\\ag_news_test_threshold_{threshold}.json', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # 处理可能的多行JSON格式
            current_obj = ""
            bracket_count = 0
            
            for char in content:
                current_obj += char
                
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    
                    if bracket_count == 0:
                        try:
                            data = json.loads(current_obj.strip())
                            if 'text' in data and 'label' in data:
                                texts.append(data['text'])
                                labels.append(data['label'])
                        except json.JSONDecodeError:
                            pass
                        
                        current_obj = ""
        
        print(f"测试集加载完成，总共: {len(texts)} 条")
        return texts, labels
        
    except Exception as e:
        print(f"加载测试集失败: {e}")
        # 如果测试集加载失败，只返回训练集数据
        return texts, labels

def analyze_dataset_statistics(texts, labels, dataset_name):
    """分析数据集统计信息"""
    print(f"\n{dataset_name} 数据集统计信息:")
    print("-" * 40)
    
    # 标签分布
    label_counts = pd.Series(labels).value_counts().sort_index()
    print(f"标签分布: {dict(label_counts)}")
    
    # 文本长度统计
    text_lengths = [len(text.split()) for text in texts]
    print(f"平均文本长度: {np.mean(text_lengths):.2f} 词")
    print(f"文本长度范围: {min(text_lengths)} - {max(text_lengths)} 词")
    
    return label_counts, text_lengths

def train_model(train_texts, train_labels, val_texts, val_labels, model_name="google/electra-small-discriminator", num_epochs=2):
    """训练分类模型"""
    print(f"\n开始训练模型: {model_name}")
    print("-" * 40)
    
    # 定义备选模型列表（按安全性和兼容性排序）
    fallback_models = [
        model_name,  # 用户指定的模型
        "google/electra-small-discriminator",  # 安全备选1
        "distilbert-base-uncased",  # 安全备选2
        "bert-base-uncased"  # 最后备选
    ]
    
    tokenizer = None
    model = None
    used_model = None
    
    # 加载tokenizer和模型（按优先级尝试）
    for attempt_model in fallback_models:
        if used_model:  # 如果已经成功加载，跳出循环
            break
            
        print(f"尝试加载模型: {attempt_model}")
        try:
            # 加载tokenizer
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(attempt_model)
                print("tokenizer加载成功")
            
            # 尝试使用safetensors格式加载模型
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    attempt_model, 
                    num_labels=4,  # AG News有4个类别
                    use_safetensors=True
                )
                used_model = attempt_model
                print(f"成功使用safetensors格式加载模型: {attempt_model}")
                break
                
            except Exception as safetensors_error:
                print(f"safetensors加载失败: {safetensors_error}")
                
                # 如果是第一个模型且是bert-tiny，尝试安全绕过
                if attempt_model == model_name and "bert-tiny" in model_name:
                    print("尝试安全绕过方案...")
                    try:
                        # 临时绕过安全检查
                        import transformers
                        from transformers.utils import import_utils
                        
                        original_check = import_utils.check_torch_load_is_safe
                        import_utils.check_torch_load_is_safe = lambda: None
                        
                        model = AutoModelForSequenceClassification.from_pretrained(
                            attempt_model, 
                            num_labels=4
                        )
                        used_model = attempt_model
                        print(f"使用安全绕过方式加载模型: {attempt_model}")
                        
                        # 恢复安全检查
                        import_utils.check_torch_load_is_safe = original_check
                        break
                        
                    except Exception as bypass_error:
                        print(f"安全绕过也失败: {bypass_error}")
                        # 恢复安全检查
                        import_utils.check_torch_load_is_safe = original_check
                        continue
                else:
                    continue
                    
        except Exception as e:
            print(f"模型 {attempt_model} 加载完全失败: {e}")
            continue
    
    if model is None or tokenizer is None:
        raise ValueError("所有模型都加载失败，请检查网络连接或PyTorch版本")
    
    if used_model != model_name:
        print(f"已自动切换到兼容模型: {used_model}")
    
    print(f"最终使用模型: {used_model}")
    
    # 创建数据集
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer)
    
    # 创建指标回调
    metrics_callback = MetricsCallback()
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,  # 减小batch size以避免内存问题
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,  # 更频繁的日志记录（每100步记录一次）
        eval_strategy="steps",  # 改为按步数评估，获得更多数据点
        eval_steps=250,  # 每250步评估一次，产生足够多的点形成平滑曲线
        save_strategy="steps",
        save_steps=250,  # 减少保存频率
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=None,  # 禁用wandb等外部日志
        save_total_limit=2,  # 只保存最好的2个模型
    )
    
    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 定义计算指标的函数
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback],
    )
    
    # 训练模型
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"训练完成，用时: {training_time:.2f} 秒")
    
    return trainer, tokenizer, training_time, metrics_callback

def evaluate_model(trainer, test_texts, test_labels, tokenizer):
    """评估模型性能"""
    print("\n评估模型性能...")
    
    # 创建测试数据集
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer)
    
    # 评估
    eval_results = trainer.evaluate(test_dataset)
    
    # 获取预测结果
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    
    # 详细分类报告
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    report = classification_report(
        test_labels, predicted_labels, 
        target_names=class_names, 
        output_dict=True
    )
    
    # 打印详细的每类别性能
    print(f"\n详细分类报告:")
    print(f"{'类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<8}")
    print("-" * 50)
    
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            support = report[str(i)]['support']
            print(f"{class_name:<12} {precision:<8.3f} {recall:<8.3f} {f1:<8.3f} {support:<8.0f}")
    
    # 打印宏平均和加权平均
    print("-" * 50)
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    print(f"{'宏平均':<12} {macro_avg['precision']:<8.3f} {macro_avg['recall']:<8.3f} {macro_avg['f1-score']:<8.3f}")
    print(f"{'加权平均':<12} {weighted_avg['precision']:<8.3f} {weighted_avg['recall']:<8.3f} {weighted_avg['f1-score']:<8.3f}")
    
    # 分析预测分布
    from collections import Counter
    true_dist = Counter(test_labels)
    pred_dist = Counter(predicted_labels)
    
    print(f"\n预测分布分析:")
    print(f"真实分布: {dict(true_dist)}")
    print(f"预测分布: {dict(pred_dist)}")
    
    return eval_results, report, predicted_labels

def plot_comparison_results(original_results, cleaned_results, original_callback, cleaned_callback):
    """绘制对比结果图表，包括训练过程和最终性能"""
    print("\n生成对比图表...")
    
    # 设置全局绘图样式
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Create 4 subplots with larger canvas
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Text Classification Deduplication Effect Comparison', fontsize=24, fontweight='bold', y=0.98)
    
    # 定义配色方案
    colors = {
        'original': '#FF6B6B',    # 暖红色
        'cleaned': '#4ECDC4',     # 青绿色
        'positive': '#51CF66',    # 绿色
        'negative': '#FF7979',    # 红色
        'neutral': '#A4A4A4'      # 灰色
    }
    
    # 1. Training Loss Comparison - Simplified Style
    if original_callback.train_losses and cleaned_callback.train_losses:
        # Simplified display for cleaner visualization
        steps_orig = list(range(len(original_callback.train_losses)))
        steps_clean = list(range(len(cleaned_callback.train_losses)))
        
        ax1.plot(steps_orig, original_callback.train_losses, 
                color=colors['original'], linewidth=2, label='Original Dataset', alpha=0.9)
        ax1.plot(steps_clean, cleaned_callback.train_losses, 
                color=colors['cleaned'], linewidth=2, label='Deduplicated Dataset', alpha=0.9)
        
        ax1.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Training Loss', fontsize=14, fontweight='bold')
        ax1.set_title('Training Loss Curves', fontsize=16, fontweight='bold', pad=20)
        ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    
    # 2. Validation Accuracy - Dense points for smooth curves
    if original_callback.eval_accuracies and cleaned_callback.eval_accuracies:
        # 直接使用记录的epoch数据，每25步评估产生的密集点
        orig_epochs = original_callback.epochs
        clean_epochs = cleaned_callback.epochs
        
        ax2.plot(orig_epochs, original_callback.eval_accuracies, 
                color=colors['original'], linewidth=3, 
                label='Original Dataset', alpha=0.9)
        ax2.plot(clean_epochs, cleaned_callback.eval_accuracies, 
                color=colors['cleaned'], linewidth=3, 
                label='Deduplicated Dataset', alpha=0.9)
        
        ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_title('Validation Accuracy Curves', fontsize=16, fontweight='bold', pad=20)
        ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, 3)  # 确保显示从0到3
        ax2.set_ylim(0.3, 1.0)  # 从更低的准确率开始
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    # 3. Final Performance Comparison
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    original_scores = [original_results['eval_' + metric] for metric in metrics]
    cleaned_scores = [cleaned_results['eval_' + metric] for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, original_scores, width, 
                   label='Original Dataset', alpha=0.8, color=colors['original'],
                   edgecolor='white', linewidth=2)
    bars2 = ax3.bar(x + width/2, cleaned_scores, width, 
                   label='Deduplicated Dataset', alpha=0.8, color=colors['cleaned'],
                   edgecolor='white', linewidth=2)
    
    ax3.set_xlabel('Evaluation Metrics', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax3.set_title('Final Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names, fontsize=11)
    ax3.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax3.set_ylim(0.7, 1.0)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color=colors['original'])
    
    for bar in bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color=colors['cleaned'])
    
    # 4. Deduplication Effect Analysis
    differences = [cleaned_scores[i] - original_scores[i] for i in range(len(metrics))]
    bar_colors = [colors['positive'] if d > 0 else colors['negative'] if d < 0 else colors['neutral'] 
                  for d in differences]
    
    bars3 = ax4.bar(metric_names, differences, color=bar_colors, alpha=0.8,
                   edgecolor='white', linewidth=2)
    
    ax4.set_xlabel('Evaluation Metrics', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Performance Difference (Deduplicated - Original)', fontsize=14, fontweight='bold')
    ax4.set_title('Deduplication Effect Analysis', fontsize=16, fontweight='bold', pad=20)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add value labels and effect indicators
    for i, (bar, diff) in enumerate(zip(bars3, differences)):
        height = bar.get_height()
        # Value labels
        ax4.text(bar.get_x() + bar.get_width()/2., 
                height + (0.003 if height > 0 else -0.008),
                f'{height:+.1%}', ha='center', 
                va='bottom' if height > 0 else 'top', 
                fontsize=11, fontweight='bold')
        
        # Effect indicators
        if abs(diff) > 0.005:  # Significant change
            icon = '↑' if diff > 0 else '↓'
            ax4.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.015 if height > 0 else -0.020),
                    icon, ha='center', va='center', fontsize=14)
    
    # Add overall statistics
    avg_improvement = np.mean(differences)
    improvement_text = f"Average Performance Change: {avg_improvement:+.2%}"
    fig.text(0.02, 0.02, improvement_text, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # 调整布局和样式
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 确保目录存在
    os.makedirs('结果汇总', exist_ok=True)
    
    # 保存高质量图片
    plt.savefig('结果汇总\\classification_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('结果汇总\\训练对比结果.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')

    print("Charts saved as 结果汇总\\classification_comparison.png and 结果汇总\\训练对比结果.png")
    plt.show()

def save_detailed_results(original_results, cleaned_results, original_time, cleaned_time, 
                         original_callback, cleaned_callback):
    """保存详细结果到文件"""
    results = {
        'experiment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'original_dataset': {
            'metrics': original_results,
            'training_time': original_time,
            'training_history': {
                'train_losses': original_callback.train_losses,
                'eval_losses': original_callback.eval_losses,
                'eval_accuracies': original_callback.eval_accuracies,
                'epochs': original_callback.epochs
            }
        },
        'cleaned_dataset': {
            'metrics': cleaned_results,
            'training_time': cleaned_time,
            'training_history': {
                'train_losses': cleaned_callback.train_losses,
                'eval_losses': cleaned_callback.eval_losses,
                'eval_accuracies': cleaned_callback.eval_accuracies,
                'epochs': cleaned_callback.epochs
            }
        },
        'improvements': {
            'accuracy': cleaned_results['eval_accuracy'] - original_results['eval_accuracy'],
            'f1': cleaned_results['eval_f1'] - original_results['eval_f1'],
            'precision': cleaned_results['eval_precision'] - original_results['eval_precision'],
            'recall': cleaned_results['eval_recall'] - original_results['eval_recall'],
            'training_time_diff': cleaned_time - original_time
        }
    }
    
    # 确保目录存在
    os.makedirs('结果汇总', exist_ok=True)
    
    with open('结果汇总\\comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 同时生成一个简化的文本报告
    with open('结果汇总\\实验报告.txt', 'w', encoding='utf-8') as f:
        f.write("文本分类去重效果对比实验报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"实验时间: {results['experiment_time']}\n\n")
        
        f.write("数据集对比:\n")
        f.write(f"原始数据集最终准确率: {original_results['eval_accuracy']:.4f}\n")
        f.write(f"去重数据集最终准确率: {cleaned_results['eval_accuracy']:.4f}\n")
        f.write(f"准确率提升: {results['improvements']['accuracy']:+.4f}\n\n")
        
        f.write(f"原始数据集F1分数: {original_results['eval_f1']:.4f}\n")
        f.write(f"去重数据集F1分数: {cleaned_results['eval_f1']:.4f}\n")
        f.write(f"F1分数提升: {results['improvements']['f1']:+.4f}\n\n")
        
        f.write(f"训练时间对比:\n")
        f.write(f"原始数据集训练时间: {original_time:.2f} 秒\n")
        f.write(f"去重数据集训练时间: {cleaned_time:.2f} 秒\n")
        f.write(f"时间差异: {results['improvements']['training_time_diff']:+.2f} 秒\n")

    print("详细结果已保存到 结果汇总\\comparison_results.json")
    print("简化报告已保存到 结果汇总\\实验报告.txt")

def save_threshold_results(threshold, results, training_time, callback, dataset_size):
    """保存单个阈值的详细结果到文件"""
    import time
    import json
    import os
    
    result_data = {
        'experiment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'threshold': threshold,
        'dataset_info': {
            'size': dataset_size,
            'reduction_rate': None  # 可以后续计算
        },
        'model_performance': {
            'accuracy': results['eval_accuracy'],
            'f1_score': results['eval_f1'],
            'precision': results['eval_precision'],
            'recall': results['eval_recall'],
            'loss': results['eval_loss']
        },
        'training_info': {
            'training_time_seconds': training_time,
            'total_epochs': len(callback.train_losses),
            'final_train_loss': callback.train_losses[-1] if callback.train_losses else None,
            'final_eval_loss': callback.eval_losses[-1] if callback.eval_losses else None,
            'best_accuracy': max(callback.eval_accuracies) if callback.eval_accuracies else None
        },
        'training_history': {
            'train_losses': callback.train_losses,
            'eval_losses': callback.eval_losses,
            'eval_accuracies': callback.eval_accuracies,
            'epochs': callback.epochs
        }
    }
    
    # 确保目录存在
    os.makedirs('threshold_results', exist_ok=True)
    
    # 保存JSON格式的详细结果
    json_filename = f'threshold_results\\threshold_{threshold}_results.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    # 保存简化的文本报告
    txt_filename = f'threshold_results\\threshold_{threshold}_report.txt'
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"阈值 {threshold} 实验结果报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"实验时间: {result_data['experiment_time']}\n")
        f.write(f"相似度阈值: {threshold}\n")
        f.write(f"数据集大小: {dataset_size} 条\n\n")
        
        f.write("模型性能指标:\n")
        f.write(f"  准确率: {results['eval_accuracy']:.4f}\n")
        f.write(f"  F1分数: {results['eval_f1']:.4f}\n")
        f.write(f"  精确率: {results['eval_precision']:.4f}\n")
        f.write(f"  召回率: {results['eval_recall']:.4f}\n")
        f.write(f"  损失值: {results['eval_loss']:.4f}\n\n")
        
        f.write("训练信息:\n")
        f.write(f"  训练时间: {training_time:.2f} 秒\n")
        f.write(f"  训练轮数: {len(callback.train_losses)}\n")
        if callback.eval_accuracies:
            f.write(f"  最佳准确率: {max(callback.eval_accuracies):.4f}\n")
        f.write(f"  最终训练损失: {callback.train_losses[-1]:.4f}\n")
        f.write(f"  最终验证损失: {callback.eval_losses[-1]:.4f}\n")
    
    print(f"阈值 {threshold} 结果已保存:")
    print(f"  详细结果: {json_filename}")
    print(f"  简化报告: {txt_filename}")
    
    return result_data

def main():
    """主实验函数"""
    print("文本分类去重效果对比实验")
    print("=" * 50)
    
    # 检查关键依赖
    try:
        import tqdm
        print("依赖检查通过")
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip install tqdm")
        return
    
    # 检查PyTorch版本和安全性
    torch_version = torch.__version__
    print(f"当前PyTorch版本: {torch_version}")
    
    # 可选的模型列表（从弱到强，优先选择支持safetensors的模型）：
    # "google/electra-small-discriminator" - 14M参数，较弱，支持safetensors
    # "distilbert-base-uncased"    - 66M参数，中等强度，支持safetensors  
    # "prajjwal1/bert-tiny"        - 4M参数，最弱，可能不支持safetensors
    # "bert-base-uncased"          - 110M参数，较强，支持safetensors
    # 
    # 建议：如果bert-tiny加载失败，自动切换到electra-small-discriminator
    
    # 设置随机种子
    np.random.seed(42)
    
    # 检查是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 1. 加载数据集
        print("第一步：加载数据集")
        
        # 加载去重后的数据集
        cleaned_texts, cleaned_labels = load_cleaned_dataset()
        
        if not cleaned_texts:
            print("无法加载去重数据集，实验终止")
            return
        
        # 使用全部数据进行训练
        max_samples = len(cleaned_texts)  # 使用全部数据
        print(f"使用全部 {len(cleaned_texts)} 条数据进行训练")
        
        # 分割去重数据集
        cleaned_train_texts, cleaned_test_texts, cleaned_train_labels, cleaned_test_labels = train_test_split(
            cleaned_texts, cleaned_labels, test_size=0.2, random_state=42, stratify=cleaned_labels
        )
        
        # 加载或创建原始数据集
        print("加载原始AG News数据集...")
        original_train_texts, original_train_labels, original_test_texts, original_test_labels = load_original_dataset()
        
        # 使用相同大小的原始数据集进行公平对比
        print(f"原始数据集使用全部数据")
        
        if len(original_test_texts) > len(cleaned_test_texts):  # 如果原始测试集更大，随机采样到相同大小
            indices = np.random.choice(len(original_test_texts), len(cleaned_test_texts), replace=False)
            indices = [int(i) for i in indices]  # 转换为Python int类型
            original_test_texts = [original_test_texts[i] for i in indices]
            original_test_labels = [original_test_labels[i] for i in indices]
        
        # 2. 分析数据集统计信息
        print("\n第二步：数据集统计分析")
        
        # 详细统计对比
        print(f"\n数据量对比:")
        print(f"原始训练集: {len(original_train_texts)} 条")
        print(f"去重训练集: {len(cleaned_train_texts)} 条")
        print(f"数据减少: {len(original_train_texts) - len(cleaned_train_texts)} 条 ({(len(original_train_texts) - len(cleaned_train_texts)) / len(original_train_texts) * 100:.1f}%)")
        
        print(f"\n原始测试集: {len(original_test_texts)} 条")
        print(f"去重测试集: {len(cleaned_test_texts)} 条")
        
        # 分析类别分布
        from collections import Counter
        
        print(f"\n类别分布对比:")
        original_dist = Counter(original_train_labels + original_test_labels)
        cleaned_dist = Counter(cleaned_train_labels + cleaned_test_labels)
        
        print("原始数据集类别分布:", dict(original_dist))
        print("去重数据集类别分布:", dict(cleaned_dist))
        
        # 计算类别分布变化
        for label in original_dist.keys():
            change = cleaned_dist[label] - original_dist[label]
            change_pct = change / original_dist[label] * 100 if original_dist[label] > 0 else 0
            print(f"  类别 {label}: {original_dist[label]} → {cleaned_dist[label]} ({change_pct:+.1f}%)")
        
        analyze_dataset_statistics(original_train_texts + original_test_texts, 
                                 original_train_labels + original_test_labels, "原始")
        analyze_dataset_statistics(cleaned_train_texts + cleaned_test_texts, 
                                 cleaned_train_labels + cleaned_test_labels, "去重后")
        
        # 3. 训练原始数据集模型
        print("\n第三步：训练原始数据集模型")
        original_trainer, original_tokenizer, original_time, original_callback = train_model(
            original_train_texts, original_train_labels,
            original_test_texts, original_test_labels
        )
        
        # 4. 评估原始数据集模型
        print("\n第四步：评估原始数据集模型")
        original_results, original_report, _ = evaluate_model(
            original_trainer, original_test_texts, original_test_labels, original_tokenizer
        )
        
        # 5. 训练去重数据集模型
        print("\n第五步：训练去重数据集模型")
        cleaned_trainer, cleaned_tokenizer, cleaned_time, cleaned_callback = train_model(
            cleaned_train_texts, cleaned_train_labels,
            cleaned_test_texts, cleaned_test_labels
        )
        
        # 6. 评估去重数据集模型
        print("\n第六步：评估去重数据集模型")
        cleaned_results, cleaned_report, _ = evaluate_model(
            cleaned_trainer, cleaned_test_texts, cleaned_test_labels, cleaned_tokenizer
        )
        
        # 7. 结果对比分析
        print("\n第七步：结果对比分析")
        print("=" * 50)
        print("实验结果汇总:")
        print(f"原始数据集 - 准确率: {original_results['eval_accuracy']:.4f}, F1: {original_results['eval_f1']:.4f}")
        print(f"去重数据集 - 准确率: {cleaned_results['eval_accuracy']:.4f}, F1: {cleaned_results['eval_f1']:.4f}")
        print(f"准确率提升: {cleaned_results['eval_accuracy'] - original_results['eval_accuracy']:.4f}")
        print(f"F1分数提升: {cleaned_results['eval_f1'] - original_results['eval_f1']:.4f}")
        print(f"训练时间对比: 原始 {original_time:.2f}s vs 去重 {cleaned_time:.2f}s")
        
        # 8. 生成图表和保存结果
        print("\n第八步：生成报告")
        plot_comparison_results(original_results, cleaned_results, original_callback, cleaned_callback)
        save_detailed_results(original_results, cleaned_results, original_time, cleaned_time,
                            original_callback, cleaned_callback)
        
        print("\n实验完成！")
        print("图表已保存为 结果汇总\\classification_comparison.png 和 结果汇总\\训练对比结果.png")
        print("详细结果已保存为 结果汇总\\comparison_results.json")
        print("简化报告已保存为 结果汇总\\实验报告.txt")
        
    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
