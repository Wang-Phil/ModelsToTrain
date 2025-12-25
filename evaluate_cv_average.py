"""
五折交叉验证平均结果评估脚本
评估所有fold并计算平均指标
"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 导入单fold评估函数
from evaluate_per_class import (
    load_model, evaluate_model, calculate_per_class_metrics,
    print_per_class_report, save_per_class_table, plot_per_class_comparison
)
from train_multiclass import (
    ImageFolderDataset, get_data_augmentation
)
from torch.utils.data import DataLoader, Subset, Dataset
import torch
from PIL import Image
# 设置字体（使用英文，不需要中文字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False


class TransformSubset(Dataset):
    """支持不同transform的Subset"""
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取原始数据
        img_path, label = self.base_dataset.samples[self.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        # 应用指定的transform
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate_single_fold(model_path, data_dir, device, args, val_indices=None):
    """评估单个fold"""
    # 加载模型
    model, checkpoint_info = load_model(model_path, device, num_classes=args.num_classes)
    
    # 获取类别信息
    if checkpoint_info['class_to_idx']:
        class_to_idx = checkpoint_info['class_to_idx']
        idx_to_class = checkpoint_info['idx_to_class']
        class_names = sorted(class_to_idx.keys())
        num_classes = len(class_names)
    else:
        if args.num_classes is None:
            raise ValueError("无法确定类别数")
        num_classes = args.num_classes
        class_names = [f'Class_{i}' for i in range(num_classes)]
        idx_to_class = {i: name for i, name in enumerate(class_names)}
    
    # 数据变换
    _, val_transform = get_data_augmentation('none', img_size=args.img_size)
    
    # 加载完整数据集
    full_dataset = ImageFolderDataset(data_dir, transform=None)
    
    # 如果提供了验证集索引，只评估验证集数据（避免数据泄露）
    if val_indices is not None:
        print(f"  Evaluating validation set only: {len(val_indices)} samples")
        dataset = TransformSubset(full_dataset, val_indices, transform=val_transform)
    else:
        print(f"  Warning: Validation indices not provided, evaluating entire dataset (may include training data)")
        dataset = ImageFolderDataset(data_dir, transform=val_transform)
    
    # 数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 评估模型
    y_pred, y_true, y_probs = evaluate_model(
        model, dataloader, device, num_classes
    )
    
    # 计算整体准确率
    from sklearn.metrics import accuracy_score
    overall_accuracy = accuracy_score(y_true, y_pred)
    
    # 计算每个类别的指标
    per_class_metrics, confusion_mat = calculate_per_class_metrics(
        y_true, y_pred, y_probs, num_classes, class_names
    )
    
    return {
        'overall_accuracy': overall_accuracy,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': confusion_mat,
        'class_names': class_names,
        'num_classes': num_classes,
        'y_true': y_true,
        'y_probs': y_probs
    }


def calculate_average_metrics(all_fold_results):
    """计算所有fold的平均指标"""
    num_folds = len(all_fold_results)
    
    # 获取类别信息（从第一个fold）
    first_result = all_fold_results[0]
    class_names = first_result['class_names']
    num_classes = first_result['num_classes']
    
    # 初始化累加器
    overall_accuracy_sum = 0.0
    
    # 每个类别的指标累加器
    per_class_accumulators = defaultdict(lambda: {
        'precision': [],
        'recall': [],
        'f1_score': [],
        'specificity': [],
        'auc': [],
        'accuracy': [],
        'support': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'TN': []
    })
    
    # 累加所有fold的结果
    for fold_result in all_fold_results:
        overall_accuracy_sum += fold_result['overall_accuracy']
        
        for class_metrics in fold_result['per_class_metrics']:
            class_name = class_metrics['class_name']
            per_class_accumulators[class_name]['precision'].append(class_metrics['precision'])
            per_class_accumulators[class_name]['recall'].append(class_metrics['recall'])
            per_class_accumulators[class_name]['f1_score'].append(class_metrics['f1_score'])
            per_class_accumulators[class_name]['specificity'].append(class_metrics['specificity'])
            per_class_accumulators[class_name]['auc'].append(class_metrics['auc'])
            per_class_accumulators[class_name]['accuracy'].append(class_metrics['accuracy'])
            per_class_accumulators[class_name]['support'].append(class_metrics['support'])
            per_class_accumulators[class_name]['TP'].append(class_metrics['TP'])
            per_class_accumulators[class_name]['FP'].append(class_metrics['FP'])
            per_class_accumulators[class_name]['FN'].append(class_metrics['FN'])
            per_class_accumulators[class_name]['TN'].append(class_metrics['TN'])
    
    # 计算平均值
    avg_overall_accuracy = overall_accuracy_sum / num_folds
    
    # 计算每个类别的平均指标
    avg_per_class_metrics = []
    for i, class_name in enumerate(class_names):
        acc = per_class_accumulators[class_name]
        
        # 计算均值和标准差
        avg_metrics = {
            'class_index': i,
            'class_name': class_name,
            'support': int(np.mean(acc['support'])),  # 平均样本数
            'TP': int(np.mean(acc['TP'])),
            'FP': int(np.mean(acc['FP'])),
            'FN': int(np.mean(acc['FN'])),
            'TN': int(np.mean(acc['TN'])),
            'precision': float(np.mean(acc['precision'])),
            'precision_std': float(np.std(acc['precision'])),
            'recall': float(np.mean(acc['recall'])),
            'recall_std': float(np.std(acc['recall'])),
            'f1_score': float(np.mean(acc['f1_score'])),
            'f1_score_std': float(np.std(acc['f1_score'])),
            'specificity': float(np.mean(acc['specificity'])),
            'specificity_std': float(np.std(acc['specificity'])),
            'auc': float(np.mean(acc['auc'])),
            'auc_std': float(np.std(acc['auc'])),
            'accuracy': float(np.mean(acc['accuracy'])),
            'accuracy_std': float(np.std(acc['accuracy']))
        }
        avg_per_class_metrics.append(avg_metrics)
    
    return avg_overall_accuracy, avg_per_class_metrics


def print_average_report(avg_per_class_metrics, avg_overall_accuracy, num_folds):
    """打印平均结果报告"""
    print("\n" + "=" * 110)
    print(f"Cross-Validation Average Results (based on {num_folds} folds)")
    print("=" * 110)
    print(f"\nOverall Accuracy: {avg_overall_accuracy*100:.2f}%\n")
    
    # Table header
    print(f"{'Class':<25} {'Support':<10} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Specificity':<15} {'AUC':<15}")
    print(f"{'':<25} {'':<10} {'(mean±std)':<15} {'(mean±std)':<15} {'(mean±std)':<15} {'(mean±std)':<15} {'(mean±std)':<15}")
    print("-" * 110)
    
    # 每个类别的指标
    for metrics in avg_per_class_metrics:
        print(f"{metrics['class_name']:<25} "
              f"{metrics['support']:<10} "
              f"{metrics['precision']*100:>5.2f}±{metrics['precision_std']*100:>4.2f}% "
              f"{metrics['recall']*100:>5.2f}±{metrics['recall_std']*100:>4.2f}% "
              f"{metrics['f1_score']*100:>5.2f}±{metrics['f1_score_std']*100:>4.2f}% "
              f"{metrics['specificity']*100:>5.2f}±{metrics['specificity_std']*100:>4.2f}% "
              f"{metrics['auc']:>5.4f}±{metrics['auc_std']:>5.4f}")
    
    print("\n" + "=" * 110)


def save_average_table(avg_per_class_metrics, avg_overall_accuracy, num_folds, output_path):
    """Save average results table"""
    # Prepare data
    data = []
    for metrics in avg_per_class_metrics:
        data.append({
            'Class': metrics['class_name'],
            'Class Index': metrics['class_index'],
            'Support': metrics['support'],
            'TP (mean)': round(metrics['TP'], 1),
            'FP (mean)': round(metrics['FP'], 1),
            'FN (mean)': round(metrics['FN'], 1),
            'TN (mean)': round(metrics['TN'], 1),
            'Precision (%)': f"{metrics['precision']*100:.2f}±{metrics['precision_std']*100:.2f}",
            'Recall (%)': f"{metrics['recall']*100:.2f}±{metrics['recall_std']*100:.2f}",
            'F1-Score (%)': f"{metrics['f1_score']*100:.2f}±{metrics['f1_score_std']*100:.2f}",
            'Specificity (%)': f"{metrics['specificity']*100:.2f}±{metrics['specificity_std']*100:.2f}",
            'AUC': f"{metrics['auc']:.4f}±{metrics['auc_std']:.4f}",
            'Accuracy (%)': f"{metrics['accuracy']*100:.2f}±{metrics['accuracy_std']*100:.2f}"
        })
    
    df = pd.DataFrame(data)
    
    # Add overall accuracy row
    summary_row = pd.DataFrame([{
        'Class': 'Overall',
        'Class Index': '-',
        'Support': '-',
        'TP (mean)': '-',
        'FP (mean)': '-',
        'FN (mean)': '-',
        'TN (mean)': '-',
        'Precision (%)': '-',
        'Recall (%)': '-',
        'F1-Score (%)': '-',
        'Specificity (%)': '-',
        'AUC': '-',
        'Accuracy (%)': f"{avg_overall_accuracy*100:.2f}%"
    }])
    
    df = pd.concat([df, summary_row], ignore_index=True)
    
    # 保存CSV
    csv_path = output_path.replace('.xlsx', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved CSV table: {csv_path}")
    
    # 保存Excel（如果openpyxl可用）
    if output_path.endswith('.xlsx'):
        try:
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"Saved Excel table: {output_path}")
        except ImportError:
            print(f"Warning: openpyxl not installed, skipping Excel output. Please run: pip install openpyxl")
        except Exception as e:
            print(f"Warning: Error saving Excel: {e}")


def plot_average_comparison(avg_per_class_metrics, output_path, all_fold_results=None, figsize=(10, 6)):
    """Plot average results comparison charts, each chart saved separately"""
    class_names = [m['class_name'] for m in avg_per_class_metrics]
    num_classes = len(class_names)
    
    # 提取指标
    precision = [m['precision'] * 100 for m in avg_per_class_metrics]
    recall = [m['recall'] * 100 for m in avg_per_class_metrics]
    f1 = [m['f1_score'] * 100 for m in avg_per_class_metrics]
    specificity = [m['specificity'] * 100 for m in avg_per_class_metrics]
    auc = [m['auc'] for m in avg_per_class_metrics]
    support = [m['support'] for m in avg_per_class_metrics]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    # 获取基础文件名（不含扩展名）
    base_path = output_path.rsplit('.', 1)[0]
    
    # 1. Precision, Recall, F1对比
    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只显示大于0的值
                # 将标签放在柱状图上方，如果太高则放在内部
                y_pos = height + 1 if height < 95 else height - 3
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{height:.1f}%',
                       ha='center', va='bottom' if height < 95 else 'top', 
                       fontsize=7, rotation=0)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Precision, Recall, F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 110])
    ax.set_yticks(np.arange(0, 111, 20))
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(pad=3.0)
    plt.savefig(f'{base_path}_1_precision_recall_f1.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Saved chart: {base_path}_1_precision_recall_f1.png")
    
    # 2. Specificity
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(x, specificity, alpha=0.8, color='green')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            y_pos = height + 1 if height < 95 else height - 3
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height < 95 else 'top', 
                   fontsize=7, rotation=0)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Specificity (%)', fontsize=12)
    ax.set_title('Specificity', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 110])
    ax.set_yticks(np.arange(0, 111, 20))
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(pad=3.0)
    plt.savefig(f'{base_path}_2_specificity.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Saved chart: {base_path}_2_specificity.png")
    
    # 3. AUC
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(x, auc, alpha=0.8, color='purple')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            y_pos = height + 0.02 if height < 0.95 else height - 0.03
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:.3f}',
                   ha='center', va='bottom' if height < 0.95 else 'top', 
                   fontsize=7, rotation=0)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('ROC AUC', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(pad=3.0)
    plt.savefig(f'{base_path}_3_auc.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Saved chart: {base_path}_3_auc.png")
    
    # 4. Sample Count Distribution
    fig, ax = plt.subplots(figsize=(16, 8))
    max_support = max(support) if support else 100
    bars = ax.bar(x, support, alpha=0.8, color='orange')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            y_pos = height + max_support * 0.02
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{int(height)}',
                   ha='center', va='bottom', 
                   fontsize=7, rotation=0)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Sample Count per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, max_support * 1.15])
    y_ticks = np.linspace(0, max_support, 6)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(pad=3.0)
    plt.savefig(f'{base_path}_4_sample_count.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Saved chart: {base_path}_4_sample_count.png")
    
    # 5. Composite Score
    composite_scores = []
    for i in range(len(class_names)):
        score = (precision[i] + recall[i] + f1[i] + specificity[i]) / 4
        composite_scores.append(score)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(x, composite_scores, alpha=0.8, color='red')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            y_pos = height + 1 if height < 95 else height - 3
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height < 95 else 'top', 
                   fontsize=7, rotation=0)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Composite Score (%)', fontsize=12)
    ax.set_title('Composite Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 110])
    ax.set_yticks(np.arange(0, 111, 20))
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(pad=3.0)
    plt.savefig(f'{base_path}_5_composite_score.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Saved chart: {base_path}_5_composite_score.png")
    
    # 6. Heatmap: Metrics Matrix
    metrics_matrix = np.array([
        precision,
        recall,
        f1,
        specificity,
        [a * 100 for a in auc]
    ])
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(metrics_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels(['Precision', 'Recall', 'F1', 'Specificity', 'AUC*100'], fontsize=10)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_title('Metrics Heatmap', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    
    # 添加数值标注
    for i in range(5):
        for j in range(len(class_names)):
            ax.text(j, i, f'{metrics_matrix[i, j]:.1f}',
                   ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{base_path}_6_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved chart: {base_path}_6_heatmap.png")
    
    # 7. ROC Curves
    if all_fold_results is not None:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # Collect all y_true and y_probs from all folds
        all_y_true = []
        all_y_probs = []
        
        for fold_result in all_fold_results:
            if 'y_true' in fold_result and 'y_probs' in fold_result:
                all_y_true.append(fold_result['y_true'])
                all_y_probs.append(fold_result['y_probs'])
        
        if len(all_y_true) > 0:
            # Concatenate all folds
            y_true_all = np.concatenate(all_y_true)
            y_probs_all = np.concatenate(all_y_probs, axis=0)
            
            # Binarize the labels for multi-class ROC
            y_true_bin = label_binarize(y_true_all, classes=range(num_classes))
            
            # Compute ROC curve and AUC for each class
            fig, ax = plt.subplots(figsize=(12, 10))
            colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
            
            for i, class_name in enumerate(class_names):
                if num_classes == 2:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_all[:, i])
                else:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_all[:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=colors[i], lw=2,
                       label=f'{class_name} (AUC = {roc_auc:.3f})')
            
            # Plot diagonal line
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curves for All Classes', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=9, ncol=1)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{base_path}_7_roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved chart: {base_path}_7_roc_curves.png")


def evaluate_cv_average(args):
    """主评估函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有fold的模型
    model_base_dir = Path(args.model_base_dir)
    if not model_base_dir.exists():
        raise ValueError(f"模型目录不存在: {model_base_dir}")
    
    # 查找所有fold
    fold_dirs = sorted([d for d in model_base_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    
    if len(fold_dirs) == 0:
        raise ValueError(f"在 {model_base_dir} 中未找到fold目录")
    
    print(f"\n找到 {len(fold_dirs)} 个fold: {[d.name for d in fold_dirs]}")
    
    # 读取folds_info.json以获取每个fold的验证集索引
    folds_info_path = model_base_dir / 'folds_info.json'
    folds_info = {}
    if folds_info_path.exists():
        with open(folds_info_path, 'r') as f:
            folds_info = json.load(f)
        print(f"✓ Loaded data split information: {folds_info_path}")
    else:
        print(f"⚠ Warning: {folds_info_path} not found, evaluating entire dataset (may include training data)")
    
    # 评估每个fold
    all_fold_results = []
    for fold_dir in fold_dirs:
        fold_num = fold_dir.name
        model_path = fold_dir / 'best_model.pth'
        
        if not model_path.exists():
            print(f"Warning: {model_path} does not exist, skipping")
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating {fold_num}...")
        print(f"{'='*80}")
        
        # 获取当前fold的验证集索引
        val_indices = None
        if fold_num in folds_info and 'val_indices' in folds_info[fold_num]:
            val_indices = folds_info[fold_num]['val_indices']
            print(f"  Validation set size: {len(val_indices)} samples")
        
        try:
            fold_result = evaluate_single_fold(
                str(model_path), args.data_dir, device, args, val_indices=val_indices
            )
            all_fold_results.append(fold_result)
            print(f"✓ {fold_num} evaluation completed")
        except Exception as e:
            print(f"✗ {fold_num} evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_fold_results) == 0:
        raise ValueError("没有成功评估任何fold")
    
    print(f"\n{'='*80}")
    print(f"Successfully evaluated {len(all_fold_results)} folds, calculating average results...")
    print(f"{'='*80}")
    
    # 计算平均指标
    avg_overall_accuracy, avg_per_class_metrics = calculate_average_metrics(all_fold_results)
    
    # 打印报告
    print_average_report(avg_per_class_metrics, avg_overall_accuracy, len(all_fold_results))
    
    # 保存结果
    print(f"\nSaving results to: {args.output_dir}")
    
    # 保存JSON
    results = {
        'model_name': args.model_name if hasattr(args, 'model_name') else model_base_dir.name,
        'model_base_dir': str(model_base_dir),
        'num_folds': len(all_fold_results),
        'avg_overall_accuracy': float(avg_overall_accuracy),
        'num_classes': all_fold_results[0]['num_classes'],
        'class_names': all_fold_results[0]['class_names'],
        'avg_per_class_metrics': avg_per_class_metrics,
        'individual_fold_results': [
            {
                'fold': f'fold_{i+1}',
                'overall_accuracy': float(r['overall_accuracy'])
            }
            for i, r in enumerate(all_fold_results)
        ]
    }
    
    json_path = os.path.join(args.output_dir, 'cv_average_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON results: {json_path}")
    
    # 保存表格
    table_path = os.path.join(args.output_dir, 'cv_average_metrics.xlsx')
    save_average_table(avg_per_class_metrics, avg_overall_accuracy, len(all_fold_results), table_path)
    
    # 绘制可视化
    plot_path = os.path.join(args.output_dir, 'cv_average_comparison.png')
    plot_average_comparison(avg_per_class_metrics, plot_path, all_fold_results)
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='五折交叉验证平均结果评估')
    
    # 必需参数
    parser.add_argument('--model-base-dir', type=str, required=True,
                        help='模型基础目录（包含fold_1, fold_2等子目录）')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='评估数据目录（按类别组织）')
    
    # 可选参数
    parser.add_argument('--num-classes', type=int, default=None,
                        help='类别数（如果模型文件中没有保存）')
    parser.add_argument('--img-size', type=int, default=224,
                        help='输入图像大小')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认：基于模型目录名）')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')
    
    args = parser.parse_args()
    
    # 如果没有指定输出目录，使用默认值
    if args.output_dir is None:
        model_name = Path(args.model_base_dir).name
        args.output_dir = f'evaluation_results/{model_name}_cv_average'
    
    evaluate_cv_average(args)


if __name__ == '__main__':
    main()

