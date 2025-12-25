"""
按类别评估模型性能脚本
用于评估单个模型在不同类别上的详细指标
"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# 导入训练脚本中的模型和数据加载器
from train_multiclass import (
    ImageFolderDataset, create_model, get_data_augmentation
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_model(checkpoint_path, device, num_classes=None):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型信息
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    else:
        # 从路径推断模型名称
        model_name = os.path.basename(checkpoint_path).split('_')[0]
    
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        num_classes = len(class_to_idx)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
    else:
        if num_classes is None:
            raise ValueError("无法确定类别数，请提供num_classes参数")
        class_to_idx = None
        idx_to_class = None
    
    # 创建模型
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    # 使用 strict=False 来处理缺失的键（例如旧模型可能没有GRN参数）
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    checkpoint_info = {
        'model_name': model_name,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_acc': checkpoint.get('val_acc', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown'),
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class
    }
    
    print(f"模型: {model_name}")
    print(f"类别数: {num_classes}")
    if checkpoint_info['epoch'] != 'unknown':
        print(f"训练轮数: {checkpoint_info['epoch']}")
    if checkpoint_info['val_acc'] != 'unknown':
        print(f"验证准确率: {checkpoint_info['val_acc']:.2f}%")
    
    return model, checkpoint_info


def evaluate_model(model, dataloader, device, num_classes):
    """评估模型，返回预测结果和真实标签"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='评估中'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_per_class_metrics(y_true, y_pred, y_probs, num_classes, class_names):
    """
    计算每个类别的详细指标
    
    Returns:
        per_class_metrics: 每个类别的指标字典列表
    """
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # 每个类别的指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # 计算每个类别的TP, FP, FN, TN
    per_class_metrics = []
    
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        
        # 计算特异性
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        
        # 计算支持度（真实样本数）
        support = cm[i, :].sum()
        
        # 计算ROC AUC（如果可能）
        try:
            y_true_binary = (y_true == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                auc_score = roc_auc_score(y_true_binary, y_probs[:, i])
            else:
                auc_score = 0.0
        except:
            auc_score = 0.0
        
        # 计算准确率（对于这个类别）
        class_accuracy = TP / support if support > 0 else 0.0
        
        class_metrics = {
            'class_index': i,
            'class_name': class_names[i] if class_names else f'Class_{i}',
            'support': int(support),
            'TP': int(TP),
            'FP': int(FP),
            'FN': int(FN),
            'TN': int(TN),
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'specificity': float(specificity),
            'auc': float(auc_score),
            'accuracy': float(class_accuracy)
        }
        
        per_class_metrics.append(class_metrics)
    
    return per_class_metrics, cm


def print_per_class_report(per_class_metrics, overall_accuracy):
    """打印按类别的详细报告"""
    print("\n" + "=" * 100)
    print("按类别评估报告")
    print("=" * 100)
    print(f"\n整体准确率 (Overall Accuracy): {overall_accuracy*100:.2f}%\n")
    
    # 表头
    print(f"{'类别':<20} {'样本数':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'特异性':<10} {'AUC':<10} {'准确率':<10}")
    print("-" * 100)
    
    # 每个类别的指标
    for metrics in per_class_metrics:
        print(f"{metrics['class_name']:<20} "
              f"{metrics['support']:<10} "
              f"{metrics['precision']*100:>8.2f}% "
              f"{metrics['recall']*100:>8.2f}% "
              f"{metrics['f1_score']*100:>8.2f}% "
              f"{metrics['specificity']*100:>8.2f}% "
              f"{metrics['auc']:>8.4f} "
              f"{metrics['accuracy']*100:>8.2f}%")
    
    print("\n" + "=" * 100)
    print("\n详细混淆矩阵统计:")
    print("-" * 100)
    for metrics in per_class_metrics:
        print(f"\n{metrics['class_name']}:")
        print(f"  TP (真阳性): {metrics['TP']}")
        print(f"  FP (假阳性): {metrics['FP']}")
        print(f"  FN (假阴性): {metrics['FN']}")
        print(f"  TN (真阴性): {metrics['TN']}")
    
    print("\n" + "=" * 100)


def save_per_class_table(per_class_metrics, overall_accuracy, output_path):
    """保存按类别的指标表格（CSV和Excel）"""
    # 准备数据
    data = []
    for metrics in per_class_metrics:
        data.append({
            '类别': metrics['class_name'],
            '类别索引': metrics['class_index'],
            '样本数': metrics['support'],
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'FN': metrics['FN'],
            'TN': metrics['TN'],
            '精确率 (%)': round(metrics['precision'] * 100, 2),
            '召回率 (%)': round(metrics['recall'] * 100, 2),
            'F1分数 (%)': round(metrics['f1_score'] * 100, 2),
            '特异性 (%)': round(metrics['specificity'] * 100, 2),
            'AUC': round(metrics['auc'], 4),
            '准确率 (%)': round(metrics['accuracy'] * 100, 2)
        })
    
    df = pd.DataFrame(data)
    
    # 添加整体准确率行
    summary_row = pd.DataFrame([{
        '类别': '整体',
        '类别索引': '-',
        '样本数': df['样本数'].sum(),
        'TP': '-',
        'FP': '-',
        'FN': '-',
        'TN': '-',
        '精确率 (%)': '-',
        '召回率 (%)': '-',
        'F1分数 (%)': '-',
        '特异性 (%)': '-',
        'AUC': '-',
        '准确率 (%)': round(overall_accuracy * 100, 2)
    }])
    
    df = pd.concat([df, summary_row], ignore_index=True)
    
    # 保存CSV
    csv_path = output_path.replace('.xlsx', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n已保存CSV表格: {csv_path}")
    
    # 保存Excel（如果openpyxl可用）
    if output_path.endswith('.xlsx'):
        try:
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"已保存Excel表格: {output_path}")
        except ImportError:
            print(f"警告: 未安装openpyxl，跳过Excel输出。请运行: pip install openpyxl")
        except Exception as e:
            print(f"警告: 保存Excel时出错: {e}")


def plot_per_class_comparison(per_class_metrics, output_path, figsize=(10, 6)):
    """绘制每个类别指标的对比图，每张图单独保存"""
    class_names = [m['class_name'] for m in per_class_metrics]
    
    # 提取指标
    precision = [m['precision'] * 100 for m in per_class_metrics]
    recall = [m['recall'] * 100 for m in per_class_metrics]
    f1 = [m['f1_score'] * 100 for m in per_class_metrics]
    specificity = [m['specificity'] * 100 for m in per_class_metrics]
    auc = [m['auc'] for m in per_class_metrics]
    support = [m['support'] for m in per_class_metrics]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    # 获取基础文件名（不含扩展名）
    base_path = output_path.rsplit('.', 1)[0]
    
    # 1. Precision, Recall, F1对比
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('分数 (%)', fontsize=12)
    ax.set_title('Precision, Recall, F1-Score 对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, 110])
    ax.set_yticks(np.arange(0, 111, 20))
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{base_path}_1_precision_recall_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {base_path}_1_precision_recall_f1.png")
    
    # 2. Specificity
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, specificity, alpha=0.8, color='green')
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('特异性 (%)', fontsize=12)
    ax.set_title('特异性 (Specificity)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, 110])
    ax.set_yticks(np.arange(0, 111, 20))
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{base_path}_2_specificity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {base_path}_2_specificity.png")
    
    # 3. AUC
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, auc, alpha=0.8, color='purple')
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('ROC AUC', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{base_path}_3_auc.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {base_path}_3_auc.png")
    
    # 4. 样本数分布
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, support, alpha=0.8, color='orange')
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('样本数', fontsize=12)
    ax.set_title('各类别样本数', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    max_support = max(support) if support else 100
    ax.set_ylim([0, max_support * 1.1])
    y_ticks = np.linspace(0, max_support, 6)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{base_path}_4_sample_count.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {base_path}_4_sample_count.png")
    
    # 5. 综合指标
    composite_scores = []
    for i in range(len(class_names)):
        score = (precision[i] + recall[i] + f1[i] + specificity[i]) / 4
        composite_scores.append(score)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, composite_scores, alpha=0.8, color='red')
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('综合分数 (%)', fontsize=12)
    ax.set_title('综合指标', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, 110])
    ax.set_yticks(np.arange(0, 111, 20))
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{base_path}_5_composite_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {base_path}_5_composite_score.png")
    
    # 6. 热力图：各类别指标矩阵
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
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('指标', fontsize=12)
    ax.set_title('指标热力图', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('数值', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    
    # 添加数值标注
    for i in range(5):
        for j in range(len(class_names)):
            ax.text(j, i, f'{metrics_matrix[i, j]:.1f}',
                   ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{base_path}_6_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存图表: {base_path}_6_heatmap.png")


def evaluate(args):
    """主评估函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, checkpoint_info = load_model(args.model_path, device, num_classes=args.num_classes)
    
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
    
    print(f"\n类别: {class_names}")
    
    # 数据变换
    _, val_transform = get_data_augmentation('none', img_size=args.img_size)
    
    # 加载数据集
    dataset = ImageFolderDataset(args.data_dir, transform=val_transform)
    
    # 数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 评估模型
    print("\n开始评估...")
    y_pred, y_true, y_probs = evaluate_model(
        model, dataloader, device, num_classes
    )
    
    # 计算整体准确率
    overall_accuracy = accuracy_score(y_true, y_pred)
    
    # 计算每个类别的指标
    print("\n计算每个类别的详细指标...")
    per_class_metrics, confusion_mat = calculate_per_class_metrics(
        y_true, y_pred, y_probs, num_classes, class_names
    )
    
    # 打印报告
    print_per_class_report(per_class_metrics, overall_accuracy)
    
    # 保存结果
    print(f"\n保存结果到: {args.output_dir}")
    
    # 保存JSON
    results = {
        'model_name': checkpoint_info['model_name'],
        'model_path': args.model_path,
        'overall_accuracy': float(overall_accuracy),
        'num_classes': num_classes,
        'class_names': class_names,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': confusion_mat.tolist()
    }
    
    json_path = os.path.join(args.output_dir, 'per_class_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"已保存JSON结果: {json_path}")
    
    # 保存表格
    table_path = os.path.join(args.output_dir, 'per_class_metrics.xlsx')
    save_per_class_table(per_class_metrics, overall_accuracy, table_path)
    
    # 绘制可视化
    plot_path = os.path.join(args.output_dir, 'per_class_comparison.png')
    plot_per_class_comparison(per_class_metrics, plot_path)
    
    print(f"\n评估完成！结果已保存到: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='按类别评估模型性能')
    
    # 必需参数
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型检查点路径 (.pth文件)')
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
    parser.add_argument('--output-dir', type=str, default='per_class_evaluation',
                        help='输出目录')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == '__main__':
    main()

