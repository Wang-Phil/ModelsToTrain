"""
模型性能评估脚本
支持多分类任务的全面评估，包括准确率、精确率、召回率、F1分数、混淆矩阵、ROC曲线等
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# 导入训练脚本中的模型和数据加载器
from train_multiclass import (
    ImageFolderDataset, create_model, get_data_augmentation
)

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_model(checkpoint_path, device, num_classes=None):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 设备
        num_classes: 类别数（如果checkpoint中没有保存）
    
    Returns:
        model: 加载的模型
        checkpoint_info: 检查点信息
    """
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
    model.load_state_dict(checkpoint['model_state_dict'])
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
    """
    评估模型，返回预测结果和真实标签
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_classes: 类别数
    
    Returns:
        all_preds: 所有预测结果
        all_labels: 所有真实标签
        all_probs: 所有预测概率
        inference_times: 推理时间列表
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='评估中'):
            images = images.to(device)
            labels = labels.to(device)
            
            # 记录推理时间
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), inference_times


def calculate_metrics(y_true, y_pred, y_probs, num_classes, class_names=None):
    """
    计算各种评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_probs: 预测概率
        num_classes: 类别数
        class_names: 类别名称列表
    
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 基本指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 加权平均指标
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 宏平均指标
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 微平均指标
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # 每个类别的指标
    metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # 计算特异性（Specificity）
    cm = metrics['confusion_matrix']
    per_class_specificity = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        per_class_specificity.append(specificity)
    metrics['per_class_specificity'] = np.array(per_class_specificity)
    metrics['specificity_weighted'] = np.average(
        per_class_specificity,
        weights=cm.sum(axis=1) / cm.sum()
    )
    
    # ROC AUC（多分类）
    try:
        y_true_binary = label_binarize(y_true, classes=list(range(num_classes)))
        
        # 微平均AUC
        if num_classes == 2:
            metrics['roc_auc_micro'] = roc_auc_score(y_true, y_probs[:, 1])
        else:
            metrics['roc_auc_micro'] = roc_auc_score(
                y_true_binary, y_probs, average='micro'
            )
        
        # 宏平均AUC
        metrics['roc_auc_macro'] = roc_auc_score(
            y_true_binary, y_probs, average='macro'
        )
        
        # 每个类别的AUC
        per_class_auc = []
        for i in range(num_classes):
            if len(np.unique(y_true_binary[:, i])) > 1:
                auc_score = roc_auc_score(y_true_binary[:, i], y_probs[:, i])
                per_class_auc.append(auc_score)
            else:
                per_class_auc.append(0.0)
        metrics['per_class_auc'] = np.array(per_class_auc)
        
        # 计算ROC曲线数据
        metrics['roc_curves'] = {}
        for i in range(num_classes):
            if len(np.unique(y_true_binary[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_probs[:, i])
                metrics['roc_curves'][i] = {'fpr': fpr, 'tpr': tpr}
    except Exception as e:
        print(f"警告: 计算ROC AUC时出错: {e}")
        metrics['roc_auc_micro'] = 0.0
        metrics['roc_auc_macro'] = 0.0
        metrics['per_class_auc'] = np.zeros(num_classes)
        metrics['roc_curves'] = {}
    
    # 分类报告
    if class_names:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
    else:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path, figsize=(12, 10)):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 绘制热力图
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'}
    )
    
    plt.title('Confusion Matrix (Percentage)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 同时保存原始数值版本
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix (Count)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_count.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(roc_curves, class_names, save_path, figsize=(10, 8)):
    """
    绘制ROC曲线
    
    Args:
        roc_curves: ROC曲线数据字典
        class_names: 类别名称列表
        save_path: 保存路径
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        if i in roc_curves:
            fpr = roc_curves[i]['fpr']
            tpr = roc_curves[i]['tpr']
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{class_name} (AUC = {auc(fpr, tpr):.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Each Class', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(metrics, class_names, save_path, figsize=(14, 8)):
    """
    绘制每个类别的指标
    
    Args:
        metrics: 指标字典
        class_names: 类别名称列表
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    # Precision
    axes[0, 0].bar(x - width, metrics['per_class_precision'], width, label='Precision', alpha=0.8)
    axes[0, 0].bar(x, metrics['per_class_recall'], width, label='Recall', alpha=0.8)
    axes[0, 0].bar(x + width, metrics['per_class_f1'], width, label='F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('Class', fontsize=10)
    axes[0, 0].set_ylabel('Score', fontsize=10)
    axes[0, 0].set_title('Precision, Recall, and F1-Score per Class', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # Specificity
    axes[0, 1].bar(x, metrics['per_class_specificity'], alpha=0.8, color='green')
    axes[0, 1].set_xlabel('Class', fontsize=10)
    axes[0, 1].set_ylabel('Specificity', fontsize=10)
    axes[0, 1].set_title('Specificity per Class', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # AUC
    axes[1, 0].bar(x, metrics['per_class_auc'], alpha=0.8, color='purple')
    axes[1, 0].set_xlabel('Class', fontsize=10)
    axes[1, 0].set_ylabel('AUC', fontsize=10)
    axes[1, 0].set_title('ROC AUC per Class', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # 混淆矩阵热力图（简化版）
    cm = metrics['confusion_matrix']
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1, 1].figure.colorbar(im, ax=axes[1, 1])
    axes[1, 1].set(xticks=np.arange(len(class_names)),
                   yticks=np.arange(len(class_names)),
                   xticklabels=class_names,
                   yticklabels=class_names,
                   title='Confusion Matrix',
                   ylabel='True Label',
                   xlabel='Predicted Label')
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
    
    # 在格子中添加数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_metrics_summary(metrics, class_names):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        class_names: 类别名称列表
    """
    print("\n" + "=" * 80)
    print("评估结果摘要")
    print("=" * 80)
    
    print("\n【整体指标】")
    print(f"准确率 (Accuracy)              : {metrics['accuracy']*100:.2f}%")
    print(f"\n加权平均 (Weighted Average):")
    print(f"  精确率 (Precision)           : {metrics['precision_weighted']*100:.2f}%")
    print(f"  召回率 (Recall/Sensitivity)  : {metrics['recall_weighted']*100:.2f}%")
    print(f"  F1分数 (F1-Score)            : {metrics['f1_weighted']*100:.2f}%")
    print(f"  特异性 (Specificity)        : {metrics['specificity_weighted']*100:.2f}%")
    
    print(f"\n宏平均 (Macro Average):")
    print(f"  精确率 (Precision)           : {metrics['precision_macro']*100:.2f}%")
    print(f"  召回率 (Recall)              : {metrics['recall_macro']*100:.2f}%")
    print(f"  F1分数 (F1-Score)            : {metrics['f1_macro']*100:.2f}%")
    
    print(f"\nROC AUC:")
    print(f"  微平均 (Micro Average)       : {metrics['roc_auc_micro']:.4f}")
    print(f"  宏平均 (Macro Average)      : {metrics['roc_auc_macro']:.4f}")
    
    print("\n【各类别详细指标】")
    print(f"{'类别':<25} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'特异性':<10} {'AUC':<10}")
    print("-" * 80)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} "
              f"{metrics['per_class_precision'][i]*100:>8.2f}% "
              f"{metrics['per_class_recall'][i]*100:>8.2f}% "
              f"{metrics['per_class_f1'][i]*100:>8.2f}% "
              f"{metrics['per_class_specificity'][i]*100:>8.2f}% "
              f"{metrics['per_class_auc'][i]:>8.4f}")
    
    print("\n" + "=" * 80)


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
    
    # 确保类别顺序一致
    if dataset.class_to_idx != class_to_idx:
        print("警告: 数据集类别与模型类别不一致，尝试重新映射...")
        # 这里可以添加类别映射逻辑
    
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
    y_pred, y_true, y_probs, inference_times = evaluate_model(
        model, dataloader, device, num_classes
    )
    
    # 计算指标
    print("\n计算评估指标...")
    metrics = calculate_metrics(y_true, y_pred, y_probs, num_classes, class_names)
    
    # 打印摘要
    print_metrics_summary(metrics, class_names)
    
    # 推理时间统计
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    print(f"\n推理时间统计:")
    print(f"  平均推理时间: {avg_inference_time*1000:.2f} ms/样本")
    print(f"  总推理时间: {total_inference_time:.2f} 秒")
    print(f"  吞吐量: {len(dataset)/total_inference_time:.2f} 样本/秒")
    
    # 保存结果
    print(f"\n保存结果到: {args.output_dir}")
    
    # 保存指标到JSON
    metrics_to_save = {
        'accuracy': float(metrics['accuracy']),
        'precision_weighted': float(metrics['precision_weighted']),
        'recall_weighted': float(metrics['recall_weighted']),
        'f1_weighted': float(metrics['f1_weighted']),
        'specificity_weighted': float(metrics['specificity_weighted']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'roc_auc_micro': float(metrics['roc_auc_micro']),
        'roc_auc_macro': float(metrics['roc_auc_macro']),
        'per_class_precision': metrics['per_class_precision'].tolist(),
        'per_class_recall': metrics['per_class_recall'].tolist(),
        'per_class_f1': metrics['per_class_f1'].tolist(),
        'per_class_specificity': metrics['per_class_specificity'].tolist(),
        'per_class_auc': metrics['per_class_auc'].tolist(),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'classification_report': metrics['classification_report'],
        'inference_time': {
            'avg_ms': float(avg_inference_time * 1000),
            'total_sec': float(total_inference_time),
            'throughput': float(len(dataset) / total_inference_time)
        },
        'class_names': class_names,
        'num_samples': len(dataset)
    }
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    
    # 保存分类报告（文本格式）
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write("分类报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        f.write("\n\n混淆矩阵\n")
        f.write("=" * 80 + "\n\n")
        f.write(str(metrics['confusion_matrix']))
    
    # 绘制可视化
    print("生成可视化图表...")
    
    # 混淆矩阵
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # ROC曲线
    if metrics['roc_curves']:
        plot_roc_curves(
            metrics['roc_curves'],
            class_names,
            os.path.join(args.output_dir, 'roc_curves.png')
        )
    
    # 每个类别的指标
    plot_per_class_metrics(
        metrics,
        class_names,
        os.path.join(args.output_dir, 'per_class_metrics.png')
    )
    
    print(f"\n评估完成！结果已保存到: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='模型性能评估脚本')
    
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
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='输出目录')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == '__main__':
    main()

