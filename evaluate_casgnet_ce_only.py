#!/usr/bin/env python3
"""
为 casgnet_s1_ce_only 模型生成 ROC 曲线和混淆矩阵
使用与训练时完全相同的 SupConClassifierNet 架构
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# 导入项目中的模块
sys.path.insert(0, '/home/ln/wangweicheng/ModelsTotrain')
from train_multiclass import ImageFolderDataset, get_data_augmentation
from train_casgnet_contrastive_newdata import SupConClassifierNet, build_supcon_encoder
from torch.utils.data import DataLoader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_checkpoint(checkpoint_path, device):
    """加载checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型信息
    model_name = ckpt.get('model', 'casgnet_s1')
    num_classes = int(ckpt.get('num_classes', 7))
    class_to_idx = ckpt.get('class_to_idx', {})
    
    # 创建类别名称列表
    if class_to_idx:
        class_names = [''] * num_classes
        for name, idx in class_to_idx.items():
            class_names[int(idx)] = name
    else:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    print(f"Model: {model_name}")
    print(f"Num classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    return ckpt, model_name, num_classes, class_names


def evaluate_and_visualize(checkpoint_path, val_dir, output_dir, device):
    """评估模型并生成可视化"""
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载checkpoint
    ckpt, model_name, num_classes, class_names = load_checkpoint(checkpoint_path, device)
    state_dict = ckpt.get('state_dict', ckpt)
    
    # 推断投影维度
    w0 = state_dict.get("proj.0.weight")
    w2 = state_dict.get("proj.2.weight")
    if w0 is not None and w2 is not None:
        proj_dim = int(w2.shape[0])
        hidden_dim = int(w0.shape[0])
    else:
        proj_dim = 128
        hidden_dim = 512
    
    print(f"Proj dim: {proj_dim}, Hidden dim: {hidden_dim}")
    
    # 使用与训练时相同的方式构建模型
    # 先构建 encoder
    encoder = build_supcon_encoder(model_name, num_classes=num_classes, pretrained=False)
    
    # 然后手动构建 SupConClassifierNet
    from torch import nn
    import torch.nn.functional as F
    
    class SimpleSupConNet(nn.Module):
        def __init__(self, encoder, proj_dim, hidden_dim):
            super().__init__()
            self.encoder = encoder
            d = int(encoder.in_channel)
            self.proj = nn.Sequential(
                nn.Linear(d, hidden_dim), 
                nn.ReLU(inplace=True), 
                nn.Linear(hidden_dim, proj_dim)
            )
        
        def forward(self, x):
            fe = self.encoder.forward_features(x)
            logits = self.encoder.head(fe)
            p = F.normalize(self.proj(fe), dim=1)
            return logits, p
        
        @torch.inference_mode()
        def forward_logits(self, x):
            return self.encoder.head(self.encoder.forward_features(x))
    
    model = SimpleSupConNet(encoder, proj_dim, hidden_dim)
    
    # 尝试加载状态字典
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded state dict with strict=True ✓")
    except RuntimeError as e:
        print(f"Warning: Loading with strict=False due to: {e}")
        model.load_state_dict(state_dict, strict=False)
        print("Loaded state dict with strict=False (some keys may be missing or unexpected)")
    
    model = model.to(device)
    model.eval()
    
    # 准备数据
    _, val_transform = get_data_augmentation(augmentation_type='standard', img_size=224)
    val_dataset = ImageFolderDataset(val_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nEvaluating on {len(val_dataset)} samples...")
    
    # 收集预测结果 - 使用 forward_logits 方法
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            # 使用 forward_logits 直接获取 logits
            logits = model.forward_logits(images)
            probs = torch.softmax(logits.float(), dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
    
    # 合并结果
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    # ========== 1. 绘制 ROC 曲线 ==========
    print("\nGenerating ROC curves...")
    
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    
    # 二值化标签用于多分类ROC
    y_true_bin = label_binarize(all_labels, classes=np.arange(num_classes))
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    macro_auc = 0
    valid_classes = 0
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        y_true_class = y_true_bin[:, i]
        
        # 检查是否有足够的样本
        if len(np.unique(y_true_class)) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(y_true_class, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        macro_auc += roc_auc
        valid_classes += 1
        
        ax_roc.plot(fpr, tpr, lw=2, color=color, 
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    if valid_classes > 0:
        macro_auc /= valid_classes
    
    # 添加对角线
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title(f'Multiclass ROC Curves\nMacro OvR AUC = {macro_auc:.4f}', 
                    fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=9)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    roc_path = output_dir / 'roc_curves.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves: {roc_path}")
    plt.close(fig_roc)
    
    # ========== 2. 绘制混淆矩阵 ==========
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(all_labels, all_preds)
    
    fig_cm, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 原始计数
    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.figure.colorbar(im1, ax=ax1)
    ax1.set(xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            title='Confusion Matrix (Counts)',
            ylabel='True Label',
            xlabel='Predicted Label')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在格子上标注数字
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax2.figure.colorbar(im2, ax=ax2)
    ax2.set(xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            title='Normalized Confusion Matrix',
            ylabel='True Label',
            xlabel='Predicted Label')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在格子上标注百分比
    thresh_norm = 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            ax2.text(j, i, format(cm_normalized[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh_norm else "black")
    
    plt.tight_layout()
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix: {cm_path}")
    plt.close(fig_cm)
    
    # ========== 3. 打印分类报告 ==========
    print("\nClassification Report:")
    print("=" * 80)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    print(report)
    
    # 保存分类报告
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n")
        f.write(report)
    print(f"\nSaved classification report: {report_path}")
    
    # ========== 4. 计算各类别AUC ==========
    print("\nPer-class AUC:")
    print("=" * 80)
    per_class_auc = {}
    for i, class_name in enumerate(class_names):
        y_true_class = y_true_bin[:, i]
        if len(np.unique(y_true_class)) >= 2:
            fpr, tpr, _ = roc_curve(y_true_class, all_probs[:, i])
            class_auc = auc(fpr, tpr)
            per_class_auc[class_name] = class_auc
            print(f"{class_name:30s}: {class_auc:.4f}")
    
    # 计算整体准确率
    accuracy = float((all_labels == all_preds).mean())
    
    # 保存结果
    results = {
        'model': model_name,
        'num_samples': len(all_labels),
        'num_classes': num_classes,
        'classes': class_names,
        'per_class_auc': per_class_auc,
        'macro_auc': macro_auc,
        'accuracy': accuracy,
        'classification_report': report
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved evaluation results: {results_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CASGNet model and generate visualizations')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--val-dir', type=str, required=True,
                       help='Path to validation directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as checkpoint parent dir)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 设置输出目录
    if args.output_dir is None:
        checkpoint_parent = Path(args.checkpoint).parent
        args.output_dir = str(checkpoint_parent / 'plots_val')
    
    # 运行评估
    evaluate_and_visualize(
        checkpoint_path=args.checkpoint,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        device=device
    )
