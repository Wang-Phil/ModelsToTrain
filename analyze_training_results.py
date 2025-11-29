"""
分析训练结果脚本
可视化训练过程中的loss和准确率，并生成混淆矩阵
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

# 导入训练脚本中的模型和数据加载器
from train_multiclass import (
    ImageFolderDataset, create_model, get_data_augmentation
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_training_history(checkpoint_dir):
    """加载训练历史"""
    history_path = os.path.join(checkpoint_dir, 'history.json')
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"训练历史文件不存在: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return history, config


def plot_training_curves(history, output_dir, model_name="Model"):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title(f'{model_name} - Loss Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title(f'{model_name} - Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存: {output_path}")
    plt.close()


def load_model_for_evaluation(checkpoint_path, device):
    """加载模型用于评估"""
    print(f"\n加载模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型信息
    model_name = checkpoint.get('model_name', 'resnet50')
    class_to_idx = checkpoint.get('class_to_idx', {})
    num_classes = len(class_to_idx) if class_to_idx else 9
    
    # 创建模型
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else {}
    
    print(f"模型: {model_name}")
    print(f"类别数: {num_classes}")
    if checkpoint.get('epoch'):
        print(f"训练轮数: {checkpoint['epoch']}")
    if checkpoint.get('val_acc'):
        print(f"验证准确率: {checkpoint['val_acc']:.2f}%")
    
    return model, class_to_idx, idx_to_class


def evaluate_and_generate_confusion_matrix(model, dataloader, device, class_names, output_dir):
    """评估模型并生成混淆矩阵"""
    print("\n开始评估模型...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n总体准确率: {accuracy * 100:.2f}%")
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存混淆矩阵
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n混淆矩阵已保存: {cm_path}")
    plt.close()
    
    # 计算归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存归一化混淆矩阵
    cm_norm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    print(f"归一化混淆矩阵已保存: {cm_norm_path}")
    plt.close()
    
    return accuracy, cm


def main():
    parser = argparse.ArgumentParser(description='分析训练结果')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='检查点目录路径（包含history.json和best_model.pth）')
    parser.add_argument('--val-dir', type=str, default=None,
                        help='验证数据目录（如果与config.json中的不同）')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备 (cuda:0, cuda:1, cpu)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认与checkpoint-dir相同）')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = args.checkpoint_dir
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载训练历史
    print("=" * 60)
    print("加载训练历史...")
    print("=" * 60)
    history, config = load_training_history(args.checkpoint_dir)
    
    model_name = config.get('model', 'Model')
    print(f"模型: {model_name}")
    print(f"训练轮数: {len(history['train_loss'])}")
    print(f"最佳训练准确率: {max(history['train_acc']):.2f}%")
    print(f"最佳验证准确率: {max(history['val_acc']):.2f}%")
    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
    
    # 绘制训练曲线
    print("\n" + "=" * 60)
    print("绘制训练曲线...")
    print("=" * 60)
    plot_training_curves(history, args.output_dir, model_name)
    
    # 加载模型进行评估
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"\n警告: 未找到best_model.pth，跳过混淆矩阵生成")
        return
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载模型
    model, class_to_idx, idx_to_class = load_model_for_evaluation(checkpoint_path, device)
    
    # 获取类别名称
    if idx_to_class:
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        class_names = [f'Class {i}' for i in range(len(class_to_idx))]
    
    # 准备验证数据
    val_dir = args.val_dir if args.val_dir else config.get('val_dir', 'data/val')
    if not os.path.exists(val_dir):
        print(f"\n警告: 验证数据目录不存在: {val_dir}，跳过混淆矩阵生成")
        return
    
    # 数据增强（验证集只需要resize和normalize）
    _, val_transform = get_data_augmentation(
        augmentation_type='none',
        img_size=config.get('img_size', 224)
    )
    
    val_dataset = ImageFolderDataset(val_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # 生成混淆矩阵
    print("\n" + "=" * 60)
    print("生成混淆矩阵...")
    print("=" * 60)
    accuracy, cm = evaluate_and_generate_confusion_matrix(
        model, val_loader, device, class_names, args.output_dir
    )
    
    # 保存评估结果
    results = {
        'model': model_name,
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'best_train_acc': max(history['train_acc']),
        'best_val_acc': max(history['val_acc']),
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1]
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n评估结果已保存: {results_path}")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"输出文件保存在: {args.output_dir}")
    print("  - training_curves.png: 训练曲线")
    print("  - confusion_matrix.png: 混淆矩阵")
    print("  - confusion_matrix_normalized.png: 归一化混淆矩阵")
    print("  - evaluation_results.json: 评估结果")


if __name__ == '__main__':
    main()

