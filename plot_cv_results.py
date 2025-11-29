#!/usr/bin/env python3
"""
绘制五折交叉验证的损失和准确率曲线
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_fold_history(fold_dir):
    """加载单个fold的历史数据"""
    history_file = Path(fold_dir) / "history.json"
    if not history_file.exists():
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    return history

def plot_cv_results(checkpoint_dir, save_dir=None):
    """
    绘制五折交叉验证的结果
    
    Args:
        checkpoint_dir: checkpoint目录路径
        save_dir: 图片保存目录，如果为None则保存到checkpoint_dir
    """
    checkpoint_dir = Path(checkpoint_dir)
    if save_dir is None:
        save_dir = checkpoint_dir
    else:
        save_dir = Path(save_dir)
    
    # 加载所有fold的数据
    folds_data = {}
    for fold_idx in range(1, 6):
        fold_dir = checkpoint_dir / f"fold_{fold_idx}"
        history = load_fold_history(fold_dir)
        if history is not None:
            folds_data[fold_idx] = history
            print(f"Loaded fold_{fold_idx}: {len(history['train_loss'])} epochs")
    
    if not folds_data:
        print(f"No fold data found in {checkpoint_dir}")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据 - 找到所有fold中最长的epoch数
    num_epochs_list = [len(history['train_loss']) for history in folds_data.values()]
    num_epochs = max(num_epochs_list)
    print(f"Using {num_epochs} epochs (maximum across all folds)")
    epochs = np.arange(1, num_epochs + 1)
    
    # 保留原始数据，允许不同fold有不同的长度
    # ========== 绘制4个子图：训练损失、验证损失、训练准确率、验证准确率 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========== 子图1: 训练损失 ==========
    ax1 = axes[0, 0]
    train_losses_all = []
    for fold_idx, history in folds_data.items():
        train_loss = history['train_loss']
        fold_epochs = np.arange(1, len(train_loss) + 1)
        ax1.plot(fold_epochs, train_loss, label=f'Fold {fold_idx}', 
                linewidth=1.5, alpha=0.7)
        train_losses_all.append(train_loss)
    
    # 计算并绘制平均训练损失（只计算所有fold都有数据的epoch）
    # 找出所有fold都有数据的最大epoch数
    min_epochs = min(len(loss) for loss in train_losses_all)
    mean_train_loss = []
    std_train_loss = []
    mean_train_epochs = []
    for epoch_idx in range(min_epochs):  # 只遍历所有fold都有的epoch
        epoch_values = []
        for fold_loss in train_losses_all:
            epoch_values.append(fold_loss[epoch_idx])
        # 所有fold都有该epoch的数据，计算平均值
        if epoch_values and len(epoch_values) == len(train_losses_all):
            mean_train_loss.append(np.mean(epoch_values))
            std_train_loss.append(np.std(epoch_values))
            mean_train_epochs.append(epoch_idx + 1)
    
    mean_epochs = np.array(mean_train_epochs)
    # 只绘制有数据的epoch的平均值
    if mean_epochs.size > 0:
        ax1.plot(mean_epochs, mean_train_loss, 'k-', label='Mean', linewidth=2.5)
        ax1.fill_between(mean_epochs, 
                         np.array(mean_train_loss) - np.array(std_train_loss), 
                         np.array(mean_train_loss) + np.array(std_train_loss), 
                         alpha=0.2, color='black')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, num_epochs)
    
    # ========== 子图2: 验证损失 ==========
    ax2 = axes[0, 1]
    val_losses_all = []
    for fold_idx, history in folds_data.items():
        val_loss = history['val_loss']
        fold_epochs = np.arange(1, len(val_loss) + 1)
        ax2.plot(fold_epochs, val_loss, label=f'Fold {fold_idx}', 
                linewidth=1.5, alpha=0.7)
        val_losses_all.append(val_loss)
    
    # 计算并绘制平均验证损失（只计算所有fold都有数据的epoch）
    min_epochs = min(len(loss) for loss in val_losses_all)
    mean_val_loss = []
    std_val_loss = []
    mean_val_epochs = []
    for epoch_idx in range(min_epochs):  # 只遍历所有fold都有的epoch
        epoch_values = []
        for fold_loss in val_losses_all:
            epoch_values.append(fold_loss[epoch_idx])
        # 所有fold都有该epoch的数据，计算平均值
        if epoch_values and len(epoch_values) == len(val_losses_all):
            mean_val_loss.append(np.mean(epoch_values))
            std_val_loss.append(np.std(epoch_values))
            mean_val_epochs.append(epoch_idx + 1)
    
    mean_epochs = np.array(mean_val_epochs)
    # 只绘制有数据的epoch的平均值
    if mean_epochs.size > 0:
        ax2.plot(mean_epochs, mean_val_loss, 'k-', label='Mean', linewidth=2.5)
        ax2.fill_between(mean_epochs, 
                         np.array(mean_val_loss) - np.array(std_val_loss), 
                         np.array(mean_val_loss) + np.array(std_val_loss), 
                         alpha=0.2, color='black')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, num_epochs)
    
    # ========== 子图3: 训练准确率 ==========
    ax3 = axes[1, 0]
    train_accs_all = []
    for fold_idx, history in folds_data.items():
        train_acc = history['train_acc']
        fold_epochs = np.arange(1, len(train_acc) + 1)
        ax3.plot(fold_epochs, train_acc, label=f'Fold {fold_idx}', 
                linewidth=1.5, alpha=0.7)
        train_accs_all.append(train_acc)
    
    # 计算并绘制平均训练准确率（只计算所有fold都有数据的epoch）
    min_epochs = min(len(acc) for acc in train_accs_all)
    mean_train_acc = []
    std_train_acc = []
    mean_train_acc_epochs = []
    for epoch_idx in range(min_epochs):  # 只遍历所有fold都有的epoch
        epoch_values = []
        for fold_acc in train_accs_all:
            epoch_values.append(fold_acc[epoch_idx])
        # 所有fold都有该epoch的数据，计算平均值
        if epoch_values and len(epoch_values) == len(train_accs_all):
            mean_train_acc.append(np.mean(epoch_values))
            std_train_acc.append(np.std(epoch_values))
            mean_train_acc_epochs.append(epoch_idx + 1)
    
    mean_epochs = np.array(mean_train_acc_epochs)
    # 只绘制有数据的epoch的平均值
    if mean_epochs.size > 0:
        ax3.plot(mean_epochs, mean_train_acc, 'k-', label='Mean', linewidth=2.5)
        ax3.fill_between(mean_epochs, 
                         np.array(mean_train_acc) - np.array(std_train_acc), 
                         np.array(mean_train_acc) + np.array(std_train_acc), 
                         alpha=0.2, color='black')
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, num_epochs)
    
    # ========== 子图4: 验证准确率 ==========
    ax4 = axes[1, 1]
    val_accs_all = []
    for fold_idx, history in folds_data.items():
        val_acc = history['val_acc']
        fold_epochs = np.arange(1, len(val_acc) + 1)
        ax4.plot(fold_epochs, val_acc, label=f'Fold {fold_idx}', 
                linewidth=1.5, alpha=0.7)
        val_accs_all.append(val_acc)
    
    # 计算并绘制平均验证准确率（只计算所有fold都有数据的epoch）
    min_epochs = min(len(acc) for acc in val_accs_all)
    mean_val_acc = []
    std_val_acc = []
    mean_val_acc_epochs = []
    for epoch_idx in range(min_epochs):  # 只遍历所有fold都有的epoch
        epoch_values = []
        for fold_acc in val_accs_all:
            epoch_values.append(fold_acc[epoch_idx])
        # 所有fold都有该epoch的数据，计算平均值
        if epoch_values and len(epoch_values) == len(val_accs_all):
            mean_val_acc.append(np.mean(epoch_values))
            std_val_acc.append(np.std(epoch_values))
            mean_val_acc_epochs.append(epoch_idx + 1)
    
    mean_epochs = np.array(mean_val_acc_epochs)
    # 只绘制有数据的epoch的平均值
    if mean_epochs.size > 0:
        ax4.plot(mean_epochs, mean_val_acc, 'k-', label='Mean', linewidth=2.5)
        ax4.fill_between(mean_epochs, 
                         np.array(mean_val_acc) - np.array(std_val_acc), 
                         np.array(mean_val_acc) + np.array(std_val_acc), 
                         alpha=0.2, color='black')
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, num_epochs)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = save_dir / "cv_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    
    # 打印统计信息（基于每个fold的最后一个epoch）
    print("\n=== Cross-Validation Statistics ===")
    # 找到每个fold最后一个epoch的值
    final_train_losses = [loss[-1] for loss in train_losses_all]
    final_val_losses = [loss[-1] for loss in val_losses_all]
    final_train_accs = [acc[-1] for acc in train_accs_all]
    final_val_accs = [acc[-1] for acc in val_accs_all]
    
    # 找到所有epoch中的最佳验证准确率
    best_val_acc = -1
    best_fold = -1
    best_epoch = -1
    for fold_idx, val_acc in enumerate(val_accs_all):
        max_idx = np.argmax(val_acc)
        if val_acc[max_idx] > best_val_acc:
            best_val_acc = val_acc[max_idx]
            best_fold = list(folds_data.keys())[fold_idx]
            best_epoch = max_idx + 1
    
    print(f"Final Train Loss: {np.mean(final_train_losses):.4f} ± {np.std(final_train_losses):.4f}")
    print(f"Final Val Loss: {np.mean(final_val_losses):.4f} ± {np.std(final_val_losses):.4f}")
    print(f"Final Train Acc: {np.mean(final_train_accs):.2f}% ± {np.std(final_train_accs):.2f}%")
    print(f"Final Val Acc: {np.mean(final_val_accs):.2f}% ± {np.std(final_val_accs):.2f}%")
    print(f"Best Val Acc: {best_val_acc:.2f}% (Fold {best_fold}, Epoch {best_epoch})")
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    # 默认路径
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    else:
        checkpoint_dir = "/home/ln/wangweicheng/ModelsTotrain/checkpoints/cv_multi_models/starnet_s1"
    
    plot_cv_results(checkpoint_dir)

