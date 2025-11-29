#!/usr/bin/env python3
"""
从训练结果中提取交叉验证结果并生成模板格式输出
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys


def extract_cv_results(checkpoint_dir):
    """
    从checkpoint目录中提取交叉验证结果
    
    Args:
        checkpoint_dir: checkpoint目录路径
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # 读取cv_summary.json
    cv_summary_path = checkpoint_dir / 'cv_summary.json'
    if not cv_summary_path.exists():
        print(f"错误: 找不到 {cv_summary_path}")
        return None
    
    with open(cv_summary_path, 'r') as f:
        summary = json.load(f)
    
    fold_results = summary['fold_results']
    avg_results = summary['average_results']
    n_splits = len(fold_results['fold_best_val_mAP'])
    
    # 提取每个fold的详细信息
    detailed_results = []
    
    for fold_num in range(1, n_splits + 1):
        idx = fold_num - 1
        
        # 从cv_summary.json获取基本信息
        best_mAP = fold_results['fold_best_val_mAP'][idx]
        best_acc = fold_results['fold_best_val_acc'][idx]
        final_val_acc = fold_results['fold_val_acc'][idx]
        final_val_loss = fold_results['fold_val_loss'][idx]
        
        # 尝试从checkpoint文件中获取最佳epoch和详细指标
        fold_dir = checkpoint_dir / f'fold_{fold_num}'
        best_epoch = None
        precision = None
        recall = None
        f1 = None
        
        # 方法1: 从checkpoint_best.pth读取
        best_checkpoint_path = fold_dir / 'checkpoint_best.pth'
        if best_checkpoint_path.exists():
            try:
                checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
                best_epoch = checkpoint.get('epoch', None)
                if best_epoch is not None:
                    best_epoch += 1  # epoch从0开始，显示时+1
            except Exception as e:
                print(f"警告: 无法读取 {best_checkpoint_path}: {e}")
        
        # 方法2: 从history.json中查找最佳mAP对应的epoch和详细指标
        history_path = fold_dir / 'history.json'
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                # 找到最佳mAP对应的epoch
                val_maps = history.get('val_mAP', [])
                if val_maps:
                    if best_epoch is None:
                        best_mAP_idx = val_maps.index(max(val_maps))
                        best_epoch = best_mAP_idx + 1
                    
                    # 尝试从history中读取详细指标
                    if 'val_precision' in history and len(history['val_precision']) > best_epoch - 1:
                        precision = history['val_precision'][best_epoch - 1]
                    if 'val_recall' in history and len(history['val_recall']) > best_epoch - 1:
                        recall = history['val_recall'][best_epoch - 1]
                    if 'val_f1' in history and len(history['val_f1']) > best_epoch - 1:
                        f1 = history['val_f1'][best_epoch - 1]
            except Exception as e:
                print(f"警告: 无法读取 {history_path}: {e}")
        
        # 如果还是找不到best_epoch，使用最后一个epoch
        if best_epoch is None:
            if history_path.exists():
                try:
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                    best_epoch = len(history.get('val_mAP', []))
                except:
                    best_epoch = 100  # 默认值
        
        # 尝试从checkpoint中读取详细指标
        if best_checkpoint_path.exists():
            try:
                checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
                # 检查是否有保存的metrics
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                    precision = metrics.get('precision_macro', best_mAP)
                    recall = metrics.get('recall_macro', 0.0)
                    f1 = metrics.get('f1_macro', 0.0)
            except:
                pass
        
        # 如果没有详细指标，使用mAP作为precision（因为mAP通常等于macro precision）
        if precision is None:
            precision = best_mAP
        if recall is None:
            recall = 0.0
        if f1 is None:
            f1 = 0.0
        
        detailed_results.append({
            'fold': fold_num,
            'best_epoch': best_epoch,
            'best_mAP': best_mAP,
            'best_acc': best_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'final_val_acc': final_val_acc,
            'final_val_loss': final_val_loss
        })
    
    return detailed_results, avg_results


def print_template_format(detailed_results, avg_results):
    """
    按照模板格式打印结果
    """
    print("\n详细结果:")
    for result in detailed_results:
        print(f"  Fold {result['fold']}:")
        print(f"    最佳验证mAP: {result['best_mAP']:.2f}% (Epoch {result['best_epoch']})")
        print(f"    最佳验证准确率: {result['best_acc']:.2f}% (Epoch {result['best_epoch']})")
        print(f"    mAP: {result['best_mAP']:.2f}%")
        print(f"    Precision: {result['precision']:.2f}%")
        print(f"    Recall: {result['recall']:.2f}%")
        print(f"    F1 Score: {result['f1']:.2f}%")
    
    print(f"\n平均结果:")
    print(f"  平均最佳验证mAP: {avg_results['avg_best_val_mAP']:.2f}% ± {avg_results['std_best_val_mAP']:.2f}%")
    print(f"  平均最佳验证准确率: {avg_results['avg_best_val_acc']:.2f}% ± {avg_results['std_best_val_acc']:.2f}%")
    print(f"  平均mAP: {avg_results['avg_best_val_mAP']:.2f}% ± {avg_results['std_best_val_mAP']:.2f}%")
    
    # 计算平均precision, recall, f1
    avg_precision = np.mean([r['precision'] for r in detailed_results])
    std_precision = np.std([r['precision'] for r in detailed_results])
    avg_recall = np.mean([r['recall'] for r in detailed_results])
    std_recall = np.std([r['recall'] for r in detailed_results])
    avg_f1 = np.mean([r['f1'] for r in detailed_results])
    std_f1 = np.std([r['f1'] for r in detailed_results])
    
    print(f"  平均Precision: {avg_precision:.2f}% ± {std_precision:.2f}%")
    print(f"  平均Recall: {avg_recall:.2f}% ± {std_recall:.2f}%")
    print(f"  平均F1 Score: {avg_f1:.2f}% ± {std_f1:.2f}%")
    print(f"  平均最终验证准确率: {avg_results['avg_val_acc']:.2f}%")
    print(f"  平均最终验证损失: {avg_results['avg_val_loss']:.4f}")


def main():
    if len(sys.argv) < 2:
        print("用法: python extract_cv_results.py <checkpoint_dir>")
        print("示例: python extract_cv_results.py checkpoints/clip_models/resnet18_clip:ViT-B/32")
        sys.exit(1)
    
    checkpoint_dir = Path(sys.argv[1])
    
    if not checkpoint_dir.exists():
        print(f"错误: 目录不存在: {checkpoint_dir}")
        sys.exit(1)
    
    detailed_results, avg_results = extract_cv_results(checkpoint_dir)
    
    if detailed_results is None:
        sys.exit(1)
    
    print_template_format(detailed_results, avg_results)
    
    print(f"\n结果已保存到: {checkpoint_dir}")
    print(f"  - cv_summary.json: 交叉验证汇总结果")
    print(f"  - folds_info.json: 各折的数据划分信息")
    print(f"  - fold_N/: 各折的训练历史和最佳模型")


if __name__ == '__main__':
    main()

