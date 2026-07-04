"""
交叉验证模型 Grad-CAM 可视化脚本
针对每个fold的最佳模型，为其对应的验证集生成Grad-CAM热力图
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

# 导入模型和工具
from train_multiclass import ImageFolderDataset, create_model, get_data_augmentation
from gradcam_visualization import GradCAM, get_target_layers

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_cv_results(cv_dir: Path) -> dict:
    """
    加载交叉验证结果
    
    Args:
        cv_dir: 交叉验证结果目录（包含cv_summary.json和folds_info.json）
    
    Returns:
        dict: 包含cv_summary和folds_info的字典
    """
    cv_summary_path = cv_dir / "cv_summary.json"
    folds_info_path = cv_dir / "folds_info.json"
    
    if not cv_summary_path.exists():
        raise FileNotFoundError(f"未找到cv_summary.json: {cv_summary_path}")
    if not folds_info_path.exists():
        raise FileNotFoundError(f"未找到folds_info.json: {folds_info_path}")
    
    with open(cv_summary_path, 'r') as f:
        cv_summary = json.load(f)
    
    with open(folds_info_path, 'r') as f:
        folds_info = json.load(f)
    
    return {
        'cv_summary': cv_summary,
        'folds_info': folds_info
    }


def get_val_samples_from_fold(fold_info: dict) -> List[int]:
    """
    根据fold信息获取验证集样本索引
    
    Args:
        fold_info: fold信息字典（包含val_indices）
    
    Returns:
        List[int]: 验证集样本索引列表
    """
    if 'val_indices' in fold_info:
        return fold_info['val_indices']
    else:
        raise ValueError(f"fold_info中未找到val_indices: {fold_info.keys()}")


def load_best_model(checkpoint_path: Path, model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    加载最佳模型检查点
    
    Args:
        checkpoint_path: 检查点路径
        model_name: 模型名称
        num_classes: 类别数（如果检查点中有class_to_idx，将使用其长度）
        device: 设备
    
    Returns:
        model: 加载的模型
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 从检查点获取类别数（如果存在）
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'class_to_idx' in checkpoint:
            num_classes_from_ckpt = len(checkpoint['class_to_idx'])
            if num_classes_from_ckpt != num_classes:
                print(f"  警告: 检查点中的类别数 ({num_classes_from_ckpt}) 与数据集类别数 ({num_classes}) 不同，使用检查点的类别数")
                num_classes = num_classes_from_ckpt
    else:
        state_dict = checkpoint
        # 尝试从head的权重推断类别数
        if 'head.weight' in state_dict:
            num_classes = state_dict['head.weight'].shape[0]
            print(f"  从检查点推断类别数: {num_classes}")
    
    # 创建模型
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model = model.to(device)
    
    # 加载权重，允许部分匹配
    try:
        if 'model_state_dict' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            epoch = checkpoint.get('epoch', 'unknown')
            val_acc = checkpoint.get('val_acc', 'unknown')
            print(f"  加载检查点: Epoch {epoch}, Val Acc: {val_acc:.2f}%")
            if missing_keys:
                print(f"  警告: 缺失的键 ({len(missing_keys)} 个): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  警告: 缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"  警告: 意外的键 ({len(unexpected_keys)} 个): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  警告: 意外的键: {unexpected_keys}")
        else:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"  加载检查点（仅权重）")
            if missing_keys:
                print(f"  警告: 缺失的键: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"  警告: 意外的键: {len(unexpected_keys)} 个")
    except Exception as e:
        print(f"  错误: 加载权重时出错: {e}")
        raise
    
    model.eval()
    return model


def visualize_gradcam_for_fold(
    fold_num: int,
    model_dir: Path,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    num_samples_per_class: Optional[int] = None,
    target_classes: Optional[List[int]] = None
):
    """
    为单个fold生成Grad-CAM可视化
    
    Args:
        fold_num: fold编号 (1-5)
        model_dir: 包含fold_X目录的模型目录
        data_dir: 数据目录（完整数据集）
        output_dir: 输出目录
        device: 设备
        num_samples_per_class: 每个类别生成多少张热力图（None表示全部）
        target_classes: 目标类别列表（None表示所有类别）
    """
    print(f"\n{'='*60}")
    print(f"处理 Fold {fold_num}")
    print(f"{'='*60}")
    
    # 加载交叉验证结果
    cv_results = load_cv_results(model_dir)
    cv_summary = cv_results['cv_summary']
    folds_info = cv_results['folds_info']
    
    # 获取fold信息
    fold_key = f"fold_{fold_num}"
    if fold_key not in folds_info:
        print(f"  警告: 未找到{fold_key}，跳过")
        return
    
    fold_info = folds_info[fold_key]
    val_indices = get_val_samples_from_fold(fold_info)
    
    # 获取fold结果
    fold_results = None
    for fr in cv_summary['fold_results']:
        if fr['fold'] == fold_num:
            fold_results = fr
            break
    
    if fold_results is None:
        print(f"  警告: 未找到fold {fold_num}的结果，跳过")
        return
    
    print(f"  最佳Epoch: {fold_results['best_epoch']}")
    print(f"  最佳验证准确率: {fold_results['best_val_acc']:.2f}%")
    print(f"  验证集大小: {len(val_indices)}")
    
    # 加载数据集
    print(f"  加载数据集: {data_dir}")
    _, val_transform = get_data_augmentation(augmentation_type='standard', img_size=224)
    full_dataset = ImageFolderDataset(data_dir, transform=val_transform)
    
    # 创建验证集子集
    val_subset = Subset(full_dataset, val_indices)
    num_classes = len(full_dataset.class_to_idx)
    class_names = list(full_dataset.class_to_idx.keys())
    
    print(f"  类别数: {num_classes}")
    print(f"  类别: {class_names}")
    
    # 加载模型
    fold_dir = model_dir / f"fold_{fold_num}"
    checkpoint_path = fold_dir / "best_model.pth"
    print(f"  加载模型: {checkpoint_path}")
    
    model = load_best_model(
        checkpoint_path,
        cv_summary['model'],
        num_classes,
        device
    )
    
    # 获取目标层
    target_layers = get_target_layers(model, cv_summary['model'])
    print(f"  目标层: {[str(layer) for layer in target_layers]}")
    
    # 创建Grad-CAM
    gradcam = GradCAM(model, target_layers, use_cuda=device.type == 'cuda')
    
    # 创建输出目录
    fold_output_dir = output_dir / f"fold_{fold_num}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 按类别组织样本
    # 注意：val_subset返回的idx是子集的索引，需要映射回原始数据集索引
    samples_by_class = {}
    for subset_idx, (img, label) in enumerate(val_subset):
        # 获取原始数据集索引
        original_idx = val_indices[subset_idx]
        label_idx = int(label)
        if label_idx not in samples_by_class:
            samples_by_class[label_idx] = []
        samples_by_class[label_idx].append((subset_idx, original_idx))
    
    # 确定要处理的类别
    if target_classes is None:
        target_classes = sorted(samples_by_class.keys())
    
    # 处理每个类别
    total_processed = 0
    for class_idx in target_classes:
        if class_idx not in samples_by_class:
            print(f"  警告: 类别 {class_idx} ({class_names[class_idx]}) 在验证集中不存在，跳过")
            continue
        
        class_name = class_names[class_idx]
        class_samples = samples_by_class[class_idx]
        
        # 限制每个类别的样本数
        if num_samples_per_class is not None:
            class_samples = class_samples[:num_samples_per_class]
        
        print(f"\n  处理类别 {class_idx}: {class_name} ({len(class_samples)} 个样本)")
        
        # 为每个样本生成热力图
        for sample_idx, (subset_idx, original_idx) in enumerate(tqdm(class_samples, desc=f"    {class_name}")):
            # 获取原始样本
            img, label = full_dataset[original_idx]
            img_path = full_dataset.samples[original_idx][0]
            
            # 准备输入
            input_tensor = img.unsqueeze(0).to(device)
            
            # 生成Grad-CAM
            try:
                cam = gradcam(input_tensor, target_category=int(label))
                
                # 加载原始图像（用于可视化）
                original_img = Image.open(img_path).convert('RGB')
                original_img_np = np.array(original_img)
                
                # 调整CAM大小
                cam_resized = cv2.resize(cam, (original_img_np.shape[1], original_img_np.shape[0]))
                
                # 生成热力图（叠加在原图上）
                # 将原图归一化到[0,1]
                img_normalized = original_img_np.astype(np.float32) / 255.0
                
                # 生成彩色热力图
                heatmap = cm.jet(cam_resized)[:, :, :3]
                
                # 叠加：60%原图 + 40%热力图
                superimposed = 0.6 * img_normalized + 0.4 * heatmap
                
                # 创建单个图像（无边框、无标题）
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(superimposed)
                ax.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
                # 保存图像
                img_name = Path(img_path).stem
                # 确保文件名安全（移除特殊字符）
                safe_class_name = class_name.replace('/', '_').replace('\\', '_')
                save_path = fold_output_dir / f"class_{class_idx}_{safe_class_name}" / f"{img_name}_fold{fold_num}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                
                total_processed += 1
                
            except Exception as e:
                print(f"    错误: 处理样本 {img_path} 时出错: {e}")
                continue
        
        print(f"    完成类别 {class_name}: 生成了 {len(class_samples)} 张热力图")
    
    print(f"\n  Fold {fold_num} 完成: 总共生成了 {total_processed} 张Grad-CAM热力图")
    print(f"  输出目录: {fold_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='为交叉验证模型生成Grad-CAM热力图')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='交叉验证模型目录（包含cv_summary.json和fold_X目录）')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='完整数据集目录')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认为model_dir/gradcam_output）')
    parser.add_argument('--folds', type=str, default='all',
                        help='要处理的fold编号，用逗号分隔（如1,2,3）或"all"处理所有fold')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='每个类别生成的样本数（默认：全部）')
    parser.add_argument('--target-classes', type=str, default=None,
                        help='目标类别索引，用逗号分隔（如0,1,2）。默认：所有类别')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备（cuda:0, cuda:1, cpu等）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 路径处理
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_dir / "gradcam_output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型目录: {model_dir}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 解析fold列表
    if args.folds.lower() == 'all':
        folds_to_process = [1, 2, 3, 4, 5]
    else:
        folds_to_process = [int(x.strip()) for x in args.folds.split(',')]
    
    # 解析目标类别
    target_classes = None
    if args.target_classes:
        target_classes = [int(x.strip()) for x in args.target_classes.split(',')]
    
    # 处理每个fold
    for fold_num in folds_to_process:
        try:
            visualize_gradcam_for_fold(
                fold_num=fold_num,
                model_dir=model_dir,
                data_dir=data_dir,
                output_dir=output_dir,
                device=device,
                num_samples_per_class=args.num_samples,
                target_classes=target_classes
            )
        except Exception as e:
            print(f"\n错误: 处理fold {fold_num}时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("所有fold处理完成！")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

