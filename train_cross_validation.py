"""
五折交叉验证训练脚本
支持K折交叉验证，只在训练集使用数据增强
"""

import os
import sys
import time
import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def compute_drw_weights(cls_num_list, beta=0.9999):
    cls_num_list = np.array(cls_num_list, dtype=np.float32)
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / (effective_num + 1e-12)
    weights = weights / np.sum(weights) * len(cls_num_list)
    return weights

# 导入训练脚本中的模型和数据加载器
from train_multiclass import (
    ImageFolderDataset, create_model, get_data_augmentation,
    get_loss_function, train_epoch, validate
)
from calculate_metrics import (
    calculate_classification_metrics, calculate_model_complexity,
    evaluate_model_comprehensive
)


def create_folds_from_dataset(dataset, n_splits=5, shuffle=True, random_state=42):
    """
    从数据集创建K折交叉验证的折
    
    Args:
        dataset: ImageFolderDataset实例
        n_splits: 折数
        shuffle: 是否打乱
        random_state: 随机种子
    
    Returns:
        folds: 包含(train_indices, val_indices)的列表
    """
    # 获取所有样本的标签
    labels = [label for _, label in dataset.samples]
    
    # 使用分层K折（StratifiedKFold）确保每折中类别分布相同
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in skf.split(range(len(dataset)), labels):
        folds.append((train_idx, val_idx))
    
    return folds


def compute_class_counts_in_fold(base_dataset, train_indices):
    """
    计算当前fold训练集中每个类别的样本数量
    
    Args:
        base_dataset: 完整数据集
        train_indices: 训练集索引
    
    Returns:
        class_counts: 字典，{class_idx: count}
    """
    class_counts = defaultdict(int)
    for idx in train_indices:
        _, label = base_dataset.samples[idx]
        class_counts[label] += 1
    return dict(class_counts)


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


class DynamicAugmentationSubset(Dataset):
    """
    支持根据类别样本数量动态选择数据增强策略的Dataset
    
    规则：
    - 类别样本数 >= high_threshold: 无增强
    - low_threshold <= 类别样本数 < high_threshold: 标准增强
    - 类别样本数 < low_threshold: 强增强
    """
    def __init__(self, base_dataset, indices, class_counts, 
                 no_aug_transform, standard_transform, strong_transform,
                 low_threshold=50, high_threshold=300):
        """
        Args:
            base_dataset: 基础数据集
            indices: 样本索引
            class_counts: 每个类别的样本数量 {class_idx: count}
            no_aug_transform: 无增强的transform（用于样本数 >= high_threshold）
            standard_transform: 标准增强的transform（用于 low_threshold <= 样本数 < high_threshold）
            strong_transform: 强增强的transform（用于样本数 < low_threshold）
            low_threshold: 低阈值（默认50）
            high_threshold: 高阈值（默认300）
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.class_counts = class_counts
        self.no_aug_transform = no_aug_transform
        self.standard_transform = standard_transform
        self.strong_transform = strong_transform
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # 为每个类别确定增强策略
        self.class_aug_strategy = {}
        for class_idx, count in class_counts.items():
            if count >= high_threshold:
                self.class_aug_strategy[class_idx] = 'none'
            elif count >= low_threshold:
                self.class_aug_strategy[class_idx] = 'standard'
            else:
                self.class_aug_strategy[class_idx] = 'strong'
        
        # 统计信息
        self._print_augmentation_stats()
    
    def _print_augmentation_stats(self):
        """打印数据增强统计信息"""
        stats = {'none': 0, 'standard': 0, 'strong': 0}
        for strategy in self.class_aug_strategy.values():
            stats[strategy] += 1
        
        print(f"\n动态数据增强统计 (阈值: {self.low_threshold}/{self.high_threshold}):")
        print(f"  无增强类别数 (>= {self.high_threshold}): {stats['none']}")
        print(f"  标准增强类别数 ({self.low_threshold}-{self.high_threshold-1}): {stats['standard']}")
        print(f"  强增强类别数 (< {self.low_threshold}): {stats['strong']}")
        
        # 打印每个类别的增强策略
        class_names = list(self.base_dataset.class_to_idx.keys())
        idx_to_name = {v: k for k, v in self.base_dataset.class_to_idx.items()}
        print(f"\n各类别增强策略:")
        for class_idx in sorted(self.class_aug_strategy.keys()):
            class_name = idx_to_name.get(class_idx, f"Class_{class_idx}")
            count = self.class_counts.get(class_idx, 0)
            strategy = self.class_aug_strategy[class_idx]
            print(f"  {class_name} (样本数={count}): {strategy}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取原始数据
        img_path, label = self.base_dataset.samples[self.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        
        # 根据类别选择transform
        strategy = self.class_aug_strategy.get(label, 'standard')
        if strategy == 'none':
            transform = self.no_aug_transform
        elif strategy == 'strong':
            transform = self.strong_transform
        else:  # standard
            transform = self.standard_transform
        
        # 应用transform
        if transform:
            image = transform(image)
        
        return image, label


def train_fold(
    model,
    train_dataset,
    val_dataset,
    train_indices,
    val_indices,
    fold_num,
    args,
    device,
    class_to_idx,
    class_counts=None
):
    """
    训练一个折
    
    Args:
        model: 模型实例
        train_dataset: 完整训练数据集（包含所有数据）
        val_dataset: 验证数据集（用于最终评估，可选）
        train_indices: 当前折的训练集索引
        val_indices: 当前折的验证集索引
        fold_num: 折的编号（1-5）
        args: 训练参数
        device: 设备
    """
    print(f"\n{'='*80}")
    print(f"训练 Fold {fold_num}/{args.n_splits}")
    print(f"{'='*80}")
    
    # 使用传入的数据集（已经应用了正确的transform）
    # train_dataset是训练集（有数据增强），val_dataset是验证集（无数据增强）
    fold_train_dataset = train_dataset
    fold_val_dataset = val_dataset
    
    print(f"训练集大小: {len(fold_train_dataset)}")
    print(f"验证集大小: {len(fold_val_dataset)}")
    
    # 数据加载器
    use_cuda = device.type == 'cuda'
    if not use_cuda:
        num_workers = 0
    elif sys.platform == 'win32':
        num_workers = 0
    else:
        num_workers = args.num_workers
    
    train_loader = DataLoader(
        fold_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    
    val_loader = DataLoader(
        fold_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    
    # 损失函数
    num_classes = len(class_to_idx)
    if class_counts is None:
        cls_num_list = [0] * num_classes
    else:
        cls_num_list = [class_counts.get(idx, 0) for idx in range(num_classes)]
    loss_kwargs = {}
    if args.loss == 'focal':
        loss_kwargs['focal_gamma'] = args.focal_gamma
    elif args.loss == 'label_smoothing':
        loss_kwargs['smoothing'] = args.label_smoothing
    elif args.loss == 'asl':
        loss_kwargs['asl_gamma_neg'] = args.asl_gamma_neg
        loss_kwargs['asl_gamma_pos'] = args.asl_gamma_pos
        loss_kwargs['asl_clip'] = args.asl_clip
        loss_kwargs['asl_eps'] = args.asl_eps
    elif args.loss == 'ldam':
        loss_kwargs['cls_num_list'] = cls_num_list
        loss_kwargs['ldam_margin'] = args.ldam_margin
        loss_kwargs['ldam_s'] = args.ldam_s
    
    criterion = get_loss_function(args.loss, num_classes=num_classes, **loss_kwargs)
    criterion = criterion.to(device)
    drw_weights = None
    if args.use_drw and args.loss == 'ldam':
        drw_values = compute_drw_weights(cls_num_list, beta=args.drw_beta)
        drw_weights = torch.tensor(drw_values, dtype=torch.float32, device=device)
    drw_applied = False
    val_acc_recent = deque(maxlen=args.drw_window)
    
    # 优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    # 学习率调度器
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    if args.scheduler == 'onecycle':
        max_lr = args.onecycle_max_lr if args.onecycle_max_lr else args.lr
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.warmup_ratio,
            anneal_strategy=args.onecycle_anneal_strategy,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor
        )
        use_onecycle = True
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        use_onecycle = False
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        use_onecycle = False
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        use_onecycle = False
    else:
        scheduler = None
        use_onecycle = False
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_mAP': []
    }
    
    best_val_mAP = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    min_delta = args.early_stopping_min_delta
    # 默认patience设为20，如果用户指定了且>0则使用用户的值
    patience = args.early_stopping_patience if args.early_stopping_patience > 0 else 20
    
    print(f"\n开始训练 Fold {fold_num}...")
    print(f"学习率: {args.lr}")
    print(f"优化器: {args.optimizer}")
    print(f"损失函数: {args.loss}")
    use_amp = getattr(args, 'use_amp', True)
    if use_amp and device.type == 'cuda':
        print(f"AMP混合精度: 启用 (可减少30-40%显存)")
    else:
        print(f"AMP混合精度: 禁用")
    print("-" * 80)
    
    drw_applied = False
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        if args.use_drw and args.loss == 'ldam' and drw_weights is not None and epoch >= args.drw_start_epoch and not drw_applied:
            criterion.weight = drw_weights
            drw_applied = True
            print(f"DRW 权重在第 {epoch} 轮开启")
        # 训练（使用AMP混合精度）
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scheduler if use_onecycle else None,
            use_amp=getattr(args, 'use_amp', True),  # 默认启用AMP
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1)  # 梯度累积步数
        )
        
        # 验证（使用AMP混合精度）
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device,
            use_amp=getattr(args, 'use_amp', True)  # 默认启用AMP
        )
        val_acc_recent.append(val_acc)
        if args.use_drw and args.loss == 'ldam' and drw_weights is not None and not drw_applied:
            if epoch >= args.drw_start_epoch and len(val_acc_recent) == args.drw_window:
                diff = max(val_acc_recent) - min(val_acc_recent)
                if diff <= args.drw_threshold * 100:
                    criterion.weight = drw_weights
                    drw_applied = True
                    print(f"DRW 权重在 Fold {fold_num} 的 epoch {epoch} 启用 (diff={diff:.2f}%)")
        
        # 定期清理GPU内存，防止内存碎片化（每10个epoch清理一次）
        if device.type == 'cuda' and epoch % 10 == 0:
            torch.cuda.empty_cache()
        
        # 计算当前epoch的mAP
        val_metrics = calculate_classification_metrics(
            np.array(val_labels),
            np.array(val_preds),
            num_classes
        )
        val_mAP = val_metrics['mAP']
        
        # 更新学习率
        if scheduler and not use_onecycle:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_mAP'].append(val_mAP)
        
        # 保存最佳模型（基于mAP，适用于多分类任务）
        improved = False
        if val_mAP > best_val_mAP + min_delta:
            best_val_mAP = val_mAP
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_loss = val_loss
            # 保存最佳epoch的预测结果（用于后续计算详细指标）
            best_val_preds = val_preds.copy() if isinstance(val_preds, list) else val_preds
            best_val_labels = val_labels.copy() if isinstance(val_labels, list) else val_labels
            patience_counter = 0
            
            # 保存当前折的最佳模型
            fold_output_dir = Path(args.output_dir) / f'fold_{fold_num}'
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mAP': val_mAP,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_to_idx': class_to_idx,
                'model_name': args.model,
                'fold': fold_num
            }, fold_output_dir / 'best_model.pth')
        else:
            patience_counter += 1
        
        # Early Stopping（基于mAP，适用于多分类任务）
        if patience is not None and patience_counter >= patience:
            print(f"\nEarly Stopping触发! (Fold {fold_num})")
            print(f"最佳验证mAP: {best_val_mAP:.2f}% (Epoch {best_epoch})")
            print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
            break
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Fold {fold_num} - Epoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val mAP: {val_mAP:.2f}%")
            print(f"  Best Val mAP: {best_val_mAP:.2f}% (Epoch {best_epoch})")
            if scheduler:
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    total_time = time.time() - start_time
    
    # 保存训练历史
    fold_output_dir = Path(args.output_dir) / f'fold_{fold_num}'
    with open(fold_output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 计算模型复杂度（参数量和FLOPs）
    print(f"\n计算模型复杂度...")
    try:
        complexity = calculate_model_complexity(model, input_size=(1, 3, args.img_size, args.img_size), device=device)
        params_millions = complexity['params_millions']
        flops_millions = complexity['flops_millions']
    except Exception as e:
        print(f"警告: 无法计算模型复杂度: {e}")
        params_millions = 0.0
        flops_millions = 0.0
    
    # 计算最佳模型的详细指标（使用最佳epoch的预测结果）
    print(f"计算最佳模型的详细指标...")
    if 'best_val_preds' in locals() and 'best_val_labels' in locals():
        final_metrics = calculate_classification_metrics(
            np.array(best_val_labels),
            np.array(best_val_preds),
            num_classes
        )
        # 添加类别名称
        final_metrics['class_names'] = list(class_to_idx.keys())
    else:
        # 如果最佳预测结果不存在，重新评估
        checkpoint = torch.load(fold_output_dir / 'best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        final_metrics = evaluate_model_comprehensive(
            model, val_loader, criterion, device, num_classes,
            class_names=list(class_to_idx.keys())
        )
    
    print(f"\nFold {fold_num} 训练完成!")
    print(f"  最佳验证mAP: {best_val_mAP:.2f}% (Epoch {best_epoch}) [主要指标]")
    print(f"  最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  mAP: {final_metrics['mAP']:.2f}%")
    print(f"  Precision (Macro): {final_metrics['precision_macro']:.2f}%")
    print(f"  Recall (Macro): {final_metrics['recall_macro']:.2f}%")
    print(f"  F1 Score (Macro): {final_metrics['f1_macro']:.2f}%")
    print(f"  准确率: {final_metrics.get('accuracy', best_val_acc):.2f}%")
    print(f"  参数量: {params_millions:.2f}M")
    print(f"  FLOPs: {flops_millions:.2f}M")
    print(f"  训练时间: {total_time / 60:.2f} 分钟")
    
    return {
        'fold': fold_num,
        'best_val_mAP': best_val_mAP,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_val_acc': val_acc,
        'final_val_loss': val_loss,
        'mAP': final_metrics['mAP'],
        'precision_macro': final_metrics['precision_macro'],
        'recall_macro': final_metrics['recall_macro'],
        'f1_macro': final_metrics['f1_macro'],
        'params_millions': params_millions,
        'flops_millions': flops_millions,
        'metrics': final_metrics,
        'history': history
    }


def train_cross_validation(args):
    """主交叉验证训练函数"""
    # 设置随机种子
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"随机种子已设置为: {args.seed}")
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
        if 'cuda' in args.device and not torch.cuda.is_available():
            raise RuntimeError(f"指定的设备 {args.device} 不可用")
        if 'cuda' in args.device:
            gpu_id = int(args.device.split(':')[1]) if ':' in args.device else 0
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"指定的GPU {gpu_id} 不存在")
    elif args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(device)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据增强：准备三种增强策略用于动态选择
    # 无增强（用于样本数 >= high_threshold）
    no_aug_transform, val_transform = get_data_augmentation(
        augmentation_type='none',
        img_size=args.img_size
    )
    # 标准增强（用于 low_threshold <= 样本数 < high_threshold）
    standard_transform, _ = get_data_augmentation(
        augmentation_type='standard',
        img_size=args.img_size
    )
    # 强增强（用于样本数 < low_threshold）
    strong_transform, _ = get_data_augmentation(
        augmentation_type='strong',
        img_size=args.img_size
    )
    
    # 如果用户指定了特定的增强类型，使用该类型作为标准增强
    if args.augmentation != 'standard':
        standard_transform, _ = get_data_augmentation(
            augmentation_type=args.augmentation,
            img_size=args.img_size
        )
    
    # 加载完整数据集（用于交叉验证）
    # 注意：这里使用no_aug_transform作为基础transform，实际的transform会在创建子集时应用
    full_dataset = ImageFolderDataset(args.data_dir, transform=no_aug_transform)
    num_classes = len(full_dataset.class_to_idx)
    
    print(f"\n数据集信息:")
    print(f"  总样本数: {len(full_dataset)}")
    print(f"  类别数: {num_classes}")
    print(f"  类别: {list(full_dataset.class_to_idx.keys())}")
    
    # 创建K折
    print(f"\n创建 {args.n_splits} 折交叉验证...")
    folds = create_folds_from_dataset(
        full_dataset, 
        n_splits=args.n_splits, 
        shuffle=True, 
        random_state=args.seed or 42
    )
    
    # 保存折的信息
    folds_info = {}
    for i, (train_idx, val_idx) in enumerate(folds, 1):
        folds_info[f'fold_{i}'] = {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist()
        }
    
    with open(os.path.join(args.output_dir, 'folds_info.json'), 'w') as f:
        json.dump(folds_info, f, indent=2)
    
    # 存储所有折的结果
    all_fold_results = []
    
    # 训练每一折
    for fold_num, (train_indices, val_indices) in enumerate(folds, 1):
        # 计算当前fold训练集中每个类别的样本数
        fold_train_class_counts = compute_class_counts_in_fold(full_dataset, train_indices)
        fold_cls_num_list = [fold_train_class_counts.get(idx, 0) for idx in range(num_classes)]
        
        # 为当前折创建新的模型
        model_kwargs = {}
        # StarNet CF 模型需要 cls_num_list
        if 'starnet_cf' in args.model.lower():
            model_kwargs['cls_num_list'] = fold_cls_num_list
        
        model = create_model(
            args.model,
            num_classes=num_classes,
            pretrained=args.pretrained,
            **model_kwargs
        )
        model = model.to(device)
        
        # 创建支持动态数据增强的训练集
        if args.use_dynamic_augmentation:
            fold_train_dataset = DynamicAugmentationSubset(
                base_dataset=full_dataset,
                indices=train_indices,
                class_counts=fold_train_class_counts,
                no_aug_transform=no_aug_transform,
                standard_transform=standard_transform,
                strong_transform=strong_transform,
                low_threshold=args.dynamic_aug_low_threshold,
                high_threshold=args.dynamic_aug_high_threshold
            )
        else:
            # 使用传统方式：所有训练样本使用相同的增强策略
            fold_train_dataset = TransformSubset(
                full_dataset, 
                train_indices, 
                transform=standard_transform
            )
        
        # 验证集使用val_transform（无数据增强）
        fold_val_dataset = TransformSubset(full_dataset, val_indices, transform=val_transform)
        
        # 训练当前折
        fold_result = train_fold(
            model=model,
            train_dataset=fold_train_dataset,  # 训练集（有数据增强）
            val_dataset=fold_val_dataset,  # 验证集（无数据增强）
            train_indices=train_indices,
            val_indices=val_indices,
            fold_num=fold_num,
            args=args,
            device=device,
            class_to_idx=full_dataset.class_to_idx
            ,
            class_counts=full_dataset.class_counts
        )
        
        all_fold_results.append(fold_result)
        
        # 清理GPU内存
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # 计算平均结果
    print(f"\n{'='*80}")
    print("交叉验证结果汇总")
    print(f"{'='*80}")
    
    # 计算各项指标的平均值和标准差
    avg_val_mAP = np.mean([r['best_val_mAP'] for r in all_fold_results])
    std_val_mAP = np.std([r['best_val_mAP'] for r in all_fold_results])
    avg_val_acc = np.mean([r['best_val_acc'] for r in all_fold_results])
    std_val_acc = np.std([r['best_val_acc'] for r in all_fold_results])
    avg_final_val_acc = np.mean([r['final_val_acc'] for r in all_fold_results])
    avg_final_val_loss = np.mean([r['final_val_loss'] for r in all_fold_results])
    
    avg_mAP = np.mean([r['mAP'] for r in all_fold_results])
    std_mAP = np.std([r['mAP'] for r in all_fold_results])
    
    avg_precision = np.mean([r['precision_macro'] for r in all_fold_results])
    std_precision = np.std([r['precision_macro'] for r in all_fold_results])
    
    avg_recall = np.mean([r['recall_macro'] for r in all_fold_results])
    std_recall = np.std([r['recall_macro'] for r in all_fold_results])
    
    avg_f1 = np.mean([r['f1_macro'] for r in all_fold_results])
    std_f1 = np.std([r['f1_macro'] for r in all_fold_results])
    
    # 参数量和FLOPs（所有折相同，取第一个）
    params_millions = all_fold_results[0]['params_millions']
    flops_millions = all_fold_results[0]['flops_millions']
    
    # 打印表格格式的结果（多分类任务，突出mAP）
    print(f"\n{'='*100}")
    print(f"{'指标':<20} {'Fold 1':<12} {'Fold 2':<12} {'Fold 3':<12} {'Fold 4':<12} {'Fold 5':<12} {'平均±标准差':<20}")
    print(f"{'='*100}")
    
    # 首先显示mAP（主要指标）
    for i, result in enumerate(all_fold_results, 1):
        if i == 1:
            print(f"{'Best Val mAP (%)':<20} {result['best_val_mAP']:>10.2f}% ", end='')
        else:
            print(f"{'':<20} {result['best_val_mAP']:>10.2f}% ", end='')
    print(f"{avg_val_mAP:>6.2f}% ± {std_val_mAP:>5.2f}%")
    
    for i, result in enumerate(all_fold_results, 1):
        if i == 1:
            print(f"{'Best Val Acc (%)':<20} {result['best_val_acc']:>10.2f}% ", end='')
        else:
            print(f"{'':<20} {result['best_val_acc']:>10.2f}% ", end='')
    print(f"{avg_val_acc:>6.2f}% ± {std_val_acc:>5.2f}%")
    
    for i, result in enumerate(all_fold_results, 1):
        if i == 1:
            print(f"{'mAP (%)':<20} {result['mAP']:>10.2f}% ", end='')
        else:
            print(f"{'':<20} {result['mAP']:>10.2f}% ", end='')
    print(f"{avg_mAP:>6.2f}% ± {std_mAP:>5.2f}%")
    
    for i, result in enumerate(all_fold_results, 1):
        if i == 1:
            print(f"{'Precision (%)':<20} {result['precision_macro']:>10.2f}% ", end='')
        else:
            print(f"{'':<20} {result['precision_macro']:>10.2f}% ", end='')
    print(f"{avg_precision:>6.2f}% ± {std_precision:>5.2f}%")
    
    for i, result in enumerate(all_fold_results, 1):
        if i == 1:
            print(f"{'Recall (%)':<20} {result['recall_macro']:>10.2f}% ", end='')
        else:
            print(f"{'':<20} {result['recall_macro']:>10.2f}% ", end='')
    print(f"{avg_recall:>6.2f}% ± {std_recall:>5.2f}%")
    
    for i, result in enumerate(all_fold_results, 1):
        if i == 1:
            print(f"{'F1 Score (%)':<20} {result['f1_macro']:>10.2f}% ", end='')
        else:
            print(f"{'':<20} {result['f1_macro']:>10.2f}% ", end='')
    print(f"{avg_f1:>6.2f}% ± {std_f1:>5.2f}%")
    
    print(f"{'='*100}")
    print(f"{'Params (×10⁶)':<20} {params_millions:>10.2f}")
    print(f"{'FLOPs (×10⁶)':<20} {flops_millions:>10.2f}")
    print(f"{'='*100}")
    
    print(f"\n详细结果:")
    for result in all_fold_results:
        print(f"  Fold {result['fold']}:")
        print(f"    最佳验证mAP: {result['best_val_mAP']:.2f}% (Epoch {result['best_epoch']}) [主要指标]")
        print(f"    最佳验证准确率: {result['best_val_acc']:.2f}% (Epoch {result['best_epoch']})")
        print(f"    mAP: {result['mAP']:.2f}%")
        print(f"    Precision: {result['precision_macro']:.2f}%")
        print(f"    Recall: {result['recall_macro']:.2f}%")
        print(f"    F1 Score: {result['f1_macro']:.2f}%")
    
    print(f"\n平均结果:")
    print(f"  平均最佳验证mAP: {avg_val_mAP:.2f}% ± {std_val_mAP:.2f}% [主要指标]")
    print(f"  平均最佳验证准确率: {avg_val_acc:.2f}% ± {std_val_acc:.2f}%")
    print(f"  平均mAP: {avg_mAP:.2f}% ± {std_mAP:.2f}%")
    print(f"  平均Precision: {avg_precision:.2f}% ± {std_precision:.2f}%")
    print(f"  平均Recall: {avg_recall:.2f}% ± {std_recall:.2f}%")
    print(f"  平均F1 Score: {avg_f1:.2f}% ± {std_f1:.2f}%")
    print(f"  平均最终验证准确率: {avg_final_val_acc:.2f}%")
    print(f"  平均最终验证损失: {avg_final_val_loss:.4f}")
    
    # 保存汇总结果（多分类任务，mAP为主要指标）
    summary = {
        'n_splits': args.n_splits,
        'model': args.model,
        'pretrained': args.pretrained,
        'optimizer': args.optimizer,
        'loss': args.loss,
        'lr': args.lr,
        'augmentation': args.augmentation,
        'primary_metric': 'mAP',  # 标记主要指标为mAP
        'average_best_val_mAP': float(avg_val_mAP),
        'std_best_val_mAP': float(std_val_mAP),
        'average_best_val_acc': float(avg_val_acc),
        'std_best_val_acc': float(std_val_acc),
        'average_final_val_acc': float(avg_final_val_acc),
        'average_final_val_loss': float(avg_final_val_loss),
        'average_mAP': float(avg_mAP),
        'std_mAP': float(std_mAP),
        'average_precision': float(avg_precision),
        'std_precision': float(std_precision),
        'average_recall': float(avg_recall),
        'std_recall': float(std_recall),
        'average_f1': float(avg_f1),
        'std_f1': float(std_f1),
        'params_millions': float(params_millions),
        'flops_millions': float(flops_millions),
        'fold_results': all_fold_results
    }
    
    with open(os.path.join(args.output_dir, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到: {args.output_dir}")
    print(f"  - cv_summary.json: 交叉验证汇总结果")
    print(f"  - folds_info.json: 各折的数据划分信息")
    print(f"  - fold_N/: 各折的训练历史和最佳模型")


def train_simple_split(args):
    """简单的train/val划分训练函数（用于快速验证）"""
    # 设置随机种子
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"随机种子已设置为: {args.seed}")
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
        if 'cuda' in args.device and not torch.cuda.is_available():
            raise RuntimeError(f"指定的设备 {args.device} 不可用")
        if 'cuda' in args.device:
            gpu_id = int(args.device.split(':')[1]) if ':' in args.device else 0
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"指定的GPU {gpu_id} 不存在")
    elif args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(device)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据增强：准备三种增强策略用于动态选择
    # 无增强（用于样本数 >= high_threshold）
    no_aug_transform, val_transform = get_data_augmentation(
        augmentation_type='none',
        img_size=args.img_size
    )
    # 标准增强（用于 low_threshold <= 样本数 < high_threshold）
    standard_transform, _ = get_data_augmentation(
        augmentation_type='standard',
        img_size=args.img_size
    )
    # 强增强（用于样本数 < low_threshold）
    strong_transform, _ = get_data_augmentation(
        augmentation_type='strong',
        img_size=args.img_size
    )
    
    # 如果用户指定了特定的增强类型，使用该类型作为标准增强
    if args.augmentation != 'standard':
        standard_transform, _ = get_data_augmentation(
            augmentation_type=args.augmentation,
            img_size=args.img_size
        )
    
    # 加载完整数据集
    full_dataset = ImageFolderDataset(args.data_dir, transform=no_aug_transform)
    num_classes = len(full_dataset.class_to_idx)
    
    print(f"\n数据集信息:")
    print(f"  总样本数: {len(full_dataset)}")
    print(f"  类别数: {num_classes}")
    print(f"  类别: {list(full_dataset.class_to_idx.keys())}")
    
    # 使用分层划分创建train/val集
    labels = [label for _, label in full_dataset.samples]
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=args.val_ratio,
        stratify=labels,
        random_state=args.seed or 42,
        shuffle=True
    )
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    
    print(f"\n数据划分:")
    print(f"  训练集大小: {len(train_indices)} ({len(train_indices)/len(full_dataset)*100:.1f}%)")
    print(f"  验证集大小: {len(val_indices)} ({len(val_indices)/len(full_dataset)*100:.1f}%)")
    
    # 计算训练集中每个类别的样本数
    train_class_counts = compute_class_counts_in_fold(full_dataset, train_indices)
    train_cls_num_list = [train_class_counts.get(idx, 0) for idx in range(num_classes)]
    
    # 创建模型
    model_kwargs = {}
    # StarNet CF 模型需要 cls_num_list
    if 'starnet_cf' in args.model.lower():
        model_kwargs['cls_num_list'] = train_cls_num_list
    
    model = create_model(
        args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        **model_kwargs
    )
    model = model.to(device)
    
    # 创建数据集
    if args.use_dynamic_augmentation:
        train_dataset = DynamicAugmentationSubset(
            base_dataset=full_dataset,
            indices=train_indices,
            class_counts=train_class_counts,
            no_aug_transform=no_aug_transform,
            standard_transform=standard_transform,
            strong_transform=strong_transform,
            low_threshold=args.dynamic_aug_low_threshold,
            high_threshold=args.dynamic_aug_high_threshold
        )
    else:
        train_dataset = TransformSubset(
            full_dataset,
            train_indices,
            transform=standard_transform
        )
    
    val_dataset = TransformSubset(full_dataset, val_indices, transform=val_transform)
    
    # 训练模型
    result = train_fold(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        fold_num=1,  # 简单划分时fold_num设为1
        args=args,
        device=device,
        class_to_idx=full_dataset.class_to_idx,
        class_counts=full_dataset.class_counts
    )
    
    # 保存结果（多分类任务，mAP为主要指标）
    summary = {
        'mode': 'simple_split',
        'val_ratio': args.val_ratio,
        'model': args.model,
        'pretrained': args.pretrained,
        'optimizer': args.optimizer,
        'loss': args.loss,
        'lr': args.lr,
        'augmentation': args.augmentation,
        'use_dynamic_augmentation': args.use_dynamic_augmentation,
        'primary_metric': 'mAP',  # 标记主要指标为mAP
        'best_val_mAP': float(result.get('best_val_mAP', result['mAP'])),
        'best_val_acc': float(result['best_val_acc']),
        'best_epoch': result['best_epoch'],
        'final_val_acc': float(result['final_val_acc']),
        'final_val_loss': float(result['final_val_loss']),
        'mAP': float(result['mAP']),
        'precision_macro': float(result['precision_macro']),
        'recall_macro': float(result['recall_macro']),
        'f1_macro': float(result['f1_macro']),
        'params_millions': float(result['params_millions']),
        'flops_millions': float(result['flops_millions']),
        'metrics': result['metrics'],
        'history': result['history']
    }
    
    with open(os.path.join(args.output_dir, 'simple_split_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到: {args.output_dir}")
    print(f"  - simple_split_summary.json: 训练结果汇总")
    print(f"  - fold_1/: 训练历史和最佳模型")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='训练脚本（支持K折交叉验证或简单train/val划分）')
    
    # 数据相关
    parser.add_argument('--data-dir', type=str, required=True,
                        help='数据目录（包含所有类别子文件夹）')
    parser.add_argument('--img-size', type=int, default=224,
                        help='输入图像大小')
    
    # 训练模式
    parser.add_argument('--simple-split', action='store_true',
                        help='使用简单的train/val划分（快速验证，不使用交叉验证）')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集比例（仅在simple-split模式下使用，默认0.2）')
    
    # 交叉验证相关
    parser.add_argument('--n-splits', type=int, default=5,
                        help='交叉验证折数（默认5，仅在非simple-split模式下使用）')
    
    # 模型相关
    parser.add_argument('--model', type=str, default='resnet50',
                        help='模型名称')
    parser.add_argument('--pretrained', action='store_true',
                        help='是否使用预训练权重')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=100,
                        help='每折的训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD动量')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    
    # 优化器
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'adamw'],
                        help='优化器类型')
    
    # 学习率调度器
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'onecycle', 'cosine_warmup', 'none'],
                        help='学习率调度器')
    parser.add_argument('--step-size', type=int, default=30,
                        help='StepLR的步长')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='StepLR的衰减率')
    
    # Warmup和OneCycleLR参数
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='Warmup比例')
    parser.add_argument('--onecycle-max-lr', type=float, default=None,
                        help='OneCycleLR的最大学习率')
    parser.add_argument('--onecycle-div-factor', type=float, default=25.0,
                        help='OneCycleLR的div_factor')
    parser.add_argument('--onecycle-final-div-factor', type=float, default=10000.0,
                        help='OneCycleLR的final_div_factor')
    parser.add_argument('--onecycle-anneal-strategy', type=str, default='cos',
                        choices=['cos', 'linear'],
                        help='OneCycleLR的退火策略')
    
    # 损失函数
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'focal', 'label_smoothing', 'weighted_ce', 'asl', 'ldam'],
                        help='损失函数类型')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label Smoothing的平滑参数')
    parser.add_argument('--asl-gamma-neg', type=float, default=4.0,
                        help='ASL负类gamma')
    parser.add_argument('--asl-gamma-pos', type=float, default=1.0,
                        help='ASL正类gamma')
    parser.add_argument('--asl-clip', type=float, default=0.05,
                        help='ASL clip')
    parser.add_argument('--asl-eps', type=float, default=1e-8,
                        help='ASL eps')
    parser.add_argument('--ldam-margin', type=float, default=0.5,
                        help='LDAM margin')
    parser.add_argument('--ldam-s', type=float, default=30.0,
                        help='LDAM scale')
    parser.add_argument('--use-drw', action='store_true',
                        help='是否启用 DRW 权重')
    parser.add_argument('--drw-start-epoch', type=int, default=80,
                        help='DRW 起始 epoch')
    parser.add_argument('--drw-beta', type=float, default=0.9999,
                        help='DRW beta')
    parser.add_argument('--drw-window', type=int, default=10,
                        help='DRW 触发窗口长度（轮数）')
    parser.add_argument('--drw-threshold', type=float, default=0.05,
                        help='DRW 触发阈值（验证准确率差值）')
    
    # 数据增强
    parser.add_argument('--augmentation', type=str, default='standard',
                        choices=['none', 'minimal', 'standard', 'strong', 'medical'],
                        help='数据增强类型（仅用于训练集，当禁用动态增强时使用）')
    parser.add_argument('--no-dynamic-augmentation', action='store_false', dest='use_dynamic_augmentation',
                        default=False,
                        help='禁用动态数据增强（默认启用，根据类别样本数量自动选择增强策略）')
    parser.add_argument('--dynamic-aug-low-threshold', type=int, default=50,
                        help='动态增强低阈值：样本数 < 此值使用强增强（默认50）')
    parser.add_argument('--dynamic-aug-high-threshold', type=int, default=300,
                        help='动态增强高阈值：样本数 >= 此值不使用增强（默认300）')
    
    # Early Stopping
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                        help='Early Stopping的patience（默认20，基于val_mAP监控，适用于多分类任务）')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                        help='Early Stopping的最小改善阈值（mAP提升超过此值才算改进）')
    
    # 其他
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--output-dir', type=str, default='checkpoints/cv_results',
                        help='输出目录')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')
    parser.add_argument('--device', type=str, default=None,
                        help='指定设备')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='使用AMP混合精度训练（默认启用，可减少30-40%%显存）')
    parser.add_argument('--no-amp', action='store_false', dest='use_amp',
                        help='禁用AMP混合精度训练')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='梯度累积步数（用于减少显存占用，有效批次大小 = batch_size × gradient_accumulation_steps）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 根据参数选择训练模式
    if args.simple_split:
        print("\n使用简单train/val划分模式（快速验证）")
        train_simple_split(args)
    else:
        print("\n使用K折交叉验证模式")
        train_cross_validation(args)


if __name__ == '__main__':
    main()

