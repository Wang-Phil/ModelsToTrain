"""
增强版CLIP模型训练脚本
支持多种损失函数：分类损失、对比损失、SCCM损失、KDSP损失
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from collections import defaultdict

# 设置环境变量（在导入其他库之前）
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
if 'HF_ENDPOINT' not in os.environ:
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split

# 导入增强版CLIP模型
from models.clip_enhanced import EnhancedCLIPModel, create_enhanced_model

# 导入原始CLIP模型（用于创建teacher模型）
from models.clip import CLIPModel


class CLIPDataset(Dataset):
    """CLIP数据集 - 从按类别组织的文件夹中加载图像"""
    
    def __init__(self, root_dir, transform=None, class_texts_dict=None, class_texts_file=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # 加载类别文本描述
        self.class_texts_map = {}
        if class_texts_file is not None:
            with open(class_texts_file, 'r', encoding='utf-8') as f:
                self.class_texts_map = json.load(f)
        elif class_texts_dict is not None:
            self.class_texts_map = class_texts_dict
        
        # 获取所有类别
        excluded_folders = {'split_fewshot', '__pycache__', '.ipynb_checkpoints'}
        classes = sorted([
            d.name for d in self.root_dir.iterdir() 
            if d.is_dir() and d.name not in excluded_folders and not d.name.startswith('split_')
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # 收集所有图像文件
        for cls_name in classes:
            cls_dir = self.root_dir / cls_name
            for img_file in cls_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_file), self.class_to_idx[cls_name]))
        
        print(f"数据集: {len(self.samples)} 个样本, {len(classes)} 个类别")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, augmentation='standard'):
    """获取数据变换"""
    if augmentation == 'none':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation == 'minimal':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # standard
        train_transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, optimizer, device, epoch, use_amp=True):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_class_loss = 0.0
    running_contrastive_loss = 0.0
    running_sccm_loss = 0.0
    running_kdsp_loss = 0.0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                # 使用增强版CLIP模型的前向传播
                logits, loss_dict = model(images, labels=labels)
                
                loss = loss_dict['total_loss']
                
                # 记录各项损失
                if 'classification_loss' in loss_dict:
                    running_class_loss += loss_dict['classification_loss'].item()
                if 'contrastive_loss' in loss_dict:
                    running_contrastive_loss += loss_dict['contrastive_loss'].item()
                if 'sccm_loss' in loss_dict:
                    running_sccm_loss += loss_dict['sccm_loss'].item()
                if 'kdsp_loss' in loss_dict:
                    running_kdsp_loss += loss_dict['kdsp_loss'].item()
                
                # 计算准确率
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions.cpu() == labels.cpu()).sum().item()
                total += labels.size(0)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 使用增强版CLIP模型的前向传播
            logits, loss_dict = model(images, labels=labels)
            
            loss = loss_dict['total_loss']
            
            # 记录各项损失
            if 'classification_loss' in loss_dict:
                running_class_loss += loss_dict['classification_loss'].item()
            if 'contrastive_loss' in loss_dict:
                running_contrastive_loss += loss_dict['contrastive_loss'].item()
            if 'sccm_loss' in loss_dict:
                running_sccm_loss += loss_dict['sccm_loss'].item()
            if 'kdsp_loss' in loss_dict:
                running_kdsp_loss += loss_dict['kdsp_loss'].item()
            
            # 计算准确率
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions.cpu() == labels.cpu()).sum().item()
            total += labels.size(0)
            
            loss.backward()
            optimizer.step()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    epoch_loss = loss.item() if 'loss' in locals() else 0.0
    epoch_acc = 100.0 * correct / total
    
    metrics = {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'class_loss': running_class_loss / len(dataloader) if running_class_loss > 0 else 0.0,
        'contrastive_loss': running_contrastive_loss / len(dataloader) if running_contrastive_loss > 0 else 0.0,
        'sccm_loss': running_sccm_loss / len(dataloader) if running_sccm_loss > 0 else 0.0,
        'kdsp_loss': running_kdsp_loss / len(dataloader) if running_kdsp_loss > 0 else 0.0,
    }
    
    return metrics


def validate(model, dataloader, device):
    """验证"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='[Val]'):
            images = images.to(device)
            labels = labels.to(device)
            
            # 使用增强版CLIP模型的评估模式
            logits = model(images, labels=None)
            predictions = torch.argmax(logits, dim=1)
            
            correct += (predictions.cpu() == labels.cpu()).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100.0 * correct / total
    return acc, all_predictions, all_labels


def train_enhanced_clip(
    data_dir,
    output_dir,
    image_encoder_name='resnet50',
    text_encoder_name='pubmedbert',
    embed_dim=512,
    temperature=0.07,
    batch_size=32,
    epochs=100,
    learning_rate=1e-4,
    weight_decay=0.01,
    img_size=224,
    augmentation='standard',
    num_workers=4,
    gpu_id=0,
    use_amp=True,
    class_texts_file=None,
    # 损失函数配置
    use_classification_loss=True,
    use_contrastive_loss=True,
    use_sccm_loss=False,
    use_kdsp_loss=False,
    classification_loss_weight=1.0,
    contrastive_loss_weight=1.0,
    sccm_loss_weight=1.0,
    kdsp_loss_weight=1.0,
    # Teacher模型配置（用于KDSP损失）
    teacher_image_encoder=None,
    teacher_text_encoder=None,
    # 其他配置
    freeze_image_encoder=False,
    freeze_text_encoder=False,
    save_best=True,
    resume_from=None
):
    """训练增强版CLIP模型"""
    
    # 设置设备
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    train_transform, val_transform = get_transforms(img_size, augmentation)
    
    full_dataset = CLIPDataset(data_dir, transform=train_transform, class_texts_file=class_texts_file)
    
    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=[full_dataset.samples[i][1] for i in range(len(full_dataset))]
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 获取类别名称列表
    class_names = [full_dataset.idx_to_class[i] for i in range(len(full_dataset.class_to_idx))]
    
    # 创建teacher模型（如果使用KDSP损失）
    teacher_model = None
    if use_kdsp_loss and teacher_image_encoder is not None and teacher_text_encoder is not None:
        print("创建teacher模型用于KDSP损失...")
        teacher_model = EnhancedCLIPModel(
            image_encoder_name=teacher_image_encoder,
            text_encoder_name=teacher_text_encoder,
            embed_dim=embed_dim,
            temperature=temperature,
            class_texts=class_names,
            class_texts_file=class_texts_file,
            use_classification_loss=False,
            use_contrastive_loss=False,
            use_sccm_loss=False,
            use_kdsp_loss=False
        )
        teacher_model.eval()
        teacher_model.to(device)
        print("✓ Teacher模型创建完成")
    
    # 创建增强版CLIP模型
    print("创建增强版CLIP模型...")
    model = EnhancedCLIPModel(
        image_encoder_name=image_encoder_name,
        text_encoder_name=text_encoder_name,
        embed_dim=embed_dim,
        temperature=temperature,
        class_texts=class_names,
        class_texts_file=class_texts_file,
        teacher_model=teacher_model,
        use_classification_loss=use_classification_loss,
        use_contrastive_loss=use_contrastive_loss,
        use_sccm_loss=use_sccm_loss,
        use_kdsp_loss=use_kdsp_loss,
        classification_loss_weight=classification_loss_weight,
        contrastive_loss_weight=contrastive_loss_weight,
        sccm_loss_weight=sccm_loss_weight,
        kdsp_loss_weight=kdsp_loss_weight
    )
    
    # 冻结编码器（如果需要）
    if freeze_image_encoder:
        for param in model.image_encoder.parameters():
            param.requires_grad = False
        print("✓ 图像编码器已冻结")
    
    if freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        print("✓ 文本编码器已冻结")
    
    model.to(device)
    
    # 创建优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 恢复训练（如果需要）
    start_epoch = 0
    best_val_acc = 0.0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"从epoch {start_epoch}恢复训练")
    
    # 训练循环
    print("\n开始训练...")
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'class_loss': [],
        'contrastive_loss': [],
        'sccm_loss': [],
        'kdsp_loss': []
    }
    
    for epoch in range(start_epoch, epochs):
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, use_amp)
        
        # 验证
        val_acc, _, _ = validate(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_acc'].append(val_acc)
        history['class_loss'].append(train_metrics['class_loss'])
        history['contrastive_loss'].append(train_metrics['contrastive_loss'])
        history['sccm_loss'].append(train_metrics['sccm_loss'])
        history['kdsp_loss'].append(train_metrics['kdsp_loss'])
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        if train_metrics['class_loss'] > 0:
            print(f"  Class Loss: {train_metrics['class_loss']:.4f}")
        if train_metrics['contrastive_loss'] > 0:
            print(f"  Contrastive Loss: {train_metrics['contrastive_loss']:.4f}")
        if train_metrics['sccm_loss'] > 0:
            print(f"  SCCM Loss: {train_metrics['sccm_loss']:.4f}")
        if train_metrics['kdsp_loss'] > 0:
            print(f"  KDSP Loss: {train_metrics['kdsp_loss']:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  ✓ 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")
        
        # 保存最新checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history
        }
        torch.save(checkpoint, output_dir / 'latest_checkpoint.pth')
    
    # 保存训练历史
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型保存在: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='增强版CLIP模型训练脚本')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, required=True, help='数据目录（按类别组织的文件夹）')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    
    # 模型参数
    parser.add_argument('--image-encoder', type=str, default='resnet50', help='图像编码器名称')
    parser.add_argument('--text-encoder', type=str, default='pubmedbert', help='文本编码器名称')
    parser.add_argument('--embed-dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--temperature', type=float, default=0.07, help='温度参数')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--img-size', type=int, default=224, help='图像大小')
    parser.add_argument('--augmentation', type=str, default='standard', 
                       choices=['none', 'minimal', 'standard'], help='数据增强类型')
    
    # 损失函数配置
    parser.add_argument('--use-classification-loss', action='store_true', default=True, help='使用分类损失')
    parser.add_argument('--use-contrastive-loss', action='store_true', default=True, help='使用对比损失')
    parser.add_argument('--use-sccm-loss', action='store_true', help='使用SCCM损失')
    parser.add_argument('--use-kdsp-loss', action='store_true', help='使用KDSP损失')
    parser.add_argument('--classification-loss-weight', type=float, default=1.0, help='分类损失权重')
    parser.add_argument('--contrastive-loss-weight', type=float, default=1.0, help='对比损失权重')
    parser.add_argument('--sccm-loss-weight', type=float, default=1.0, help='SCCM损失权重')
    parser.add_argument('--kdsp-loss-weight', type=float, default=1.0, help='KDSP损失权重')
    
    # Teacher模型配置（用于KDSP损失）
    parser.add_argument('--teacher-image-encoder', type=str, default=None, help='Teacher图像编码器（用于KDSP）')
    parser.add_argument('--teacher-text-encoder', type=str, default=None, help='Teacher文本编码器（用于KDSP）')
    
    # 其他参数
    parser.add_argument('--class-texts-file', type=str, default=None, help='类别文本描述JSON文件路径')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度训练')
    parser.add_argument('--freeze-image-encoder', action='store_true', help='冻结图像编码器')
    parser.add_argument('--freeze-text-encoder', action='store_true', help='冻结文本编码器')
    parser.add_argument('--no-save-best', action='store_true', help='不保存最佳模型')
    parser.add_argument('--resume-from', type=str, default=None, help='恢复训练的checkpoint路径')
    
    args = parser.parse_args()
    
    train_enhanced_clip(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        embed_dim=args.embed_dim,
        temperature=args.temperature,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        augmentation=args.augmentation,
        num_workers=args.num_workers,
        gpu_id=args.gpu_id,
        use_amp=not args.no_amp,
        class_texts_file=args.class_texts_file,
        use_classification_loss=args.use_classification_loss,
        use_contrastive_loss=args.use_contrastive_loss,
        use_sccm_loss=args.use_sccm_loss,
        use_kdsp_loss=args.use_kdsp_loss,
        classification_loss_weight=args.classification_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        sccm_loss_weight=args.sccm_loss_weight,
        kdsp_loss_weight=args.kdsp_loss_weight,
        teacher_image_encoder=args.teacher_image_encoder,
        teacher_text_encoder=args.teacher_text_encoder,
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_text_encoder=args.freeze_text_encoder,
        save_best=not args.no_save_best,
        resume_from=args.resume_from
    )

