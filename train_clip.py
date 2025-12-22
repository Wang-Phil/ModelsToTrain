"""
CLIP模型训练脚本
支持多个图像编码器和文本编码器的组合训练
使用对比学习损失函数
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from collections import defaultdict
from itertools import product

# 设置环境变量（在导入其他库之前）
# 避免 tokenizers 在多进程环境中的警告
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

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

# 导入CLIP模型
from models.clip import CLIPModel, ImageEncoder, TextEncoder

# 导入 MetaCLIP 适配器（可选）
try:
    from models.metaclip_adapter import MetaCLIPAdapter, create_metaclip_model
    METACLIP_AVAILABLE = True
except ImportError:
    METACLIP_AVAILABLE = False
    print("Warning: MetaCLIP adapter not available. Install MetaCLIP to use MetaCLIP pretrained models.")
if 'HF_ENDPOINT' not in os.environ:
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


class CLIPDataset(Dataset):
    """
    CLIP数据集
    从按类别组织的文件夹中加载图像和对应的文本描述
    """
    
    def __init__(self, root_dir, transform=None, text_template=None, class_texts_dict=None, class_texts_file=None):
        """
        Args:
            root_dir: 数据根目录，包含按类别组织的子文件夹
            transform: 图像变换
            text_template: 文本模板，例如 "这是一张{class_name}的图片"
            class_texts_dict: 类别文本描述字典，例如 {"类别1": "这是一张类别1的图片", "类别2": "这是一张类别2的图片"}
            class_texts_file: 类别文本描述JSON文件路径，格式: {"类别1": "描述1", "类别2": "描述2"}
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.class_texts_map = {}  # 类别名称 -> 文本描述的映射
        
        # 加载类别文本描述
        if class_texts_file is not None:
            # 从文件加载
            with open(class_texts_file, 'r', encoding='utf-8') as f:
                self.class_texts_map = json.load(f)
            print(f"从文件加载类别文本描述: {class_texts_file}")
        elif class_texts_dict is not None:
            # 从字典加载
            self.class_texts_map = class_texts_dict
            print(f"使用提供的类别文本描述字典")
        
        # 默认文本模板
        if text_template is None:
            self.text_template = "{class_name}"
        else:
            self.text_template = text_template
        
        # 获取所有类别
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # 收集所有图像文件
        for class_name in classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            # 生成类别文本描述（优先级：class_texts_map > text_template）
            if class_name in self.class_texts_map:
                class_text = self.class_texts_map[class_name]
            else:
                class_text = self.text_template.format(class_name=class_name)
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), class_idx, class_text))
        
        print(f"数据集: {len(self.samples)} 个样本, {len(classes)} 个类别")
        print(f"类别: {classes}")
        
        # 打印类别文本描述
        if self.class_texts_map:
            print("\n类别文本描述:")
            for class_name in classes:
                if class_name in self.class_texts_map:
                    print(f"  {class_name}: {self.class_texts_map[class_name]}")
                else:
                    print(f"  {class_name}: {self.text_template.format(class_name=class_name)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, text = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, text
    
    def get_class_texts(self):
        """获取所有类别的文本描述"""
        class_texts = []
        for idx in sorted(self.idx_to_class.keys()):
            class_name = self.idx_to_class[idx]
            # 使用映射的文本描述，如果没有则使用模板
            if class_name in self.class_texts_map:
                class_text = self.class_texts_map[class_name]
            else:
                class_text = self.text_template.format(class_name=class_name)
            class_texts.append(class_text)
        return class_texts


class CLIPSubset(Dataset):
    """支持不同transform的CLIP数据集Subset"""
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取原始数据
        img_path, label, text = self.base_dataset.samples[self.indices[idx]]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用指定的transform（覆盖原始transform）
        if self.transform:
            image = self.transform(image)
        elif self.base_dataset.transform:
            image = self.base_dataset.transform(image)
        
        return image, label, text
    
    def get_class_texts(self):
        """获取所有类别的文本描述"""
        return self.base_dataset.get_class_texts()


def create_folds_from_dataset(dataset, n_splits=5, shuffle=True, random_state=42):
    """
    从数据集创建K折交叉验证的折
    
    Args:
        dataset: CLIPDataset实例
        n_splits: 折数
        shuffle: 是否打乱
        random_state: 随机种子
    
    Returns:
        folds: 包含(train_indices, val_indices)的列表
    """
    # 获取所有样本的标签
    labels = [label for _, label, _ in dataset.samples]
    
    # 使用分层K折（StratifiedKFold）确保每折中类别分布相同
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in skf.split(range(len(dataset)), labels):
        folds.append((train_idx, val_idx))
    
    return folds


def create_weighted_sampler(dataset, subset_indices=None, weight_method='inverse_freq', smooth_factor=1.0):
    """
    创建加权采样器，用于处理类别不平衡问题
    
    Args:
        dataset: CLIPDataset实例
        subset_indices: 子集索引（如果为None，则使用整个数据集）
        weight_method: 权重计算方法
            - 'inverse_freq': 逆频率权重，weight = total_samples / (num_classes * class_count)
            - 'inverse_sqrt': 逆平方根频率，weight = sqrt(total_samples / class_count)
            - 'balanced': 平衡权重，weight = total_samples / (num_classes * class_count)，与inverse_freq相同
        smooth_factor: 平滑因子，用于避免权重过大（默认1.0）
    
    Returns:
        sampler: WeightedRandomSampler实例
        class_weights: 每个类别的权重字典（用于打印信息）
    """
    # 获取样本标签
    if subset_indices is not None:
        labels = [dataset.samples[i][1] for i in subset_indices]
    else:
        labels = [label for _, label, _ in dataset.samples]
    
    # 统计每个类别的样本数
    from collections import Counter
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # 计算每个类别的权重
    class_weights = {}
    for class_idx, count in class_counts.items():
        if weight_method == 'inverse_freq' or weight_method == 'balanced':
            # 逆频率权重
            weight = total_samples / (num_classes * (count + smooth_factor))
        elif weight_method == 'inverse_sqrt':
            # 逆平方根频率权重（更平滑）
            weight = np.sqrt(total_samples / (count + smooth_factor))
        else:
            raise ValueError(f"Unknown weight_method: {weight_method}")
        class_weights[class_idx] = weight
    
    # 为每个样本分配权重
    sample_weights = []
    for label in labels:
        sample_weights.append(class_weights[label])
    
    # 创建加权采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # 允许重复采样，确保少数类样本有更多机会被选中
    )
    
    return sampler, class_weights


class CLIPLoss(nn.Module):
    """CLIP对比学习损失函数"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: [batch_size, embed_dim]
            text_features: [batch_size, embed_dim] 或 [num_classes, embed_dim]
        Returns:
            loss: 标量损失值
        """
        # 确保特征形状正确
        # 如果 image_features 的形状不对，尝试修复
        if len(image_features.shape) > 2:
            # 如果是 [batch_size, H, W, C] 或其他形状，需要 flatten
            image_features = image_features.view(image_features.shape[0], -1)
        
        # 检查维度是否匹配
        if len(image_features.shape) != 2 or len(text_features.shape) != 2:
            raise ValueError(f"特征形状错误: image_features={image_features.shape}, text_features={text_features.shape}")
        
        # 如果 embed_dim 不匹配，可能需要调整
        if image_features.shape[1] != text_features.shape[1]:
            # 如果 image_features 的维度不对，可能是某个地方搞混了
            # 尝试重塑或投影
            if image_features.shape[0] == text_features.shape[1] and image_features.shape[1] == text_features.shape[0]:
                # 可能是转置了
                image_features = image_features.T
            else:
                raise ValueError(f"特征维度不匹配: image_features={image_features.shape}, text_features={text_features.shape}")
        
        # 归一化特征
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # 计算相似度矩阵
        # 如果text_features是[batch_size, embed_dim]，则是image-text配对
        # 如果text_features是[num_classes, embed_dim]，则是image-class配对
        logits = image_features @ text_features.T / self.temperature
        
        batch_size = image_features.shape[0]
        
        # 如果是image-text配对（batch内对比学习）
        if text_features.shape[0] == batch_size:
            labels = torch.arange(batch_size, device=image_features.device)
            # 双向对比损失
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2
        else:
            # 如果是image-class配对（分类任务）
            # 假设每个图像对应一个类别（通过标签）
            # 这里需要传入labels，或者使用零样本分类
            # 对于零样本分类，我们不在这里计算损失
            # 而是使用标准的分类损失
            raise NotImplementedError("Image-class pairing needs labels for loss calculation")
        
        return loss


class SuperCLIPLoss(nn.Module):
    """
    SuperCLIP 损失函数（基于 SuperCLIP 官方实现）
    结合分类损失和对比损失
    参考: superclip/open_clip/loss.py SuperClipLoss
    """
    
    def __init__(self, temperature=0.07, class_loss_weight=1.0, contrastive_loss_weight=1.0,
                 use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0, world_size=1):
        """
        Args:
            temperature: 温度参数（用于对比损失，实际使用 logit_scale）
            class_loss_weight: 分类损失权重
            contrastive_loss_weight: 对比损失权重
            use_focal_loss: 是否使用 focal loss 作为分类损失（默认False，使用SuperCLIP的KL散度形式）
            focal_alpha: Focal loss 的 alpha 参数（默认0.25）
            focal_gamma: Focal loss 的 gamma 参数（默认2.0）
            world_size: 分布式训练时的world_size（默认1，单GPU）
        """
        super().__init__()
        self.temperature = temperature
        self.class_loss_weight = class_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.world_size = world_size
        
        # 用于跟踪类别频率（用于重加权）- 如果未从外部传入则使用内部buffer
        self.register_buffer('_internal_cap_fq', None)
        self.register_buffer('_internal_num_samples', None)
    
    def loss(self, logits, targets):
        """
        SuperCLIP 的分类损失实现（KL散度形式）
        参考: superclip/open_clip/loss.py SuperClipLoss.loss()
        
        Args:
            logits: [batch_size, num_classes] 分类logits
            targets: [batch_size, num_classes] 归一化的目标分布（one-hot或soft）
        Returns:
            loss: 标量损失值
        """
        # SuperCLIP 的实现方式：L1归一化 + KL散度
        norm_item = F.normalize(targets, p=1, dim=1)
        loss = -(F.log_softmax(logits, dim=1) * norm_item).sum(dim=1).mean()
        return loss
    
    def reweight_targets(self, cap_fq, num_samples, targets):
        """
        SuperCLIP 的频率重加权实现
        参考: superclip/open_clip/loss.py SuperClipLoss.reweight_targets()
        
        Args:
            cap_fq: [1, num_classes] 或 [num_classes] 类别频率统计（会被原地修改）
            num_samples: [1, 1] 或标量 样本计数（会被原地修改）
            targets: [batch_size, num_classes] one-hot 目标
        Returns:
            reweighted_targets: 重加权后的目标
        """
        # 确保 cap_fq 和 num_samples 是 tensor
        if not isinstance(cap_fq, torch.Tensor):
            cap_fq = torch.tensor(cap_fq, device=targets.device, dtype=torch.float64)
        if not isinstance(num_samples, torch.Tensor):
            num_samples = torch.tensor(num_samples, device=targets.device, dtype=torch.float64)
        
        # 确保维度正确
        if cap_fq.dim() == 1:
            cap_fq = cap_fq.unsqueeze(0)
        if num_samples.dim() == 0:
            num_samples = num_samples.unsqueeze(0).unsqueeze(0)
        elif num_samples.dim() == 1:
            num_samples = num_samples.unsqueeze(0)
        
        # SuperCLIP 的累积更新方式
        cap_fq += targets.sum(dim=0, keepdim=True) / targets.shape[0]
        num_samples += 1
        
        # 计算重加权因子（SuperCLIP 公式）
        all_batch_size = self.world_size * targets.shape[0]
        reweight_factor = torch.log(
            (num_samples + 1.0 / all_batch_size) / (cap_fq + 1.0 / all_batch_size)
        ).to(dtype=targets.dtype)
        
        # 如果 cap_fq 是 [1, num_classes]，需要 squeeze
        if reweight_factor.dim() == 2 and reweight_factor.shape[0] == 1:
            reweight_factor = reweight_factor.squeeze(0)
        
        # 应用重加权
        reweighted_targets = targets * reweight_factor.unsqueeze(0)
        
        return reweighted_targets
    
    def classification_loss(self, logits, targets):
        """
        分类损失（SuperCLIP KL散度形式 或 Focal Loss）
        Args:
            logits: [batch_size, num_classes] 分类logits
            targets: [batch_size, num_classes] 归一化的目标分布（one-hot或soft）
        Returns:
            loss: 标量损失值
        """
        if self.use_focal_loss:
            # 使用 Focal Loss（可选功能）
            true_labels = targets.argmax(dim=1)
            probs = F.softmax(logits, dim=1)
            p_t = probs.gather(1, true_labels.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.focal_gamma
            loss = -self.focal_alpha * focal_weight * torch.log(p_t + 1e-8)
            loss = loss.mean()
            
            if torch.isnan(loss) or torch.isinf(loss):
                loss = F.cross_entropy(logits, true_labels)
        else:
            # 使用 SuperCLIP 的 KL散度形式（默认）
            loss = self.loss(logits, targets)
            
            # 确保损失不为NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss):
                loss = F.cross_entropy(logits, targets.argmax(dim=1))
        
        return loss
    
    def forward(self, image_features, text_features, class_logits, labels, 
                class_text_features=None, class_counts=None, total_samples=None, 
                num_classes=None, smooth_factor=1.0, output_dict=False,
                cap_fq=None, num_samples=None, logit_scale=None):
        """
        SuperCLIP 损失函数前向传播
        支持两种调用方式：
        1. 旧接口：传入 class_counts, total_samples（向后兼容）
        2. 新接口：传入 cap_fq, num_samples（SuperCLIP 官方方式）
        
        Args:
            image_features: [batch_size, embed_dim] 图像特征
            text_features: [batch_size, embed_dim] 文本特征（用于对比损失）
            class_logits: [batch_size, num_classes] 分类logits
            labels: [batch_size] 真实类别标签
            class_text_features: [num_classes, embed_dim] 所有类别的文本特征（可选）
            class_counts: 每个类别的样本数（旧接口，用于重加权）
            total_samples: 总样本数（旧接口，用于重加权）
            num_classes: 类别数
            smooth_factor: 平滑因子（旧接口，未使用）
            output_dict: 是否返回字典格式
            cap_fq: [1, num_classes] 类别频率统计（新接口，SuperCLIP方式）
            num_samples: [1, 1] 样本计数（新接口，SuperCLIP方式）
            logit_scale: 可学习的logit scale（用于对比损失，如果为None则使用temperature）
        Returns:
            loss 或 {"class_loss": ..., "contrastive_loss": ..., "total_loss": ...}
        """
        device = image_features.device
        batch_size = image_features.shape[0]
        
        # 确定类别数
        if num_classes is None:
            num_classes = class_logits.shape[1]
        
        # 1. 分类损失
        # 创建one-hot目标
        targets = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=device)
        # labels 可能是 [B] 或 [B, seq_len]，取第一个token作为类别ID
        if labels.dim() > 1:
            labels_for_scatter = labels[:, 0].unsqueeze(1)
        else:
            labels_for_scatter = labels.unsqueeze(1)
        targets.scatter_(dim=1, index=labels_for_scatter, value=1.0)
        
        # 重加权目标（SuperCLIP 方式）
        if cap_fq is not None and num_samples is not None:
            # 新接口：使用 SuperCLIP 官方方式
            targets = self.reweight_targets(cap_fq, num_samples, targets)
        elif class_counts is not None and total_samples is not None:
            # 旧接口：兼容原有调用方式
            # 初始化内部 buffer（如果未初始化）
            if self._internal_cap_fq is None:
                self._internal_cap_fq = torch.zeros(num_classes, device=device, dtype=torch.float64)
            if self._internal_num_samples is None:
                self._internal_num_samples = torch.zeros(1, device=device, dtype=torch.float64)
            
            # 转换为 SuperCLIP 格式
            cap_fq = self._internal_cap_fq.unsqueeze(0)  # [1, num_classes]
            num_samples = self._internal_num_samples.unsqueeze(0)  # [1, 1]
            targets = self.reweight_targets(cap_fq, num_samples, targets)
        
        class_loss = self.classification_loss(class_logits, targets)
        
        # 2. 对比损失（CLIP loss）- SuperCLIP 实现方式
        # 归一化特征
        image_features_norm = F.normalize(image_features, dim=1)
        text_features_norm = F.normalize(text_features, dim=1)
        
        # 使用 logit_scale 或 temperature
        if logit_scale is not None:
            # 如果 logit_scale 是 tensor，可能需要 exp()
            if isinstance(logit_scale, torch.Tensor):
                if logit_scale.numel() == 1:
                    scale = logit_scale.exp() if logit_scale.item() < 10 else logit_scale
                else:
                    scale = logit_scale.mean()
            else:
                scale = logit_scale
        else:
            scale = 1.0 / self.temperature if self.temperature > 0 else 1.0
        
        # 计算相似度矩阵
        logits_per_image = scale * image_features_norm @ text_features_norm.T
        logits_per_text = scale * text_features_norm @ image_features_norm.T
        
        # 创建对比学习的标签（对角线匹配）
        contrastive_labels = torch.arange(batch_size, device=device)
        
        # 双向对比损失（SuperCLIP 方式）
        contrastive_loss = (
            F.cross_entropy(logits_per_image, contrastive_labels) +
            F.cross_entropy(logits_per_text, contrastive_labels)
        ) / 2
        
        # 3. 组合损失
        total_loss = self.class_loss_weight * class_loss + self.contrastive_loss_weight * contrastive_loss
        
        if output_dict:
            return {
                "class_loss": class_loss,
                "contrastive_loss": contrastive_loss,
                "total_loss": total_loss
            }
        
        return total_loss


def get_data_augmentation(augmentation_type='standard', img_size=224):
    """获取数据增强策略"""
    if augmentation_type == 'none':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'minimal':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_type == 'standard':
        train_transform = transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, use_amp=True, 
                class_texts=None, use_superclip_loss=False, class_counts=None, 
                total_samples=None, num_classes=None, smooth_factor=1.0):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_class_loss = 0.0
    running_contrastive_loss = 0.0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels, texts) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                # 获取图像特征和文本特征
                image_features, text_features = model(images, texts=texts)
                
                if use_superclip_loss and class_texts is not None and isinstance(criterion, SuperCLIPLoss):
                    # 使用 SuperCLIP 损失（需要类别文本特征）
                    # 编码所有类别的文本
                    class_text_features = model.text_encoder(texts=class_texts)
                    
                    # 计算分类 logits（图像特征与所有类别文本的相似度）
                    image_features_norm = F.normalize(image_features, dim=1)
                    class_text_features_norm = F.normalize(class_text_features, dim=1)
                    class_logits = (image_features_norm @ class_text_features_norm.T) / model.temperature
                    
                    # 计算 SuperCLIP 损失
                    loss_dict = criterion(
                        image_features=image_features,
                        text_features=text_features,
                        class_logits=class_logits,
                        labels=labels,
                        class_text_features=class_text_features,
                        class_counts=class_counts,
                        total_samples=total_samples,
                        num_classes=num_classes,
                        smooth_factor=smooth_factor,
                        output_dict=True
                    )
                    loss = loss_dict['total_loss']
                    class_loss = loss_dict['class_loss']
                    contrastive_loss = loss_dict['contrastive_loss']
                    
                    running_class_loss += class_loss.item()
                    running_contrastive_loss += contrastive_loss.item()
                else:
                    # 使用标准 CLIP 损失
                    loss = criterion(image_features, text_features)
                    class_loss = torch.tensor(0.0)
                    contrastive_loss = loss
                
                # 计算准确率（使用类别文本）
                if class_texts is not None:
                    with torch.no_grad():
                        predictions, _ = model.predict(images, class_texts)
                        correct += (predictions.cpu() == labels.cpu()).sum().item()
                        total += labels.size(0)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 获取图像特征和文本特征
            image_features, text_features = model(images, texts=texts)
            
            if use_superclip_loss and class_texts is not None and isinstance(criterion, SuperCLIPLoss):
                # 使用 SuperCLIP 损失
                class_text_features = model.text_encoder(texts=class_texts)
                image_features_norm = F.normalize(image_features, dim=1)
                class_text_features_norm = F.normalize(class_text_features, dim=1)
                class_logits = (image_features_norm @ class_text_features_norm.T) / model.temperature
                
                loss_dict = criterion(
                    image_features=image_features,
                    text_features=text_features,
                    class_logits=class_logits,
                    labels=labels,
                    class_text_features=class_text_features,
                    class_counts=class_counts,
                    total_samples=total_samples,
                    num_classes=num_classes,
                    smooth_factor=smooth_factor,
                    output_dict=True
                )
                loss = loss_dict['total_loss']
                class_loss = loss_dict['class_loss']
                contrastive_loss = loss_dict['contrastive_loss']
                
                running_class_loss += class_loss.item()
                running_contrastive_loss += contrastive_loss.item()
            else:
                loss = criterion(image_features, text_features)
                class_loss = torch.tensor(0.0)
                contrastive_loss = loss
            
            if class_texts is not None:
                with torch.no_grad():
                    predictions, _ = model.predict(images, class_texts)
                    correct += (predictions.cpu() == labels.cpu()).sum().item()
                    total += labels.size(0)
            
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # 更新进度条
        acc_str = f'{100.0 * correct / total:.2f}%' if total > 0 else 'N/A'
        if use_superclip_loss:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{class_loss.item():.4f}' if isinstance(class_loss, torch.Tensor) else f'{class_loss:.4f}',
                'clip': f'{contrastive_loss.item():.4f}' if isinstance(contrastive_loss, torch.Tensor) else f'{contrastive_loss:.4f}',
                'acc': acc_str
            })
        else:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': acc_str
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    
    result = {'loss': epoch_loss, 'acc': epoch_acc}
    if use_superclip_loss:
        result['class_loss'] = running_class_loss / len(dataloader)
        result['contrastive_loss'] = running_contrastive_loss / len(dataloader)
    
    return result


def validate(model, dataloader, criterion, device, use_amp=True, class_texts=None, 
             num_classes=None, use_superclip_loss=False, class_counts=None, 
             total_samples=None, smooth_factor=1.0):
    """验证"""
    model.eval()
    running_loss = 0.0
    running_class_loss = 0.0
    running_contrastive_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Val]')
        for images, labels, texts in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    image_features, text_features = model(images, texts=texts)
                    
                    if use_superclip_loss and class_texts is not None and isinstance(criterion, SuperCLIPLoss):
                        # 使用 SuperCLIP 损失
                        class_text_features = model.text_encoder(texts=class_texts)
                        image_features_norm = F.normalize(image_features, dim=1)
                        class_text_features_norm = F.normalize(class_text_features, dim=1)
                        class_logits = (image_features_norm @ class_text_features_norm.T) / model.temperature
                        
                        loss_dict = criterion(
                            image_features=image_features,
                            text_features=text_features,
                            class_logits=class_logits,
                            labels=labels,
                            class_text_features=class_text_features,
                            class_counts=class_counts,
                            total_samples=total_samples,
                            num_classes=num_classes,
                            smooth_factor=smooth_factor,
                            output_dict=True
                        )
                        loss = loss_dict['total_loss']
                        class_loss = loss_dict['class_loss']
                        contrastive_loss = loss_dict['contrastive_loss']
                        
                        running_class_loss += class_loss.item()
                        running_contrastive_loss += contrastive_loss.item()
                    else:
                        loss = criterion(image_features, text_features)
                        class_loss = torch.tensor(0.0)
                        contrastive_loss = loss
                    
                    if class_texts is not None:
                        predictions, probabilities = model.predict(images, class_texts)
                    else:
                        predictions = torch.zeros(labels.size(0), dtype=torch.long, device=device)
            else:
                image_features, text_features = model(images, texts=texts)
                
                if use_superclip_loss and class_texts is not None and isinstance(criterion, SuperCLIPLoss):
                    # 使用 SuperCLIP 损失
                    class_text_features = model.text_encoder(texts=class_texts)
                    image_features_norm = F.normalize(image_features, dim=1)
                    class_text_features_norm = F.normalize(class_text_features, dim=1)
                    class_logits = (image_features_norm @ class_text_features_norm.T) / model.temperature
                    
                    loss_dict = criterion(
                        image_features=image_features,
                        text_features=text_features,
                        class_logits=class_logits,
                        labels=labels,
                        class_text_features=class_text_features,
                        class_counts=class_counts,
                        total_samples=total_samples,
                        num_classes=num_classes,
                        smooth_factor=smooth_factor,
                        output_dict=True
                    )
                    loss = loss_dict['total_loss']
                    class_loss = loss_dict['class_loss']
                    contrastive_loss = loss_dict['contrastive_loss']
                    
                    running_class_loss += class_loss.item()
                    running_contrastive_loss += contrastive_loss.item()
                else:
                    loss = criterion(image_features, text_features)
                    class_loss = torch.tensor(0.0)
                    contrastive_loss = loss
                
                if class_texts is not None:
                    predictions, probabilities = model.predict(images, class_texts)
                else:
                    predictions = torch.zeros(labels.size(0), dtype=torch.long, device=device)
            
            running_loss += loss.item()
            if class_texts is not None:
                correct += (predictions.cpu() == labels.cpu()).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            acc_str = f'{100.0 * correct / total:.2f}%' if total > 0 else 'N/A'
            if use_superclip_loss:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cls': f'{class_loss.item():.4f}' if isinstance(class_loss, torch.Tensor) else f'{class_loss:.4f}',
                    'clip': f'{contrastive_loss.item():.4f}' if isinstance(contrastive_loss, torch.Tensor) else f'{contrastive_loss:.4f}',
                    'acc': acc_str
                })
            else:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': acc_str
                })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    
    # 计算mAP
    val_mAP = 0.0
    if num_classes is not None and len(all_predictions) > 0 and len(all_labels) > 0:
        try:
            from calculate_metrics import calculate_classification_metrics
            metrics = calculate_classification_metrics(all_labels, all_predictions, num_classes)
            val_mAP = metrics['mAP']
        except Exception as e:
            print(f"警告: 计算mAP失败: {e}")
            val_mAP = 0.0
    
    result = (epoch_loss, epoch_acc, all_predictions, all_labels, val_mAP)
    if use_superclip_loss:
        result = result + (running_class_loss / len(dataloader), running_contrastive_loss / len(dataloader))
    
    return result


# train_clip_model 函数已删除，只保留交叉验证版本


def train_clip_cross_validation(
    data_dir,
    output_dir,
    image_encoder_name='resnet50',
    text_encoder_name='bert-base-chinese',
    use_metaclip=False,
    metaclip_model_name='ViT-B-32-quickgelu',
    metaclip_pretrained='metaclip_400m',
    embed_dim=512,
    batch_size=32,
    epochs=100,
    learning_rate=1e-4,
    weight_decay=0.01,
    temperature=0.07,
    img_size=224,
    augmentation='standard',
    num_workers=4,
    use_amp=True,
    gpu_id=0,
    n_splits=5,
    random_state=42,
    save_best=True,
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
    early_stopping_monitor='val_loss',
    text_template=None,
    class_texts_dict=None,
    class_texts_file=None,
    use_weighted_sampling=False,
    weight_method='inverse_freq',
    weight_smooth_factor=1.0,
    use_superclip_loss=False,
    class_loss_weight=1.0,
    contrastive_loss_weight=1.0,
    use_focal_loss=False,
    focal_alpha=0.25,
    focal_gamma=2.0,
):
    """
    使用K折交叉验证训练CLIP模型
    
    Args:
        data_dir: 数据目录（按类别组织的文件夹）
        output_dir: 输出目录
        n_splits: 折数（默认5折）
        random_state: 随机种子
        其他参数同 train_clip_model
    """
    from pathlib import Path
    import json
    import random
    
    # 设置全局随机种子（确保可复现性）
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        # 设置确定性算法（可能略微影响性能，但确保可复现）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已设置为: {random_state}")
    print(f"确定性模式: CUDNN deterministic={torch.backends.cudnn.deterministic}, benchmark={torch.backends.cudnn.benchmark}")
    
    # 设置设备
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config = {
        'image_encoder': image_encoder_name,
        'text_encoder': text_encoder_name,
        'embed_dim': embed_dim,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'temperature': temperature,
        'img_size': img_size,
        'augmentation': augmentation,
        'n_splits': n_splits,
        'random_state': random_state,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 数据增强
    train_transform, val_transform = get_data_augmentation(augmentation, img_size)
    
    # 创建完整数据集（无增强，用于创建folds）
    if text_template is None:
        text_template = "{class_name}"
    
    full_dataset = CLIPDataset(
        data_dir, 
        transform=None, 
        text_template=text_template,
        class_texts_dict=class_texts_dict,
        class_texts_file=class_texts_file
    )
    class_texts = full_dataset.get_class_texts()
    num_classes = len(full_dataset.class_to_idx)
    
    print(f"\n创建 {n_splits} 折交叉验证...")
    folds = create_folds_from_dataset(full_dataset, n_splits=n_splits, shuffle=True, random_state=random_state)
    print(f"✓ 成功创建 {len(folds)} 个fold")
    
    # 存储所有fold的结果
    all_fold_results = {
        'fold_train_loss': [],
        'fold_train_acc': [],
        'fold_val_loss': [],
        'fold_val_acc': [],
        'fold_val_mAP': [],
        'fold_best_val_acc': [],
        'fold_best_val_mAP': [],
        'fold_best_epoch': [],
        'fold_best_precision': [],
        'fold_best_recall': [],
        'fold_best_f1': [],
    }
    
    # 训练每个fold
    for fold_num, (train_indices, val_indices) in enumerate(folds, 1):
        print(f"\n{'='*80}")
        print(f"训练 Fold {fold_num}/{n_splits}")
        print(f"{'='*80}")
        
        fold_dir = output_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建fold的数据集
        train_subset = CLIPSubset(full_dataset, train_indices, transform=train_transform)
        val_subset = CLIPSubset(full_dataset, val_indices, transform=val_transform)
        
        # 创建加权采样器（如果启用）
        train_sampler = None
        if use_weighted_sampling:
            if fold_num == 1:  # 只在第一个fold打印详细信息
                print(f"\nFold {fold_num} 启用加权采样以处理类别不平衡...")
                print(f"权重计算方法: {weight_method}")
                print(f"平滑因子: {weight_smooth_factor}")
            
            # 注意：CLIPSubset的索引需要映射回原始数据集
            # 我们需要为train_subset创建采样器，但权重需要基于原始索引
            # 创建一个辅助函数来获取subset中的标签
            def get_subset_labels(subset, base_dataset, indices):
                """获取子集的标签列表"""
                labels = []
                for idx in indices:
                    labels.append(base_dataset.samples[idx][1])
                return labels
            
            # 计算权重（基于训练集的类别分布）
            subset_labels = get_subset_labels(train_subset, full_dataset, train_indices)
            from collections import Counter
            class_counts = Counter(subset_labels)
            total_samples = len(subset_labels)
            num_classes = len(class_counts)
            
            # 计算每个类别的权重
            class_weights = {}
            for class_idx, count in class_counts.items():
                if weight_method == 'inverse_freq' or weight_method == 'balanced':
                    weight = total_samples / (num_classes * (count + weight_smooth_factor))
                elif weight_method == 'inverse_sqrt':
                    weight = np.sqrt(total_samples / (count + weight_smooth_factor))
                else:
                    raise ValueError(f"Unknown weight_method: {weight_method}")
                class_weights[class_idx] = weight
            
            # 为每个样本分配权重
            sample_weights = [class_weights[label] for label in subset_labels]
            
            # 创建加权采样器
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            if fold_num == 1:  # 只在第一个fold打印详细信息
                print("\n各类别权重:")
                for class_idx, weight in sorted(class_weights.items()):
                    class_name = full_dataset.idx_to_class[class_idx]
                    class_count = class_counts[class_idx]
                    print(f"  {class_name:25s}: 权重={weight:.4f}, 样本数={class_count}")
        
        # 创建DataLoader
        if train_sampler is not None:
            train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler,
                                      num_workers=num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        
        # 创建模型
        if fold_num == 1:  # 只在第一个fold打印详细信息
            print(f"\n创建CLIP模型:")
            print(f"  图像编码器: {image_encoder_name}")
            print(f"  文本编码器: {text_encoder_name}")
            print(f"  嵌入维度: {embed_dim}")
        
        # 检查是否使用 MetaCLIP
        if use_metaclip:
            if not METACLIP_AVAILABLE:
                raise ImportError("MetaCLIP is not available. Please install MetaCLIP or set METACLIP_PATH correctly.")
            
            print(f"  使用 MetaCLIP 预训练模型: {metaclip_model_name}, pretrained: {metaclip_pretrained}")
            model = create_metaclip_model(
                model_name=metaclip_model_name,
                pretrained=metaclip_pretrained,
                embed_dim=embed_dim,
                temperature=temperature,
                device=device
            ).to(device)
        elif image_encoder_name.startswith('metaclip:'):
            # 通过 image_encoder_name 指定 MetaCLIP
            if not METACLIP_AVAILABLE:
                raise ImportError("MetaCLIP is not available. Please install MetaCLIP or set METACLIP_PATH correctly.")
            
            # 解析 MetaCLIP 模型名称和预训练权重
            # 格式: metaclip:ViT-B-32-quickgelu:metaclip_400m
            parts = image_encoder_name.split(':')
            if len(parts) >= 2:
                metaclip_model_name = parts[1]
                metaclip_pretrained = parts[2] if len(parts) > 2 else 'metaclip_400m'
            else:
                raise ValueError(f"Invalid MetaCLIP model name format: {image_encoder_name}. "
                               f"Expected format: metaclip:ViT-B-32-quickgelu:metaclip_400m")
            
            print(f"  使用 MetaCLIP 预训练模型: {metaclip_model_name}, pretrained: {metaclip_pretrained}")
            model = create_metaclip_model(
                model_name=metaclip_model_name,
                pretrained=metaclip_pretrained,
                embed_dim=embed_dim,
                temperature=temperature,
                device=device
            ).to(device)
        else:
            model = CLIPModel(
                image_encoder_name=image_encoder_name,
                text_encoder_name=text_encoder_name,
                embed_dim=embed_dim,
                temperature=temperature
            ).to(device)
        
        # 损失函数
        if use_superclip_loss:
            criterion = SuperCLIPLoss(
                temperature=temperature,
                class_loss_weight=class_loss_weight,
                contrastive_loss_weight=contrastive_loss_weight,
                use_focal_loss=use_focal_loss,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma
            )
        else:
            criterion = CLIPLoss(temperature=temperature)
        
        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_mAP': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_val_mAP = 0.0
        best_epoch = 0
        best_metrics = None  # 存储最佳epoch的详细指标
        
        # 早停相关变量
        early_stopping_counter = 0
        use_early_stopping = early_stopping_patience is not None and early_stopping_patience > 0
        
        if use_early_stopping:
            print(f"Fold {fold_num} 启用早停策略 (耐心值: {early_stopping_patience})")
        
        # 计算类别统计信息（用于 SuperCLIP 损失的重加权）
        class_counts = None
        total_samples = None
        if use_superclip_loss:
            from collections import Counter
            subset_labels = [full_dataset.samples[i][1] for i in train_indices]
            class_counts = Counter(subset_labels)
            total_samples = len(subset_labels)
            print(f"\n类别统计信息（用于重加权）:")
            for class_idx, count in sorted(class_counts.items()):
                class_name = full_dataset.idx_to_class[class_idx]
                print(f"  {class_name}: {count} 个样本")
        
        # 训练循环
        for epoch in range(epochs):
            print(f"\nFold {fold_num}, Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_result = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch+1, use_amp, 
                class_texts, use_superclip_loss, class_counts, total_samples, num_classes, weight_smooth_factor
            )
            
            if isinstance(train_result, dict):
                train_loss = train_result['loss']
                train_acc = train_result['acc']
                train_class_loss = train_result.get('class_loss', 0.0)
                train_contrastive_loss = train_result.get('contrastive_loss', 0.0)
            else:
                train_loss, train_acc = train_result
                train_class_loss = 0.0
                train_contrastive_loss = 0.0
            
            # 更新学习率
            scheduler.step()
            
            # 验证
            val_result = validate(
                model, val_loader, criterion, device, use_amp, class_texts, num_classes,
                use_superclip_loss, class_counts, total_samples, weight_smooth_factor
            )
            
            if use_superclip_loss and len(val_result) > 5:
                val_loss, val_acc, all_predictions, all_labels, val_mAP, val_class_loss, val_contrastive_loss = val_result
            else:
                val_loss, val_acc, all_predictions, all_labels, val_mAP = val_result[:5]
                val_class_loss = 0.0
                val_contrastive_loss = 0.0
            
            # 计算详细指标（每个epoch都计算）
            val_precision = val_mAP  # 默认使用mAP作为precision
            val_recall = 0.0
            val_f1 = 0.0
            if num_classes is not None and len(all_predictions) > 0 and len(all_labels) > 0:
                try:
                    from calculate_metrics import calculate_classification_metrics
                    epoch_metrics = calculate_classification_metrics(all_labels, all_predictions, num_classes)
                    val_precision = epoch_metrics.get('precision_macro', val_mAP)
                    val_recall = epoch_metrics.get('recall_macro', 0.0)
                    val_f1 = epoch_metrics.get('f1_macro', 0.0)
                except Exception as e:
                    # 如果计算失败，使用默认值
                    pass
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_mAP'].append(val_mAP)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)
            
            # 如果使用 SuperCLIP 损失，记录分类损失和对比损失
            if use_superclip_loss:
                if 'train_class_loss' not in history:
                    history['train_class_loss'] = []
                    history['train_contrastive_loss'] = []
                    history['val_class_loss'] = []
                    history['val_contrastive_loss'] = []
                history['train_class_loss'].append(train_class_loss)
                history['train_contrastive_loss'].append(train_contrastive_loss)
                history['val_class_loss'].append(val_class_loss)
                history['val_contrastive_loss'].append(val_contrastive_loss)
            
            # 保存checkpoint
            checkpoint = {
                'epoch': epoch,
                'fold': fold_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_mAP': val_mAP,
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'best_val_mAP': best_val_mAP,
                'config': config,
                'class_to_idx': full_dataset.class_to_idx,
                'class_texts': class_texts,
            }
            
            # 保存最新checkpoint
            torch.save(checkpoint, fold_dir / 'checkpoint_latest.pth')
            
            # 保存最佳模型（基于mAP）
            improved = False
            if val_mAP > best_val_mAP:
                best_val_mAP = val_mAP
                best_epoch = epoch + 1  # epoch从0开始，显示时+1
                checkpoint['best_val_mAP'] = best_val_mAP
                checkpoint['best_epoch'] = best_epoch
                
                # 使用当前epoch已计算的指标作为最佳指标
                best_metrics = {
                    'precision_macro': val_precision,
                    'recall_macro': val_recall,
                    'f1_macro': val_f1
                }
                
                # 保存最佳指标到checkpoint
                checkpoint['best_precision'] = val_precision
                checkpoint['best_recall'] = val_recall
                checkpoint['best_f1'] = val_f1
                checkpoint['best_metrics'] = best_metrics
                
                if save_best:
                    torch.save(checkpoint, fold_dir / 'checkpoint_best.pth')
                print(f"✓ 保存最佳模型 (val_mAP: {val_mAP:.2f}%, Precision: {val_precision:.2f}%, Recall: {val_recall:.2f}%, F1: {val_f1:.2f}%)")
            
            # 更新其他最佳指标
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint['best_val_acc'] = best_val_acc
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint['best_val_loss'] = best_val_loss
                improved = True  # 用于早停判断
            
            # 早停检查（基于val_loss）
            if use_early_stopping:
                if early_stopping_monitor == 'val_loss':
                    # 监控验证损失（越低越好）
                    if val_loss < (best_val_loss - early_stopping_min_delta):
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                else:  # val_acc
                    if val_acc > (best_val_acc + early_stopping_min_delta):
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                
                if early_stopping_counter >= early_stopping_patience:
                    print(f"\nFold {fold_num} 早停触发！连续 {early_stopping_patience} 个epoch没有改善")
                    break
            
            # 保存历史
            with open(fold_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            if use_superclip_loss:
                print(f"Train Loss: {train_loss:.4f} (Class: {train_class_loss:.4f}, Contrastive: {train_contrastive_loss:.4f}), Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f} (Class: {val_class_loss:.4f}, Contrastive: {val_contrastive_loss:.4f}), Val Acc: {val_acc:.2f}%, Val mAP: {val_mAP:.2f}%")
            else:
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val mAP: {val_mAP:.2f}%")
            print(f"Best Val mAP: {best_val_mAP:.2f}%, Best Val Acc: {best_val_acc:.2f}%")
            if use_early_stopping:
                print(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}")
        
        # 记录fold的最终结果
        all_fold_results['fold_train_loss'].append(history['train_loss'][-1])
        all_fold_results['fold_train_acc'].append(history['train_acc'][-1])
        all_fold_results['fold_val_loss'].append(history['val_loss'][-1])
        all_fold_results['fold_val_acc'].append(history['val_acc'][-1])
        all_fold_results['fold_val_mAP'].append(history['val_mAP'][-1])
        all_fold_results['fold_best_val_acc'].append(best_val_acc)
        all_fold_results['fold_best_val_mAP'].append(best_val_mAP)
        # 如果best_epoch仍然是0（没有更新），使用最后一个epoch
        if best_epoch == 0:
            best_epoch = len(history['val_mAP'])
        all_fold_results['fold_best_epoch'].append(best_epoch)
        
        # 记录最佳epoch的详细指标
        if best_metrics is not None:
            all_fold_results['fold_best_precision'].append(best_metrics.get('precision_macro', 0.0))
            all_fold_results['fold_best_recall'].append(best_metrics.get('recall_macro', 0.0))
            all_fold_results['fold_best_f1'].append(best_metrics.get('f1_macro', 0.0))
        else:
            all_fold_results['fold_best_precision'].append(best_val_mAP)  # 如果没有详细指标，使用mAP
            all_fold_results['fold_best_recall'].append(0.0)
            all_fold_results['fold_best_f1'].append(0.0)
        
        print(f"\nFold {fold_num} 完成！最佳验证mAP: {best_val_mAP:.2f}%, 最佳验证准确率: {best_val_acc:.2f}%")
    
    # 计算平均指标
    avg_results = {
        'avg_train_loss': np.mean(all_fold_results['fold_train_loss']),
        'std_train_loss': np.std(all_fold_results['fold_train_loss']),
        'avg_train_acc': np.mean(all_fold_results['fold_train_acc']),
        'std_train_acc': np.std(all_fold_results['fold_train_acc']),
        'avg_val_loss': np.mean(all_fold_results['fold_val_loss']),
        'std_val_loss': np.std(all_fold_results['fold_val_loss']),
        'avg_val_acc': np.mean(all_fold_results['fold_val_acc']),
        'std_val_acc': np.std(all_fold_results['fold_val_acc']),
        'avg_val_mAP': np.mean(all_fold_results['fold_val_mAP']),
        'std_val_mAP': np.std(all_fold_results['fold_val_mAP']),
        'avg_best_val_acc': np.mean(all_fold_results['fold_best_val_acc']),
        'std_best_val_acc': np.std(all_fold_results['fold_best_val_acc']),
        'avg_best_val_mAP': np.mean(all_fold_results['fold_best_val_mAP']),
        'std_best_val_mAP': np.std(all_fold_results['fold_best_val_mAP']),
        'avg_best_precision': np.mean(all_fold_results['fold_best_precision']),
        'std_best_precision': np.std(all_fold_results['fold_best_precision']),
        'avg_best_recall': np.mean(all_fold_results['fold_best_recall']),
        'std_best_recall': np.std(all_fold_results['fold_best_recall']),
        'avg_best_f1': np.mean(all_fold_results['fold_best_f1']),
        'std_best_f1': np.std(all_fold_results['fold_best_f1']),
    }
    
    # 保存汇总结果
    summary = {
        'config': config,
        'fold_results': all_fold_results,
        'average_results': avg_results,
    }
    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print(f"交叉验证完成！")
    print(f"{'='*80}")
    
    # 打印详细结果
    print(f"\n详细结果:")
    for fold_num in range(1, n_splits + 1):
        idx = fold_num - 1
        best_epoch = all_fold_results['fold_best_epoch'][idx]
        best_mAP = all_fold_results['fold_best_val_mAP'][idx]
        best_acc = all_fold_results['fold_best_val_acc'][idx]
        precision = all_fold_results['fold_best_precision'][idx]
        recall = all_fold_results['fold_best_recall'][idx]
        f1 = all_fold_results['fold_best_f1'][idx]
        
        print(f"  Fold {fold_num}:")
        print(f"    最佳验证mAP: {best_mAP:.2f}% (Epoch {best_epoch})")
        print(f"    最佳验证准确率: {best_acc:.2f}% (Epoch {best_epoch})")
        print(f"    mAP: {best_mAP:.2f}%")
        print(f"    Precision: {precision:.2f}%")
        print(f"    Recall: {recall:.2f}%")
        print(f"    F1 Score: {f1:.2f}%")
    
    # 打印平均结果
    print(f"\n平均结果:")
    print(f"  平均最佳验证mAP: {avg_results['avg_best_val_mAP']:.2f}% ± {avg_results['std_best_val_mAP']:.2f}%")
    print(f"  平均最佳验证准确率: {avg_results['avg_best_val_acc']:.2f}% ± {avg_results['std_best_val_acc']:.2f}%")
    print(f"  平均mAP: {avg_results['avg_best_val_mAP']:.2f}% ± {avg_results['std_best_val_mAP']:.2f}%")
    print(f"  平均Precision: {avg_results['avg_best_precision']:.2f}% ± {avg_results['std_best_precision']:.2f}%")
    print(f"  平均Recall: {avg_results['avg_best_recall']:.2f}% ± {avg_results['std_best_recall']:.2f}%")
    print(f"  平均F1 Score: {avg_results['avg_best_f1']:.2f}% ± {avg_results['std_best_f1']:.2f}%")
    print(f"  平均最终验证准确率: {avg_results['avg_val_acc']:.2f}%")
    print(f"  平均最终验证损失: {avg_results['avg_val_loss']:.4f}")
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"  - cv_summary.json: 交叉验证汇总结果")
    print(f"  - folds_info.json: 各折的数据划分信息")
    print(f"  - fold_N/: 各折的训练历史和最佳模型")
    
    return summary


def train_multiple_configs(data_dir, output_base_dir, configs):
    """
    训练多个配置组合
    
    Args:
        data_dir: 数据目录
        output_base_dir: 输出基础目录
        configs: 配置列表，每个配置是一个字典
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"训练配置 {i+1}/{len(configs)}")
        print(f"{'='*80}")
        
        # 生成输出目录名称
        img_enc = config.get('image_encoder_name', config.get('image_encoder', 'unknown'))
        txt_enc = config.get('text_encoder_name', config.get('text_encoder', 'unknown'))
        output_dir = output_base_dir / f"{img_enc}_{txt_enc}"
        
        print(f"输出目录: {output_dir}")
        
        try:
            # 检查是否使用交叉验证
            use_cv = config.get('use_cv', False) or config.get('n_splits', 0) > 1
            if use_cv:
                n_splits = config.get('n_splits', 5)
                print(f"使用 {n_splits} 折交叉验证训练")
                
                # 提取交叉验证相关参数
                cv_params = {
                    'n_splits': n_splits,
                    'random_state': config.get('random_state', 42),
                    'save_best': config.get('save_best', True),
                }
                
                # 提取早停参数
                early_stopping_params = {
                    'early_stopping_patience': config.get('early_stopping_patience'),
                    'early_stopping_min_delta': config.get('early_stopping_min_delta', 0.0),
                    'early_stopping_monitor': config.get('early_stopping_monitor', 'val_loss'),
                }
                
                # 提取文本描述参数
                text_params = {
                    'text_template': config.get('text_template'),
                    'class_texts_dict': config.get('class_texts_dict'),
                    'class_texts_file': config.get('class_texts_file'),
                }
                
                # 提取加权采样参数
                weighted_sampling_params = {
                    'use_weighted_sampling': config.get('use_weighted_sampling', False),
                    'weight_method': config.get('weight_method', 'inverse_freq'),
                    'weight_smooth_factor': config.get('weight_smooth_factor', 1.0),
                }
                
                # 构建训练参数（排除use_cv和不需要的参数）
                train_params = {
                    'data_dir': data_dir,
                    'output_dir': output_dir,
                    'image_encoder_name': config.get('image_encoder_name', config.get('image_encoder', 'resnet50')),
                    'text_encoder_name': config.get('text_encoder_name', config.get('text_encoder', 'bert-base-chinese')),
                    'embed_dim': config.get('embed_dim', 512),
                    'batch_size': config.get('batch_size', 32),
                    'epochs': config.get('epochs', 100),
                    'learning_rate': config.get('learning_rate', 1e-4),
                    'weight_decay': config.get('weight_decay', 0.01),
                    'temperature': config.get('temperature', 0.07),
                    'img_size': config.get('img_size', 224),
                    'augmentation': config.get('augmentation', 'standard'),
                    'num_workers': config.get('num_workers', 4),
                    'use_amp': config.get('use_amp', True),
                    'gpu_id': config.get('gpu_id', 0),
                }
                
                # 提取 SuperCLIP 损失参数
                superclip_loss_params = {
                    'use_superclip_loss': config.get('use_superclip_loss', False),
                    'class_loss_weight': config.get('class_loss_weight', 1.0),
                    'contrastive_loss_weight': config.get('contrastive_loss_weight', 1.0),
                    'use_focal_loss': config.get('use_focal_loss', False),
                    'focal_alpha': config.get('focal_alpha', 0.25),
                    'focal_gamma': config.get('focal_gamma', 2.0),
                }
                
                # 合并所有参数（确保不包含use_cv等不需要的参数）
                all_params = {**train_params, **cv_params, **early_stopping_params, **text_params, **weighted_sampling_params, **superclip_loss_params}
                
                # 移除任何可能存在的use_cv参数（虽然应该不会出现，但为了安全）
                all_params.pop('use_cv', None)
                
                # 移除任何可能存在的use_cv参数（虽然应该不会出现，但为了安全）
                all_params.pop('use_cv', None)
                
                train_clip_cross_validation(**all_params)
            else:
                # 单配置训练，移除use_cv和n_splits参数
                train_config = {k: v for k, v in config.items() if k not in ['use_cv', 'n_splits']}
                # 确保加权采样参数被传递
                if 'use_weighted_sampling' not in train_config:
                    train_config['use_weighted_sampling'] = False
                if 'weight_method' not in train_config:
                    train_config['weight_method'] = 'inverse_freq'
                if 'weight_smooth_factor' not in train_config:
                    train_config['weight_smooth_factor'] = 1.0
                # 单配置训练已删除，只支持交叉验证
                print("单配置训练已删除，请使用交叉验证模式 (--use-cv)")
                continue
        except Exception as e:
            print(f"✗ 配置训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLIP模型训练脚本')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, required=True, help='数据目录（按类别组织的文件夹）')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    
    # 模型参数
    parser.add_argument('--image-encoder', type=str, default='resnet50',
                       help='图像编码器名称。使用 MetaCLIP: metaclip:ViT-B-32-quickgelu:metaclip_400m')
    parser.add_argument('--text-encoder', type=str, default='bert-base-chinese',
                       help='文本编码器名称（使用 MetaCLIP 时会被忽略）')
    parser.add_argument('--use-metaclip', action='store_true',
                       help='使用 MetaCLIP 预训练模型（替代 --image-encoder）')
    parser.add_argument('--metaclip-model', type=str, default='ViT-B-32-quickgelu',
                       help='MetaCLIP 模型名称（仅在 --use-metaclip 时使用）')
    parser.add_argument('--metaclip-pretrained', type=str, default='metaclip_400m',
                       help='MetaCLIP 预训练权重标识（仅在 --use-metaclip 时使用）')
    parser.add_argument('--embed-dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--temperature', type=float, default=0.07, help='温度参数')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--img-size', type=int, default=224, help='图像大小')
    parser.add_argument('--augmentation', type=str, default='standard', 
                       choices=['none', 'minimal', 'standard'],
                       help='数据增强类型')
    
    # 文本描述参数
    parser.add_argument('--text-template', type=str, default=None,
                       help='文本模板，例如 "这是一张{class_name}的图片"，默认使用类别名称')
    parser.add_argument('--class-texts-file', type=str, default=None,
                       help='类别文本描述JSON文件路径，格式: {"类别1": "描述1", "类别2": "描述2"}')
    
    # 其他参数
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度训练')
    parser.add_argument('--resume-from', type=str, default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--no-save-best', action='store_true', help='不保存最佳模型')
    
    # 交叉验证参数
    parser.add_argument('--use-cv', action='store_true', help='使用K折交叉验证训练')
    parser.add_argument('--n-splits', type=int, default=5, help='交叉验证折数（默认5折）')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子')
    
    # 早停参数
    parser.add_argument('--early-stopping-patience', type=int, default=None, 
                       help='早停耐心值，如果设置为None则不使用早停（默认None）')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                       help='早停最小改进阈值（默认0.0）')
    parser.add_argument('--early-stopping-monitor', type=str, default='val_loss',
                       choices=['val_acc', 'val_loss'],
                       help='早停监控指标：val_acc（验证准确率）或val_loss（验证损失），默认val_loss')
    
    # 加权采样参数（用于处理类别不平衡）
    parser.add_argument('--use-weighted-sampling', action='store_true', 
                       help='启用加权采样以处理类别不平衡问题')
    parser.add_argument('--weight-method', type=str, default='inverse_freq',
                       choices=['inverse_freq', 'inverse_sqrt', 'balanced'],
                       help='权重计算方法：inverse_freq（逆频率，默认）、inverse_sqrt（逆平方根频率）、balanced（平衡，同inverse_freq）')
    parser.add_argument('--weight-smooth-factor', type=float, default=1.0,
                       help='权重平滑因子，用于避免权重过大（默认1.0）')
    
    # SuperCLIP 损失参数
    parser.add_argument('--use-superclip-loss', action='store_true',
                       help='使用 SuperCLIP 损失函数（结合分类损失和对比损失）')
    parser.add_argument('--class-loss-weight', type=float, default=1.0,
                       help='分类损失权重（默认1.0）')
    parser.add_argument('--contrastive-loss-weight', type=float, default=1.0,
                       help='对比损失权重（默认1.0）')
    
    # Focal Loss 参数（用于分类损失）
    parser.add_argument('--use-focal-loss', action='store_true',
                       help='使用 Focal Loss 作为分类损失（默认False，使用加权交叉熵）')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                       help='Focal Loss 的 alpha 参数（默认0.25）')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Focal Loss 的 gamma 参数（默认2.0）')
    
    # 多配置训练
    parser.add_argument('--multi-config', action='store_true', help='使用多配置训练模式')
    parser.add_argument('--config-file', type=str, default=None, help='配置文件路径（JSON格式）')
    
    args = parser.parse_args()
    
    if args.multi_config or args.config_file:
        # 多配置训练模式
        if args.config_file:
            with open(args.config_file, 'r') as f:
                configs = json.load(f)
        else:
            # 默认配置组合
            image_encoders = [
                'starnet_dual_pyramid_rcf',
                'resnet50',
                'resnet18',
            ]
            text_encoders = [
                'bert-base-chinese',
            ]
            
            configs = []
            for img_enc, txt_enc in product(image_encoders, text_encoders):
                config = {
                    'image_encoder_name': img_enc,
                    'text_encoder_name': txt_enc,
                    'embed_dim': args.embed_dim,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'temperature': args.temperature,
                    'img_size': args.img_size,
                    'augmentation': args.augmentation,
                    'num_workers': args.num_workers,
                    'use_amp': not args.no_amp,
                    'gpu_id': args.gpu_id,
                    'save_best': not args.no_save_best,
                }
                configs.append(config)
        
        train_multiple_configs(args.data_dir, args.output_dir, configs)
    else:
        # 单配置训练模式
        if args.use_cv:
            # 交叉验证训练模式
            train_clip_cross_validation(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                image_encoder_name=args.image_encoder,
                text_encoder_name=args.text_encoder,
                use_metaclip=args.use_metaclip,
                metaclip_model_name=args.metaclip_model,
                metaclip_pretrained=args.metaclip_pretrained,
                embed_dim=args.embed_dim,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                img_size=args.img_size,
                augmentation=args.augmentation,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
                gpu_id=args.gpu_id,
                n_splits=args.n_splits,
                random_state=args.random_state,
                save_best=not args.no_save_best,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                early_stopping_monitor=args.early_stopping_monitor,
                text_template=args.text_template,
                class_texts_file=args.class_texts_file,
                use_weighted_sampling=args.use_weighted_sampling,
                weight_method=args.weight_method,
                weight_smooth_factor=args.weight_smooth_factor,
                use_superclip_loss=args.use_superclip_loss,
                class_loss_weight=args.class_loss_weight,
                contrastive_loss_weight=args.contrastive_loss_weight,
                use_focal_loss=args.use_focal_loss,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
            )
        else:
            # 普通训练模式已删除，只支持交叉验证
            print("错误：单配置训练模式已删除，请使用交叉验证模式 (--use-cv)")
            sys.exit(1)

