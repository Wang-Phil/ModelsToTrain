"""
多分类任务训练脚本
支持多种模型、数据增强、损失函数和训练参数配置
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
# 兼容不同 PyTorch 版本的 GradScaler 导入
try:
    from torch.cuda.amp import GradScaler
except ImportError:
    try:
        from torch.amp import GradScaler
    except ImportError:
        GradScaler = None
# 导入模型
from models.classic_models import create_model as create_classic_model
from models.starnet import starnet_s1, starnet_s2, starnet_s3, starnet_s4
from models.starnet_attention import starnet_s2_multi_head
from models.multi_head_classifiers import LDAM_CB_Loss
from models.starnet_parallel_sa import starnet_s1_parallel_sa
from models.starnet_res_fusion import starnet_s1_res_fusion
from models.starnet_s1_pyramid import starnet_s1_pyramid
from models.starnet_dual_pyramid import starnet_dual_pyramid
from models.starnet_dual_pyramid_sa import starnet_dual_pyramid_sa
from models.starnet_dual_swin_pyramid import starnet_dual_swin_pyramid
from models.starnet_dual_pyramid_rcf import starnet_dual_pyramid_rcf
from models.starnet_s1_concat import starnet_s1_cross_star, starnet_s1_cross_star_add, starnet_s1_cross_star_samescale
from models.starnet_s1_artifact import starnet_artifact_s1
from models.starnet_GRN import starnet_s1_GRN 
from models.starnet_gate import starnet_gated_s1
from models.starnet_final_model import starnet_s1_final, starnet_s2_final, starnet_s3_final
from models.starnet_sa_variants import starnet_sa_s1, starnet_sa_s2, starnet_sa_s3, starnet_sa_s4
from models.starnet_crosswithgln import starnet_s1_cross_with_gln, starnet_s2_cross_with_gln
from models.starnet_cfs import starnet_cf_s3
from models.lsnet import lsnet_t, lsnet_s, lsnet_b


class ImageFolderDataset(Dataset):
    """图像文件夹数据集，支持多分类任务"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.class_counts = defaultdict(int)
        
        # 获取所有类别
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # 收集所有图像文件
        for class_name in classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), class_idx))
                    self.class_counts[class_idx] += 1
        
        print(f"数据集: {len(self.samples)} 个样本, {len(classes)} 个类别")
        print(f"类别: {classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            # 返回一个黑色图像
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            label = 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def compute_drw_weights(cls_num_list, beta=0.9999):
    cls_num_list = np.array(cls_num_list, dtype=np.float32)
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / (effective_num + 1e-12)
    weights = weights / np.sum(weights) * len(cls_num_list)
    return weights


def get_data_augmentation(augmentation_type='standard', img_size=224):
    """
    获取数据增强策略
    
    Args:
        augmentation_type: 增强类型
            - 'none': 无增强
            - 'minimal': 最小增强（仅resize和normalize）
            - 'standard': 标准增强（随机翻转、颜色抖动等）
            - 'strong': 强增强（Mixup, CutMix等）
            - 'medical': 医学图像增强（适合医学图像）
        img_size: 图像大小
    """
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
    elif augmentation_type == 'strong':
        train_transform = transforms.Compose([
            transforms.Resize((int(img_size * 1.2), int(img_size * 1.2))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
        ])
    elif augmentation_type == 'medical':
        # 适合医学图像的增强策略
        train_transform = transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"未知的增强类型: {augmentation_type}")
    
    # 验证集使用相同的标准化，但无增强
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy"""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class AsymmetricLossSingleLabel(nn.Module):
    """Asymmetric Loss for single-label classification."""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        num_classes = inputs.size(1)
        device = inputs.device
        one_hot = torch.zeros((inputs.size(0), num_classes), device=device)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        preds = torch.sigmoid(inputs)
        xs_pos = preds
        xs_neg = 1.0 - preds

        if self.clip and self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1)

        xs_pos = torch.clamp(xs_pos, min=self.eps)
        xs_neg = torch.clamp(xs_neg, min=self.eps)

        loss = one_hot * torch.log(xs_pos) + (1.0 - one_hot) * torch.log(xs_neg)
        asymmetric_w = torch.pow(1 - xs_pos, self.gamma_pos) * one_hot + torch.pow(xs_neg, self.gamma_neg) * (1.0 - one_hot)
        loss *= asymmetric_w

        if self.reduction == 'mean':
            return -loss.mean()
        elif self.reduction == 'sum':
            return -loss.sum()
        else:
            return -loss


class LDAMLoss(nn.Module):
    """LDAM Loss (Label Distribution Aware Margin)"""

    def __init__(self, cls_num_list, max_m=0.5, s=30, reduction='mean'):
        super(LDAMLoss, self).__init__()
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list + 1e-12))
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer('m_list', torch.tensor(m_list, dtype=torch.float32))
        self.s = s
        self.reduction = reduction
        self.weight = None

    def forward(self, inputs, targets):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        index = torch.zeros_like(inputs)
        index.scatter_(1, targets.view(-1, 1), 1.0)

        margin = self.m_list[targets].unsqueeze(1)
        inputs_m = inputs - index * margin
        outputs = inputs_m * self.s
        return F.cross_entropy(outputs, targets, weight=self.weight, reduction=self.reduction)


def compute_drw_weights(cls_num_list, beta=0.9999):
    cls_num_list = np.array(cls_num_list, dtype=np.float32)
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / (effective_num + 1e-12)
    weights = weights / np.sum(weights) * len(cls_num_list)
    return weights


def get_loss_function(loss_type='ce', num_classes=9, **kwargs):
    """
    获取损失函数
    
    Args:
        loss_type: 损失函数类型
            - 'ce': CrossEntropyLoss
            - 'focal': Focal Loss
            - 'label_smoothing': Label Smoothing Cross Entropy
            - 'weighted_ce': Weighted Cross Entropy (用于类别不平衡)
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        alpha = kwargs.get('focal_alpha', None)
        gamma = kwargs.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    elif loss_type == 'asl':
        gamma_neg = kwargs.get('asl_gamma_neg', 4.0)
        gamma_pos = kwargs.get('asl_gamma_pos', 1.0)
        clip = kwargs.get('asl_clip', 0.05)
        eps = kwargs.get('asl_eps', 1e-8)
        return AsymmetricLossSingleLabel(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, eps=eps)
    elif loss_type == 'ldam':
        cls_num_list = kwargs.get('cls_num_list')
        if cls_num_list is None:
            raise ValueError("LDAM requires cls_num_list (class counts)")
        max_m = kwargs.get('ldam_margin', 0.5)
        ldam_s = kwargs.get('ldam_s', 30.0)
        return LDAMLoss(cls_num_list, max_m=max_m, s=ldam_s)
    elif loss_type == 'weighted_ce':
        weights = kwargs.get('class_weights', None)
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


def extract_logits(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    if hasattr(outputs, 'logits'):
        return outputs.logits
    return outputs


def create_model(model_name, num_classes=9, pretrained=False, **kwargs):
    """
    创建模型
    
    支持的模型:
    Classic Models: resnet18, resnet50, resnet101, inceptionv3, densenet121, densenet161, densenet201,
                    mobilenetv2, mobilenetv3_small, mobilenetv3_large, googlenet, efficientnet_b0, 
                    efficientnetv2_s, efficientnetv2_m, efficientnetv2_l,
                    starnet_s050, starnet_s100, starnet_s150
    ConvNeXtV2: convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_nano
    StarNeXt: starnext_tiny, starnext_base, starnext_large, starnext_small, starnext_nano
    StarNet: starnet_s1, starnet_s2, starnet_s3, starnet_s4
    StarNet Parallel SA: starnet_s1_parallel_sa
    StarNet Residual Fusion: starnet_s1_res_fusion
    StarNet Pyramid SA: starnet_s1_pyramid
    StarNet Global Fusion: starnet_s1_global_fusion
    StarNet Dual Pyramid: starnet_dual_pyramid
    StarNet Dual Pyramid SA: starnet_dual_pyramid_sa
    StarNet Dual Pyramid RCF: starnet_dual_pyramid_rcf
    StarNet Dual Pyramid Swin: starnet_dual_swin_pyramid
    StarNet ARConv: starnet_arconv_s1, starnet_arconv_s2, starnet_arconv_s3, starnet_arconv_s4 (StarNet with ARConv)
    StarNet DCNv4: starnet_dcnv4_s1, starnet_dcnv4_s2, starnet_dcnv4_s3, starnet_dcnv4_s4 (StarNet with DCNv4 for both dwconv and dwconv2)
    StarNet Hybrid: starnet_hybrid_s, starnet_hybrid_t (混合 StarBlock + Transformer)
    StarNet FPN: starnet_fpn_s, starnet_fpn_t (StarNet + Feature Pyramid Network)
    StarNet ViT Hybrid: starnet_vit_hybrid_s, starnet_vit_hybrid_t (StarBlock + MHA + StarMLP)
    StarNet ViT Merged: starnet_vit_merged_s, starnet_vit_merged_t (StarAttentionBlock + StarBlockConv)
    StarNet ViT Single Fusion: starnet_vit_single_fusion (Frozen ViT + StarNet, fusion at Stage4)
    StarNet LSK: starnet_s1_lsk (StarNet with Large Selective Kernel for spatial attention)
    StarNet GRN: starnet_s1_GRN (StarNet with Gated Response Normalization)
    StarNet ODConv: starnet_s1_odconv (StarNet with Omni-Dimensional Dynamic Convolution)
    StarNet Gated: starnet_gated_s1 (StarNet with configurable gating mechanisms: none, intra, pre, post, swiglu)
    StarNet Gate Stage: starnet_s1_gated (StarNet with Dynamic Inter-Stage Gating using StarGate)
    StarNet Skip Gate: starnet_s1_gated_skip (StarNet with Dynamic Skip Gating from Stage 2 to Stage 4)
    StarNet Dilated: starnet_s1_dl (StarNet with Dilated Convolution in Stages 2 and 3)
    StarNet LoRA: starnet_s1_lora (StarNet with Texture-Aware LoRA for medical imaging)
    StarNet Cross-Star: starnet_s1_cross_star (StarNet with Inception-style Cross-Star Operation)
    StarNet Artifact: starnet_artifact_s1 (StarNet with Artifact-Suppressing Soft-Gating)
    StarNet Final: starnet_s1_final, starnet_s2_final, starnet_s3_final (Final optimized StarNet models)
    StarNet SA Variants: 
        - starnet_sa_s1 (所有stage都加空间注意力: stage 0,1,2,3)
        - starnet_sa_s2 (第一个stage不加注意力: stage 1,2,3加注意力)
        - starnet_sa_s3 (前两个stage不加注意力: stage 2,3加注意力)
        - starnet_sa_s4 (前三个stage不加注意力: 只有stage 3加注意力)
    StarNet Cross-Star Ablation: 
        - starnet_s1_cross_star (D-基线: Concat((x_3A * x_7B), (x_7A * x_3B)))
        - starnet_s1_cross_star_add (D1-加法: Concat((x_3A + x_7B), (x_7A + x_3B)))
        - starnet_s1_cross_star_samescale (D2-同尺度: Concat((x_3A * x_3B), (x_7A * x_7B)))
    StarNet Cross-Star with GLN:
        - starnet_s1_cross_with_gln (StarNet with GRN + Cross-Star Block: 浅层用Block+SA, 深层用CrossStarBlock)
        - starnet_s2_cross_with_gln (StarNet with GRN + Cross-Star Block: 浅层用Block+SA, 深层用CrossStarBlock)
    StarNet Channel Frequency (CF):
        - starnet_cf_s3 (StarNet with Channel Frequency Attention & Long-Tail Re-scaling: 深层使用CFStarBlock)
    LSNet: lsnet_t, lsnet_s, lsnet_b
    MogaNet: moganet_small, moganet_base, moganet_large, moganet_xlarge (如果可用)
    """
    model_name = model_name.lower()
    
    # Classic models
    classic_models = ['resnet18', 'resnet50', 'resnet101', 'inceptionv3', 'densenet121', 'densenet161', 'densenet201',
                     'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large', 'googlenet', 'efficientnet_b0', 
                     'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l', 'starnet_s050', 'starnet_s100', 
                     'starnet_s150', 'unet', 'transunet']
    
    if model_name in classic_models:
        return create_classic_model(model_name, num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    # StarNet models
    elif model_name == 'starnet_s1':
        return starnet_s1(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s2':
        return starnet_s2(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s2_multi_head':
        # 多分类头模式，需要 cls_num_list
        cls_num_list = kwargs.get('cls_num_list', None)
        return starnet_s2_multi_head(pretrained=pretrained, num_classes=num_classes, cls_num_list=cls_num_list)
    elif model_name == 'starnet_s3':
        return starnet_s3(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s4':
        return starnet_s4(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s5':
        return starnet_s5(pretrained=pretrained, num_classes=num_classes)
    # StarNet Parallel Spatial Attention
    elif model_name == 'starnet_s1_parallel_sa':
        return starnet_s1_parallel_sa(pretrained=pretrained, num_classes=num_classes)
    
    # StarNet Residual Fusion
    elif model_name == 'starnet_s1_res_fusion':
        return starnet_s1_res_fusion(pretrained=pretrained, num_classes=num_classes)
    
    # StarNet Pyramid Spatial Attention
    elif model_name == 'starnet_s1_pyramid':
        return starnet_s1_pyramid(pretrained=pretrained, num_classes=num_classes)
    
    # StarNet Dual Pyramid (Local + Global)
    elif model_name == 'starnet_dual_pyramid':
        return starnet_dual_pyramid(pretrained=pretrained, num_classes=num_classes)
    
    # StarNet Dual Pyramid with Spatial Attention
    elif model_name == 'starnet_dual_pyramid_sa':
        return starnet_dual_pyramid_sa(pretrained=pretrained, num_classes=num_classes)
    
    # StarNet Dual Pyramid with Swin Transformer
    elif model_name == 'starnet_dual_swin_pyramid':
        return starnet_dual_swin_pyramid(pretrained=pretrained, num_classes=num_classes)
    
    # StarNet Dual Pyramid with Residual Cascaded Fusion (RCF)
    elif model_name == 'starnet_dual_pyramid_rcf':
        return starnet_dual_pyramid_rcf(pretrained=pretrained, num_classes=num_classes)

    # StarNet GRN models
    elif model_name == 'starnet_s1_grn':
        return starnet_s1_GRN(pretrained=pretrained, num_classes=num_classes)

    # StarNet Cross-Star Ablation Studies
    elif model_name == 'starnet_s1_cross_star':
        return starnet_s1_cross_star(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s1_cross_star_add':
        return starnet_s1_cross_star_add(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s1_cross_star_samescale':
        return starnet_s1_cross_star_samescale(pretrained=pretrained, num_classes=num_classes)

    # StarNet Artifact models
    elif model_name == 'starnet_artifact_s1':
        return starnet_artifact_s1(pretrained=pretrained, num_classes=num_classes)

    # StarNet Final models
    elif model_name == 'starnet_s1_final':
        return starnet_s1_final(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s2_final':
        return starnet_s2_final(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s3_final':
        return starnet_s3_final(pretrained=pretrained, num_classes=num_classes)
    
    # StarNet SA Variants (Spatial Attention Variants)
    elif model_name == 'starnet_sa_s1':
        return starnet_sa_s1(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_sa_s2':
        return starnet_sa_s2(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_sa_s3':
        return starnet_sa_s3(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_sa_s4':
        return starnet_sa_s4(pretrained=pretrained, num_classes=num_classes)
    
    # StarNet Cross-Star with GLN (GRN + Cross-Star Block)
    elif model_name == 'starnet_s1_cross_with_gln':
        return starnet_s1_cross_with_gln(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s2_cross_with_gln':
        return starnet_s2_cross_with_gln(pretrained=pretrained, num_classes=num_classes)
    
    # StarNet Channel Frequency (CF) with Long-Tail Re-scaling
    elif model_name == 'starnet_cf_s3':
        # 需要 cls_num_list 参数用于长尾重标定
        cls_num_list = kwargs.get('cls_num_list', None)
        use_attn = kwargs.get('use_attn', 2)  # 默认从 stage 2 开始使用空间注意力
        return starnet_cf_s3(pretrained=pretrained, num_classes=num_classes, 
                            cls_num_list=cls_num_list, use_attn=use_attn)
    
    # StarNet Cross-Star Ablation Studies
    elif model_name == 'starnet_s1_cross_star':
        return starnet_s1_cross_star(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s1_cross_star_add':
        return starnet_s1_cross_star_add(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'starnet_s1_cross_star_samescale':
        return starnet_s1_cross_star_samescale(pretrained=pretrained, num_classes=num_classes)

    # LSNet models
    elif model_name == 'lsnet_t':
        return lsnet_t(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'lsnet_s':
        return lsnet_s(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'lsnet_b':
        return lsnet_b(pretrained=pretrained, num_classes=num_classes)

    
    else:
        raise ValueError(f"未知的模型: {model_name}")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scheduler=None, use_amp=True, 
                is_multi_head=False, ldam_cb_loss=None, loss_weights=None, gradient_accumulation_steps=1):
    """
    训练一个epoch，支持AMP混合精度训练和多分类头
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数（用于单分类头）
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        scheduler: 学习率调度器
        use_amp: 是否使用混合精度
        is_multi_head: 是否是多分类头模式
        ldam_cb_loss: LDAM+CB Loss（多分类头模式需要）
        loss_weights: 多分类头loss权重，dict格式 {'softmax': 1.0, 'arcface': 1.0, 'cosface': 1.0, 'ldam': 1.0}
        gradient_accumulation_steps: 梯度累积步数（用于减少显存占用）
    
    Returns:
        如果是多分类头模式:
            (total_loss, avg_acc, accuracies_dict)
        如果是单分类头模式:
            (epoch_loss, epoch_acc)
    """
    model.train()
    
    if loss_weights is None:
        loss_weights = {'softmax': 1.0, 'arcface': 1.0, 'cosface': 1.0, 'ldam': 1.0}
    
    # 初始化 AMP scaler（仅在CUDA设备上使用）
    # 兼容 PyTorch 2.0+ 和旧版本
    if use_amp and device.type == 'cuda' and GradScaler is not None:
        scaler = GradScaler()
    else:
        scaler = None
    
    if is_multi_head:
        # 多分类头模式
        running_loss = 0.0
        corrects = {'softmax': 0, 'arcface': 0, 'cosface': 0, 'ldam': 0}
        total = 0
        focal_loss = FocalLoss(gamma=2.0)
        
        try:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train-MultiHead]')
            optimizer.zero_grad()  # 在循环开始前清零梯度
            
            for batch_idx, (images, labels) in enumerate(pbar):
                try:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # 使用 AMP 混合精度训练
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = model(images, labels=labels)
                            
                            # 计算每个分类头的 loss
                            loss_softmax = focal_loss(outputs['logits_softmax'], labels)
                            loss_arcface = focal_loss(outputs['logits_arcface'], labels)
                            loss_cosface = focal_loss(outputs['logits_cosface'], labels)
                            loss_ldam = ldam_cb_loss(outputs['logits_ldam'], labels)
                            
                            # 加权求和，并除以梯度累积步数
                            total_loss = (loss_weights['softmax'] * loss_softmax +
                                        loss_weights['arcface'] * loss_arcface +
                                        loss_weights['cosface'] * loss_cosface +
                                        loss_weights['ldam'] * loss_ldam) / gradient_accumulation_steps
                        
                        scaler.scale(total_loss).backward()
                        
                        # 只在累积步数达到时才更新参数
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            
                            # 清理显存
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                    else:
                        outputs = model(images, labels=labels)
                        
                        # 计算每个分类头的 loss
                        loss_softmax = focal_loss(outputs['logits_softmax'], labels)
                        loss_arcface = focal_loss(outputs['logits_arcface'], labels)
                        loss_cosface = focal_loss(outputs['logits_cosface'], labels)
                        loss_ldam = ldam_cb_loss(outputs['logits_ldam'], labels)
                        
                        # 加权求和，并除以梯度累积步数
                        total_loss = (loss_weights['softmax'] * loss_softmax +
                                    loss_weights['arcface'] * loss_arcface +
                                    loss_weights['cosface'] * loss_cosface +
                                    loss_weights['ldam'] * loss_ldam) / gradient_accumulation_steps
                        
                        total_loss.backward()
                        
                        # 只在累积步数达到时才更新参数
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # 清理显存
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                    
                    # 更新统计信息（使用原始loss值，不除以累积步数）
                    running_loss += total_loss.item() * gradient_accumulation_steps
                    
                    for head_name in ['softmax', 'arcface', 'cosface', 'ldam']:
                        logits = outputs[f'logits_{head_name}']
                        _, predicted = torch.max(logits.data, 1)
                        corrects[head_name] += (predicted == labels).sum().item()
                    
                    total += labels.size(0)
                    
                    # OneCycleLR需要在每个batch后更新
                    if scheduler is not None:
                        scheduler.step()
                        current_lr = optimizer.param_groups[0]['lr']
                        avg_acc = sum(corrects.values()) / (4 * total) * 100 if total > 0 else 0.0
                        pbar.set_postfix({
                            'loss': f'{total_loss.item() * gradient_accumulation_steps:.4f}',
                            'acc': f'{avg_acc:.2f}%',
                            'lr': f'{current_lr:.6f}'
                        })
                    else:
                        avg_acc = sum(corrects.values()) / (4 * total) * 100 if total > 0 else 0.0
                        pbar.set_postfix({
                            'loss': f'{total_loss.item() * gradient_accumulation_steps:.4f}',
                            'acc': f'{avg_acc:.2f}%'
                        })
                except Exception as e:
                    print(f"\n错误: 处理批次 {batch_idx} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    # 清理显存后重新抛出异常
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    raise
            
            # 处理最后一个不完整的累积批次
            if len(dataloader) % gradient_accumulation_steps != 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
            accuracies = {k: 100 * corrects[k] / total if total > 0 else 0.0 for k in corrects.keys()}
            avg_acc = sum(accuracies.values()) / len(accuracies)
            return epoch_loss, avg_acc, accuracies
        except Exception as e:
            print(f"\n严重错误: 训练epoch {epoch} 时出错: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        # 单分类头模式（原有逻辑）
        running_loss = 0.0
        correct = 0
        total = 0
        
        try:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
            optimizer.zero_grad()  # 在循环开始前清零梯度
            
            for batch_idx, (images, labels) in enumerate(pbar):
                try:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    # 使用 AMP 混合精度训练
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = extract_logits(model(images))
                            loss = criterion(outputs, labels)
                            # 梯度累积：除以累积步数
                            loss = loss / gradient_accumulation_steps
                        
                        scaler.scale(loss).backward()
                        
                        # 只在累积步数达到时才更新参数
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            
                            # 清理显存
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                    else:
                        outputs = extract_logits(model(images))
                        loss = criterion(outputs, labels)
                        # 梯度累积：除以累积步数
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                        
                        # 只在累积步数达到时才更新参数
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # 清理显存
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                    
                    # 更新统计信息（使用原始loss值，不除以累积步数）
                    running_loss += loss.item() * gradient_accumulation_steps
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # OneCycleLR需要在每个batch后更新
                    if scheduler is not None:
                        scheduler.step()
                        current_lr = optimizer.param_groups[0]['lr']
                        acc_str = f'{100 * correct / total:.2f}%' if total > 0 else '0.00%'
                        pbar.set_postfix({
                            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                            'acc': acc_str,
                            'lr': f'{current_lr:.6f}'
                        })
                    else:
                        acc_str = f'{100 * correct / total:.2f}%' if total > 0 else '0.00%'
                        pbar.set_postfix({
                            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                            'acc': acc_str
                        })
                except Exception as e:
                    print(f"\n错误: 处理批次 {batch_idx} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    # 清理显存后重新抛出异常
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    raise
            
            # 处理最后一个不完整的累积批次
            if len(dataloader) % gradient_accumulation_steps != 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
            epoch_acc = 100 * correct / total if total > 0 else 0.0
            return epoch_loss, epoch_acc
        except Exception as e:
            print(f"\n严重错误: 训练epoch {epoch} 时出错: {e}")
            import traceback
            traceback.print_exc()
            raise


def validate(model, dataloader, criterion, device, use_amp=True, is_multi_head=False, ldam_cb_loss=None):
    """
    验证模型，支持AMP混合精度和多分类头
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数（用于单分类头）
        device: 设备
        use_amp: 是否使用混合精度
        is_multi_head: 是否是多分类头模式
        ldam_cb_loss: LDAM+CB Loss（多分类头模式需要）
    
    Returns:
        如果是多分类头模式:
            dict: {
                'losses': {'softmax': loss, 'arcface': loss, 'cosface': loss, 'ldam': loss},
                'accuracies': {'softmax': acc, 'arcface': acc, 'cosface': acc, 'ldam': acc},
                'all_preds': {'softmax': preds, 'arcface': preds, 'cosface': preds, 'ldam': preds},
                'all_labels': labels
            }
        如果是单分类头模式:
            (epoch_loss, epoch_acc, all_preds, all_labels)
    """
    model.eval()
    
    # 仅在CUDA设备上使用AMP
    use_autocast = use_amp and device.type == 'cuda'
    
    if is_multi_head:
        # 多分类头模式
        running_losses = {'softmax': 0.0, 'arcface': 0.0, 'cosface': 0.0, 'ldam': 0.0}
        corrects = {'softmax': 0, 'arcface': 0, 'cosface': 0, 'ldam': 0}
        total = 0
        all_preds = {'softmax': [], 'arcface': [], 'cosface': [], 'ldam': []}
        all_labels = []
        
        # 创建 Focal Loss 用于 Softmax, ArcFace, CosFace
        focal_loss = FocalLoss(gamma=2.0)
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='[Val]'):
                images = images.to(device)
                labels = labels.to(device)
                
                if use_autocast:
                    with torch.amp.autocast('cuda'):
                        # 验证时也需要 labels（用于 ArcFace 和 CosFace）
                        outputs = model(images, labels=labels)
                else:
                    outputs = model(images, labels=labels)
                
                # 计算每个分类头的 loss 和 accuracy
                batch_size = labels.size(0)
                total += batch_size
                
                for head_name in ['softmax', 'arcface', 'cosface', 'ldam']:
                    logits = outputs[f'logits_{head_name}']
                    
                    # 计算 loss
                    if head_name == 'ldam':
                        loss = ldam_cb_loss(logits, labels)
                    else:
                        loss = focal_loss(logits, labels)
                    
                    running_losses[head_name] += loss.item()
                    
                    # 计算 accuracy
                    _, predicted = torch.max(logits.data, 1)
                    corrects[head_name] += (predicted == labels).sum().item()
                    all_preds[head_name].extend(predicted.cpu().numpy())
                
                all_labels.extend(labels.cpu().numpy())
        
        # 计算平均 loss 和 accuracy
        losses = {k: v / len(dataloader) for k, v in running_losses.items()}
        accuracies = {k: 100 * corrects[k] / total if total > 0 else 0.0 for k in corrects.keys()}
        
        return {
            'losses': losses,
            'accuracies': accuracies,
            'all_preds': all_preds,
            'all_labels': all_labels
        }
    else:
        # 单分类头模式（原有逻辑）
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='[Val]'):
                images = images.to(device)
                labels = labels.to(device)
                
                if use_autocast:
                    with torch.amp.autocast('cuda'):
                        outputs = extract_logits(model(images))
                        loss = criterion(outputs, labels)
                else:
                    outputs = extract_logits(model(images))
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc, all_preds, all_labels


def train(args):
    """主训练函数"""
    # 设置随机种子（用于可复现性）
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
        # 如果指定了设备，直接使用
        device = torch.device(args.device)
        if 'cuda' in args.device and not torch.cuda.is_available():
            raise RuntimeError(f"指定的设备 {args.device} 不可用，CUDA未安装或不可用")
        if 'cuda' in args.device:
            # 检查指定的GPU ID是否有效
            gpu_id = int(args.device.split(':')[1]) if ':' in args.device else 0
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"指定的GPU {gpu_id} 不存在，系统只有 {torch.cuda.device_count()} 个GPU")
    elif args.cpu:
        # 强制使用CPU
        device = torch.device('cpu')
    else:
        # 自动检测：优先使用CUDA
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
    
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(device)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 数据增强
    train_transform, val_transform = get_data_augmentation(
        augmentation_type=args.augmentation,
        img_size=args.img_size
    )
    
    # 加载数据集
    train_dataset = ImageFolderDataset(args.train_dir, transform=train_transform)
    val_dataset = ImageFolderDataset(args.val_dir, transform=val_transform)
    
    num_classes = len(train_dataset.class_to_idx)
    print(f"类别数: {num_classes}")
    cls_num_list = [train_dataset.class_counts.get(idx, 0) for idx in range(num_classes)]
    
    # 数据加载器
    use_cuda = device.type == 'cuda'
    if not use_cuda:
        num_workers = 0  # CPU模式下使用单进程
    elif sys.platform == 'win32':
        num_workers = 0  # Windows上默认单进程，避免多进程问题
        print("提示: Windows系统，使用单进程数据加载（num_workers=0）")
    else:
        num_workers = args.num_workers
    
    # 构建DataLoader参数
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': num_workers,
        'pin_memory': use_cuda,  # 只在GPU模式下使用pin_memory
    }
    
    # persistent_workers只在PyTorch 1.7+且num_workers>0时可用
    try:
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True
    except:
        pass  # 如果参数不支持，忽略
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    # 创建模型
    print(f"创建模型: {args.model}")
    
    # 检查是否是多分类头模式
    is_multi_head = 'multi_head' in args.model.lower()
    
    # 准备模型创建参数
    model_kwargs = {}
    if is_multi_head:
        model_kwargs['cls_num_list'] = cls_num_list
    # StarNet CF 模型也需要 cls_num_list 用于长尾重标定
    elif 'starnet_cf' in args.model.lower():
        model_kwargs['cls_num_list'] = cls_num_list
    
    model = create_model(
        args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        **model_kwargs
    )
    model = model.to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    if is_multi_head:
        print("✓ 多分类头模式已启用 (ArcFace, CosFace, LDAM, Softmax)")
    
    # 损失函数
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
    
    criterion = get_loss_function(args.loss, num_classes=num_classes, **loss_kwargs)
    criterion = criterion.to(device)
    
    # 多分类头模式：创建 LDAM+CB Loss
    ldam_cb_loss = None
    if is_multi_head:
        if cls_num_list is None:
            cls_num_list = [1.0] * num_classes  # 默认均匀分布
        ldam_cb_loss = LDAM_CB_Loss(cls_num_list, max_m=0.5, s=30.0, beta=0.9999)
        ldam_cb_loss = ldam_cb_loss.to(device)
        print(f"✓ LDAM+CB Loss 已创建 (类别数量列表: {cls_num_list})")
    
    # Loss 权重（多分类头模式）
    loss_weights = {'softmax': 1.0, 'arcface': 1.0, 'cosface': 1.0, 'ldam': 1.0}
    
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
    else:
        raise ValueError(f"未知的优化器: {args.optimizer}")
    
    # 学习率调度器
    # 计算每个epoch的步数（用于OneCycleLR）
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    if args.scheduler == 'onecycle':
        # OneCycleLR: 包含warmup和cycle策略
        max_lr = args.lr
        if args.onecycle_max_lr is not None:
            max_lr = args.onecycle_max_lr
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.warmup_ratio,  # warmup占总步数的比例
            anneal_strategy=args.onecycle_anneal_strategy,  # 'cos' or 'linear'
            div_factor=args.onecycle_div_factor,  # 初始学习率 = max_lr / div_factor
            final_div_factor=args.onecycle_final_div_factor  # 最终学习率 = max_lr / (div_factor * final_div_factor)
        )
        use_onecycle = True
        print(f"使用OneCycleLR调度器:")
        print(f"  最大学习率: {max_lr}")
        print(f"  总步数: {total_steps}")
        print(f"  Warmup比例: {args.warmup_ratio * 100:.1f}%")
        print(f"  初始学习率: {max_lr / args.onecycle_div_factor:.6f}")
        print(f"  最终学习率: {max_lr / (args.onecycle_div_factor * args.onecycle_final_div_factor):.6f}")
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        use_onecycle = False
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        use_onecycle = False
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        use_onecycle = False
    elif args.scheduler == 'cosine_warmup':
        # CosineAnnealingLR with warmup
        from torch.optim.lr_scheduler import LambdaLR
        warmup_epochs = int(args.epochs * args.warmup_ratio)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Warmup: 线性增长
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        use_onecycle = False
        print(f"使用CosineAnnealingLR with Warmup:")
        print(f"  Warmup轮数: {warmup_epochs}")
    else:
        scheduler = None
        use_onecycle = False
    
    # 训练历史
    if is_multi_head:
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_losses': {'softmax': [], 'arcface': [], 'cosface': [], 'ldam': []},
            'val_accs': {'softmax': [], 'arcface': [], 'cosface': [], 'ldam': []}
        }
        best_val_accs = {'softmax': 0.0, 'arcface': 0.0, 'cosface': 0.0, 'ldam': 0.0}
        best_epochs = {'softmax': 0, 'arcface': 0, 'cosface': 0, 'ldam': 0}
    else:
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        best_val_acc = 0.0
        best_epoch = 0
        best_val_loss = float('inf')
    
    # Early Stopping相关
    patience = args.early_stopping_patience if args.early_stopping_patience > 0 else None
    patience_counter = 0
    min_delta = args.early_stopping_min_delta
    
    print("\n开始训练...")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"批次大小: {args.batch_size}")
    print(f"数据加载线程数: {num_workers}")
    print(f"学习率: {args.lr}")
    print(f"优化器: {args.optimizer}")
    print(f"损失函数: {args.loss}")
    print(f"数据增强: {args.augmentation}")
    if args.early_stopping_patience > 0:
        print(f"Early Stopping: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")
    print("-" * 50)
    
    # 测试数据加载
    print("测试数据加载...")
    try:
        test_batch = next(iter(train_loader))
        print(f"✓ 数据加载正常，批次形状: {test_batch[0].shape}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    print("-" * 50)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        if args.use_drw and args.loss == 'ldam' and drw_weights is not None and not drw_applied:
            if epoch >= args.drw_start_epoch and len(val_acc_recent) == args.drw_window:
                diff = max(val_acc_recent) - min(val_acc_recent)
                if diff <= args.drw_threshold * 100:
                    criterion.weight = drw_weights
                    drw_applied = True
                    print(f"DRW 权重在 epoch {epoch} 启用 (差值 {diff:.2f}%)")
        
        if is_multi_head:
            # 多分类头模式
            train_loss, train_acc, train_accs = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch,
                scheduler if use_onecycle else None,
                use_amp=args.use_amp,
                is_multi_head=True,
                ldam_cb_loss=ldam_cb_loss,
                loss_weights=loss_weights
            )
            
            # 验证
            val_results = validate(
                model, val_loader, criterion, device,
                use_amp=args.use_amp,
                is_multi_head=True,
                ldam_cb_loss=ldam_cb_loss
            )
            
            val_losses = val_results['losses']
            val_accs = val_results['accuracies']
            
            # 更新学习率（OneCycleLR在训练循环中每个batch后更新，其他调度器在每个epoch后更新）
            if scheduler and not use_onecycle:
                # 使用平均 loss 或最佳 accuracy
                avg_val_loss = sum(val_losses.values()) / len(val_losses)
                if args.scheduler == 'plateau':
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            for head_name in ['softmax', 'arcface', 'cosface', 'ldam']:
                history['val_losses'][head_name].append(val_losses[head_name])
                history['val_accs'][head_name].append(val_accs[head_name])
            
            # 保存最佳模型（为每个分类头保存）
            for head_name in ['softmax', 'arcface', 'cosface', 'ldam']:
                if val_accs[head_name] > best_val_accs[head_name] + min_delta:
                    best_val_accs[head_name] = val_accs[head_name]
                    best_epochs[head_name] = epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_accs[head_name],
                        'val_loss': val_losses[head_name],
                        'class_to_idx': train_dataset.class_to_idx,
                        'model_name': args.model,
                        'head_name': head_name
                    }, os.path.join(args.output_dir, f'best_model_{head_name}.pth'))
            
            # Early Stopping检查（基于最佳分类头）
            best_current_acc = max(val_accs.values())
            if epoch == 1:
                best_overall_acc = best_current_acc
                patience_counter = 0
            else:
                if best_current_acc > best_overall_acc + min_delta:
                    best_overall_acc = best_current_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience is not None and patience_counter >= patience:
                print(f"\nEarly Stopping触发!")
                print(f"验证准确率在 {patience} 个epoch内没有提升")
                print(f"最佳验证准确率: {best_overall_acc:.2f}%")
                break
            
            # 定期保存检查点
            if epoch % args.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accs': val_accs,
                    'val_losses': val_losses,
                    'class_to_idx': train_dataset.class_to_idx,
                    'model_name': args.model
                }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Losses: Softmax={val_losses['softmax']:.4f}, ArcFace={val_losses['arcface']:.4f}, "
                  f"CosFace={val_losses['cosface']:.4f}, LDAM={val_losses['ldam']:.4f}")
            print(f"  Val Accs: Softmax={val_accs['softmax']:.2f}%, ArcFace={val_accs['arcface']:.2f}%, "
                  f"CosFace={val_accs['cosface']:.2f}%, LDAM={val_accs['ldam']:.2f}%")
            if scheduler:
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
        else:
            # 单分类头模式（原有逻辑）
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch,
                scheduler if use_onecycle else None,
                use_amp=args.use_amp
            )
            
            # 验证
            val_loss, val_acc, _, _ = validate(
                model, val_loader, criterion, device,
                use_amp=args.use_amp
            )
            
            # 更新学习率（OneCycleLR在训练循环中每个batch后更新，其他调度器在每个epoch后更新）
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
            
            # 保存最佳模型（基于验证准确率）
            improved = False
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                best_epoch = epoch
                best_val_loss = val_loss
                improved = True
                patience_counter = 0  # 重置patience计数器
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_to_idx': train_dataset.class_to_idx,
                    'model_name': args.model
                }, os.path.join(args.output_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            
            # Early Stopping检查
            if patience is not None and patience_counter >= patience:
                print(f"\nEarly Stopping触发!")
                print(f"验证准确率在 {patience} 个epoch内没有提升")
                print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
                break
            
            # 定期保存检查点
            if epoch % args.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_to_idx': train_dataset.class_to_idx,
                    'model_name': args.model
                }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if scheduler:
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
    
    # 保存训练历史
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 保存训练配置
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n训练完成!")
    print(f"总时间: {total_time / 3600:.2f} 小时")
    
    if is_multi_head:
        # 多分类头模式：输出每个分类头的最佳结果
        print("\n" + "=" * 80)
        print("多分类头验证结果总结")
        print("=" * 80)
        print(f"{'分类头':<15} {'最佳准确率':<15} {'最佳Epoch':<15}")
        print("-" * 80)
        
        for head_name in ['softmax', 'arcface', 'cosface', 'ldam']:
            print(f"{head_name.capitalize():<15} {best_val_accs[head_name]:<15.2f}% {best_epochs[head_name]:<15}")
        
        # 找出最佳分类头
        best_head_name = max(best_val_accs.items(), key=lambda x: x[1])[0]
        best_head_acc = best_val_accs[best_head_name]
        best_head_epoch = best_epochs[best_head_name]
        
        print("-" * 80)
        print(f"🏆 最佳分类头: {best_head_name.capitalize()}")
        print(f"   准确率: {best_head_acc:.2f}%")
        print(f"   Epoch: {best_head_epoch}")
        print("=" * 80)
        print(f"\n模型保存在: {args.output_dir}")
        print(f"每个分类头的最佳模型:")
        for head_name in ['softmax', 'arcface', 'cosface', 'ldam']:
            print(f"  - {head_name}: best_model_{head_name}.pth")
    else:
        # 单分类头模式
        print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"模型保存在: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='多分类任务训练脚本')
    
    # 数据相关
    parser.add_argument('--train-dir', type=str, default='data/train',
                        help='训练数据目录')
    parser.add_argument('--val-dir', type=str, default='data/val',
                        help='验证数据目录')
    parser.add_argument('--img-size', type=int, default=224,
                        help='输入图像大小')
    
    # 模型相关
    parser.add_argument('--model', type=str, default='resnet50',
                        help='模型名称')
    parser.add_argument('--pretrained', action='store_true',
                        help='是否使用预训练权重')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
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
                        help='Warmup占总训练步数的比例（0.0-1.0），用于OneCycleLR和cosine_warmup')
    parser.add_argument('--onecycle-max-lr', type=float, default=None,
                        help='OneCycleLR的最大学习率（默认使用--lr）')
    parser.add_argument('--onecycle-div-factor', type=float, default=25.0,
                        help='OneCycleLR的div_factor，初始学习率 = max_lr / div_factor')
    parser.add_argument('--onecycle-final-div-factor', type=float, default=10000.0,
                        help='OneCycleLR的final_div_factor，最终学习率 = max_lr / (div_factor * final_div_factor)')
    parser.add_argument('--onecycle-anneal-strategy', type=str, default='cos',
                        choices=['cos', 'linear'],
                        help='OneCycleLR的退火策略')
    
    # 损失函数
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'focal', 'label_smoothing', 'weighted_ce', 'asl'],
                        help='损失函数类型')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label Smoothing的平滑参数')
    parser.add_argument('--asl-gamma-neg', type=float, default=4.0,
                        help='Asymmetric Loss的负类gamma')
    parser.add_argument('--asl-gamma-pos', type=float, default=1.0,
                        help='Asymmetric Loss的正类gamma')
    parser.add_argument('--asl-clip', type=float, default=0.05,
                        help='Asymmetric Loss的clip')
    parser.add_argument('--asl-eps', type=float, default=1e-8,
                        help='Asymmetric Loss的eps')
    
    # 数据增强
    parser.add_argument('--augmentation', type=str, default='standard',
                        choices=['none', 'minimal', 'standard', 'strong', 'medical'],
                        help='数据增强类型')
    
    # 其他
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='输出目录')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='保存检查点的间隔（epoch）')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')
    parser.add_argument('--device', type=str, default=None,
                        help='指定设备 (例如: cuda, cuda:0, cuda:1, cpu). 如果未指定，将自动检测')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（用于可复现性）')
    
    # Early Stopping
    parser.add_argument('--early-stopping-patience', type=int, default=0,
                        help='Early Stopping的patience（0表示不使用Early Stopping，建议10-20）')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                        help='Early Stopping的最小改善阈值（验证准确率需要提升至少这个值才算改善，建议0.1-0.5）')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()

