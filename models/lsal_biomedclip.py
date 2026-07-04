"""
LSAL (LLM-Semantic Adaptive Loss) for BiomedCLIP
实现LLM语义感知自适应损失函数

核心创新：
1. LLM-Guided Soft Target Cross Entropy: 使用软标签替代硬编码的One-hot标签
2. Semantic Anchor Loss: 强制图像特征向LLM定义的"语义中心"靠拢
"""

import copy
import os.path as osp
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import os
import sys
from pathlib import Path

# 添加open_clip路径
script_dir = Path(__file__).parent
open_clip_base_path = script_dir.parent / 'open_clip'
if open_clip_base_path.exists():
    sys.path.insert(0, str(open_clip_base_path.parent))
    print(f"Using open_clip path: {open_clip_base_path}")

# dassl相关导入（可选，仅用于Trainer）
try:
    from dassl.engine import TRAINER_REGISTRY, TrainerX
    from dassl.utils import load_pretrained_weights, load_checkpoint
    from dassl.optim import build_optimizer, build_lr_scheduler
    from dassl.metrics import compute_accuracy
    DASSL_AVAILABLE = True
except ImportError:
    DASSL_AVAILABLE = False
    # 如果dassl不可用，定义占位符（仅用于类型提示）
    TRAINER_REGISTRY = None
    TrainerX = None
    def compute_accuracy(*args, **kwargs):
        return (torch.tensor(0.0),)

try:
    from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
except ImportError:
    # 如果open_clip不可用，只影响Trainer，不影响LLMSemanticSuperLoss
    create_model_from_pretrained = None
    get_tokenizer = None
    if DASSL_AVAILABLE:
        # 只有在使用Trainer时才报错
        print("Error: open_clip not found. Please install or ensure open_clip is in the path.")
        raise

# 设置环境变量
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
os.environ.setdefault('HF_HUB_DOWNLOAD_MAX_RETRIES', '10')


class LLMSemanticSuperLoss(nn.Module):
    """
    LLM语义感知自适应损失函数
    
    包含三个部分：
    1. LLM-Guided Soft Target Cross Entropy (L_LSA-CE) - 分类损失
    2. Semantic Anchor Loss (L_Anchor) - 语义锚点损失
    3. Contrastive Loss (L_Contrastive) - 对比损失（图像 vs 类别中心）
    """
    
    def __init__(self, soft_labels_matrix, class_centers, lambda_anchor=0.5, lambda_contrastive=1.0, temperature=0.07):
        """
        Args:
            soft_labels_matrix: [N_classes, N_classes] 软标签矩阵，从LLM语义相似度计算得到
            class_centers: [N_classes, Dim] 每个类别的LLM语义中心
            lambda_anchor: Semantic Anchor Loss的权重系数
            lambda_contrastive: Contrastive Loss的权重系数
            temperature: 对比损失的温度参数
        """
        super().__init__()
        # 注册为buffer，不参与梯度更新
        self.register_buffer('soft_labels_matrix', soft_labels_matrix)
        self.register_buffer('class_centers', class_centers)
        self.lambda_anchor = lambda_anchor
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature
    
    def forward(self, image_features, logits, labels):
        """
        计算LSAL损失
        
        Args:
            image_features: [Batch, Dim] 归一化后的图像特征
            logits: [Batch, N_classes] 分类logits（通常是 image @ text_classifier）
            labels: [Batch] GT类别索引
        
        Returns:
            total_loss: 总损失
            loss_cls: 分类损失（软标签交叉熵）
            loss_anchor: 语义锚点损失
            loss_contrastive: 对比损失
        """
        # ========== 创新点 1: LLM-Guided Soft Target Cross Entropy ==========
        # 获取当前batch对应的软标签
        # 例如：标签是 0 (Pneumonia)，我们不取 [1, 0, 0...]
        # 而是取预计算矩阵的第 0 行: [0.8, 0.15, 0.05...]
        target_probs = self.soft_labels_matrix[labels]  # [Batch, N_classes]
        
        # 计算 Soft Cross Entropy
        # 公式: - sum(target * log_softmax(input))
        log_probs = F.log_softmax(logits, dim=1)
        loss_cls = -(target_probs * log_probs).sum(dim=1).mean()
        
        # ========== 创新点 2: Semantic Anchor Loss ==========
        # 强制每张图像的特征，去接近它所属类别的LLM文本语义中心
        # 这比单纯的分类更强，因为它规定了特征在空间中的绝对位置
        target_centers = self.class_centers[labels]  # [Batch, Dim]
        
        # 使用 MSE 或 Cosine Distance
        # 由于特征已归一化，MSE和Cosine是等价的，MSE更快
        loss_anchor = F.mse_loss(image_features, target_centers)
        
        # ========== 创新点 3: Contrastive Loss ==========
        # 对比损失：图像应该与对应类别中心最相似，与其他类别中心不相似
        # 计算图像与所有类别中心的相似度
        batch_size = image_features.shape[0]
        n_classes = self.class_centers.shape[0]
        
        # 计算图像特征与所有类别中心的相似度矩阵
        # image_features: [Batch, Dim], class_centers: [N_classes, Dim]
        # logits_per_image: [Batch, N_classes]
        logits_per_image = image_features @ self.class_centers.t() / self.temperature
        
        # 创建配对标签（每个图像对应其真实类别）
        contrastive_labels = labels  # [Batch]
        
        # 对比损失：使用交叉熵，图像应该与对应类别中心最相似
        loss_contrastive = F.cross_entropy(logits_per_image, contrastive_labels)
        
        # 总损失
        total_loss = loss_cls + (self.lambda_anchor * loss_anchor) + (self.lambda_contrastive * loss_contrastive)
        
        return total_loss, loss_cls, loss_anchor, loss_contrastive


class CustomCLIP_LSAL(nn.Module):
    """
    简化的CLIP模型，用于LSAL训练
    只包含图像编码器和固定的文本特征（类别中心）
    """
    
    def __init__(self, cfg, classnames, biomedclip_model, class_centers):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.n_cls = len(classnames)
        
        # 图像编码器（可训练）
        self.image_encoder = biomedclip_model.visual
        
        # 类别中心（固定，不训练）
        self.register_buffer('class_centers', class_centers)
        
        # Logit scale（从原始模型获取）
        if hasattr(biomedclip_model, 'logit_scale'):
            if isinstance(biomedclip_model.logit_scale, nn.Parameter):
                self.logit_scale = biomedclip_model.logit_scale
            else:
                self.logit_scale = nn.Parameter(torch.tensor(biomedclip_model.logit_scale))
        else:
            # 默认logit scale
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, image, label=None):
        """
        前向传播
        
        Args:
            image: [Batch, 3, H, W] 图像
            label: [Batch] 标签（训练时需要）
        
        Returns:
            训练模式: (image_features, logits, labels)
            评估模式: logits
        """
        device = image.device
        
        # 编码图像
        image_features = self.image_encoder(image)
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 计算logits（使用类别中心作为分类器权重）
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.class_centers.t()
        
        if self.training and label is not None:
            return image_features, logits, label
        else:
            return logits


# 只有在dassl可用时才定义Trainer类
if DASSL_AVAILABLE and TrainerX is not None:
    @TRAINER_REGISTRY.register()
    class LSAL_BiomedCLIP(TrainerX):
        """
        LSAL (LLM-Semantic Adaptive Loss) Trainer for BiomedCLIP
        
        特点：
        1. 使用LLM生成的软标签矩阵替代硬编码的One-hot标签
        2. 添加语义锚点损失，强制图像特征向LLM语义中心对齐
        3. 只训练图像编码器，文本编码器冻结
        4. 极简架构，无Prompt Learning
        """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.LSAL.PREC in ["fp16", "fp32", "amp"]
        assert hasattr(cfg.TRAINER.LSAL, 'SEMANTICS_DIR') or hasattr(cfg.TRAINER.LSAL, 'SEMANTICS_FILE')
        assert hasattr(cfg.TRAINER.LSAL, 'LAMBDA_ANCHOR')
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        print("="*80)
        print("Building LSAL_BiomedCLIP Model")
        print("="*80)
        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        print(f"注意：如果网络连接有问题，将自动使用本地缓存的模型")
        
        try:
            biomedclip_model, preprocess = create_model_from_pretrained(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            print("✓ 成功加载 BiomedCLIP 模型")
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            print("提示：如果网络连接有问题，请确保模型已下载到本地缓存")
            raise
        
        if cfg.TRAINER.LSAL.PREC == "fp32" or cfg.TRAINER.LSAL.PREC == "amp":
            biomedclip_model.float()
        
        # ========== 加载LLM语义矩阵和类别中心 ==========
        print("\nLoading LLM Semantics...")
        semantics_dir = getattr(cfg.TRAINER.LSAL, 'SEMANTICS_DIR', None)
        semantics_file = getattr(cfg.TRAINER.LSAL, 'SEMANTICS_FILE', None)
        
        if semantics_file:
            # 如果指定了单个文件，尝试从文件加载
            semantics_path = Path(semantics_file)
            if semantics_path.is_dir():
                semantics_dir = semantics_path
            else:
                # 假设是包含语义文件的目录
                semantics_dir = semantics_path.parent
        
        if semantics_dir is None:
            # 尝试默认路径
            default_path = Path(__file__).parent.parent / 'semantics'
            if default_path.exists():
                semantics_dir = default_path
            else:
                raise ValueError(
                    "Cannot find semantics directory. Please specify SEMANTICS_DIR or SEMANTICS_FILE in cfg.TRAINER.LSAL"
                )
        
        semantics_dir = Path(semantics_dir)
        centers_path = semantics_dir / 'class_centers.pt'
        matrix_path = semantics_dir / 'soft_labels_matrix.pt'
        
        if not centers_path.exists() or not matrix_path.exists():
            raise FileNotFoundError(
                f"Semantics files not found in {semantics_dir}. "
                f"Please run build_llm_semantics.py first to generate them."
            )
        
        class_centers = torch.load(centers_path, map_location='cpu')
        soft_labels_matrix = torch.load(matrix_path, map_location='cpu')
        
        print(f"✓ Loaded class centers from: {centers_path}")
        print(f"  Shape: {class_centers.shape}")
        print(f"✓ Loaded soft labels matrix from: {matrix_path}")
        print(f"  Shape: {soft_labels_matrix.shape}")
        
        # 验证类别数量是否匹配
        if class_centers.shape[0] != len(classnames):
            print(f"Warning: Number of classes mismatch!")
            print(f"  Classnames: {len(classnames)}")
            print(f"  Semantics: {class_centers.shape[0]}")
            print(f"  Attempting to match by order...")
            # 如果数量不匹配，尝试截断或扩展（按顺序）
            if class_centers.shape[0] > len(classnames):
                class_centers = class_centers[:len(classnames)]
                soft_labels_matrix = soft_labels_matrix[:len(classnames), :len(classnames)]
                print(f"  Truncated to {len(classnames)} classes")
            else:
                raise ValueError(
                    f"Semantics has fewer classes ({class_centers.shape[0]}) than dataset ({len(classnames)}). "
                    f"Please regenerate semantics with correct classnames."
                )
        
        # ========== 构建模型 ==========
        print("\nBuilding CustomCLIP_LSAL model...")
        self.model = CustomCLIP_LSAL(cfg, classnames, biomedclip_model.eval(), class_centers)
        
        # ========== 设置可训练参数 ==========
        print("\nSetting up trainable parameters...")
        print("Turning off gradients in the text encoder (using fixed class centers)")
        print("Keeping gradients for: image_encoder (visual) only")
        
        names_to_update = []
        for name, param in self.model.named_parameters():
            if name.startswith("image_encoder."):
                names_to_update.append(name)
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        
        # 统计参数
        enabled = set()
        total_params = 0
        image_encoder_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                param_count = param.numel()
                total_params += param_count
                if 'image_encoder' in name:
                    image_encoder_params += param_count
        
        print(f"Parameters to be updated: {total_params:,} parameters ({len(enabled)} parameter groups)")
        print(f"  - Image Encoder: {image_encoder_params:,} parameters")
        
        # ========== 初始化损失函数 ==========
        lambda_anchor = getattr(cfg.TRAINER.LSAL, 'LAMBDA_ANCHOR', 0.5)
        print(f"\nInitializing LLMSemanticSuperLoss with lambda_anchor={lambda_anchor}")
        self.criterion = LLMSemanticSuperLoss(
            soft_labels_matrix=soft_labels_matrix,
            class_centers=class_centers,
            lambda_anchor=lambda_anchor
        )
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        # ========== 优化器和学习率调度器 ==========
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        # 注册模型
        self.register_model("model", self.model, self.optim, self.sched)
        
        # 混合精度训练
        self.scaler = GradScaler() if cfg.TRAINER.LSAL.PREC == "amp" else None
        
        print("\n" + "="*80)
        print("Model setup complete!")
        print("="*80)
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.LSAL.PREC
        
        if prec == "amp":
            with autocast():
                image_features, logits, labels = model(image, label)
                # 计算LSAL损失
                loss, loss_cls, loss_anchor = self.criterion(image_features, logits, labels)
            
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            image_features, logits, labels = model(image, label)
            # 计算LSAL损失
            loss, loss_cls, loss_anchor = self.criterion(image_features, logits, labels)
            self.model_backward_and_update(loss)
        
        loss_summary = {
            "loss": loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_anchor": loss_anchor.item(),
            "acc": compute_accuracy(logits, label)[0].item() if DASSL_AVAILABLE else 0.0,
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

