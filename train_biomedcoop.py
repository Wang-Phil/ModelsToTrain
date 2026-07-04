"""
BiomedCoOp 模型专用训练脚本
从 train_clip.py 中抽离，专门用于训练 BiomedCoOp 模型
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split

# 直接导入 CustomCLIP（不通过适配器）
try:
    from models.biomedcoop_biomedclip import CustomCLIP as BiomedCLIPCustomCLIP
    from open_clip.src.open_clip import create_model_from_pretrained
    BIOMEDCOOP_AVAILABLE = True
except ImportError as e:
    BIOMEDCOOP_AVAILABLE = False
    print(f"Warning: Cannot import BiomedCoOp components: {e}")

# 导入其他模型的 CustomCLIP
try:
    # 先导入系统安装的 CLIP 库（不是本地的 clip 目录）
    import sys
    # 如果本地 clip 模块已导入，先移除
    if 'clip' in sys.modules and hasattr(sys.modules['clip'], '__file__'):
        clip_file = sys.modules['clip'].__file__
        if clip_file and 'ModelsTotrain/clip' in clip_file:
            del sys.modules['clip']
    # 导入系统安装的 CLIP 库
    import clip
    from models.coop_clip import CustomCLIP as CLIPCustomCLIP
    from models.coop_clip import load_clip_to_cpu
    CLIP_AVAILABLE = True
except ImportError as e:
    CLIP_AVAILABLE = False
    print(f"Warning: Cannot import CLIP components: {e}")

try:
    # 优先使用 biomedcoop_pmcclip（支持 KDSP, TAU, N_PROMPTS）
    from models.biomedcoop_pmcclip import CustomCLIP as PMCCLIPCustomCLIP
    from models.coop_pmcclip import PMCCLIP
    PMCCLIP_AVAILABLE = True
    # 验证导入的类
    if 'biomedcoop_pmcclip' in PMCCLIPCustomCLIP.__module__:
        print("✓ 成功导入 biomedcoop_pmcclip (支持 KDSP, TAU, N_PROMPTS)")
        print(f"  模块: {PMCCLIPCustomCLIP.__module__}")
    else:
        raise ImportError(f"导入的类来自错误的模块: {PMCCLIPCustomCLIP.__module__}")
except (ImportError, Exception) as e:
    # 如果失败，尝试使用 coop_pmcclip（不支持这些参数）
    print(f"⚠️  导入 biomedcoop_pmcclip 失败: {e}")
    import traceback
    traceback.print_exc()
    print("   尝试使用 coop_pmcclip 作为备选...")
    try:
        from models.coop_pmcclip import CustomCLIP as PMCCLIPCustomCLIP
        from models.coop_pmcclip import PMCCLIP
        PMCCLIP_AVAILABLE = True
        print("⚠️  Warning: Using coop_pmcclip instead of biomedcoop_pmcclip. Some parameters (kdsp-lambda, tau, n-prompts) may not be used.")
    except ImportError as e2:
        PMCCLIP_AVAILABLE = False
        print(f"✗  Error: Cannot import PMC-CLIP components: {e2}")

try:
    from models.coop_pubmedclip import CustomCLIP as PubMedCLIPCustomCLIP
    from models.coop_pubmedclip import load_clip_to_cpu as load_pubmedclip_to_cpu
    # 确保使用系统安装的 clip 包
    import sys
    if 'clip' not in sys.modules:
        import clip
    PUBMEDCLIP_AVAILABLE = True
except ImportError as e:
    PUBMEDCLIP_AVAILABLE = False
    print(f"Warning: Cannot import PubMedCLIP components: {e}")

try:
    from models.hybrid_pmcclip_biomedclip import HybridCLIP as HybridCLIPCustomCLIP
    from open_clip.src.open_clip import create_model_from_pretrained
    HYBRID_AVAILABLE = True
except ImportError as e:
    HYBRID_AVAILABLE = False
    print(f"Warning: Cannot import Hybrid PMC-CLIP + BiomedCLIP components: {e}")

try:
    from models.hybrid_pmcclip_biomedclip_coop import HybridCLIPWithCoOp as HybridCLIPWithCoOpCustomCLIP
    from open_clip.src.open_clip import create_model_from_pretrained
    HYBRID_COOP_AVAILABLE = True
except ImportError as e:
    HYBRID_COOP_AVAILABLE = False
    print(f"Warning: Cannot import Hybrid PMC-CLIP + BiomedCLIP + CoOp components: {e}")

try:
    from models.hybrid_pmcclip_only import HybridPMCCLIPOnly, load_pmcclip_model
    PMCCLIP_FULL_AVAILABLE = True
except ImportError as e:
    PMCCLIP_FULL_AVAILABLE = False
    print(f"Warning: Cannot import PMC-CLIP Full components: {e}")

try:
    from models.hybrid_pmcclip_biomedclip_coop_sccm import HybridCLIPWithCoOpSCCM
    from open_clip.src.open_clip import create_model_from_pretrained
    HYBRID_COOP_SCCM_AVAILABLE = True
except ImportError as e:
    HYBRID_COOP_SCCM_AVAILABLE = False
    print(f"Warning: Cannot import Hybrid PMC-CLIP + BiomedCLIP + CoOp + SCCM components: {e}")

# 尝试导入 yacs，如果没有则使用简单的配置类
try:
    from yacs.config import CfgNode
    HAS_YACS = True
except ImportError:
    HAS_YACS = False
    # 创建一个简单的配置类
    class CfgNode(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self
        
        def __getattr__(self, name):
            if name in self:
                return self[name]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        def __setattr__(self, name, value):
            self[name] = value

# 导入数据增强函数（从 train_clip.py）
try:
    from train_clip import get_data_augmentation, CLIPDataset, CLIPSubset
except ImportError:
    print("Error: Cannot import from train_clip.py. Please ensure train_clip.py exists.")
    sys.exit(1)

# 导入评估指标
try:
    from calculate_metrics import calculate_classification_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: calculate_metrics not available. mAP calculation will be skipped.")


def create_biomedcoop_config(epochs=100, n_ctx=4, ctx_init="a photo of a", csc=False,
                            class_token_position="end", sccm_lambda=1.0,
                            kdsp_lambda=1.0, tau=1.0, n_prompts=4,
                            use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0,
                            class_texts_file=None, use_amp=True,
                            classification_loss_weight=0.5, contrastive_loss_weight=0.5,
                            distillation_loss_weight=0.0, use_original_clip_resnet50=False):
    """
    创建 BiomedCoOp 配置对象
    
    Args:
        epochs: 训练轮数
        n_ctx: 上下文token数量
        ctx_init: 上下文初始化文本
        csc: 是否使用类别特定的上下文
        class_token_position: 类别token位置
        sccm_lambda: SCCM损失权重
        kdsp_lambda: KDSP损失权重（默认1.0）
        tau: 用于选择prompt的阈值（默认1.0）
        n_prompts: 使用的prompt数量（默认4）
        use_focal_loss: 是否使用 Focal Loss
        focal_alpha: Focal Loss alpha 参数
        focal_gamma: Focal Loss gamma 参数
        class_texts_file: 类别文本描述JSON文件路径
        use_amp: 是否使用混合精度训练
        classification_loss_weight: 分类损失权重
        contrastive_loss_weight: 对比损失权重
        distillation_loss_weight: 蒸馏损失权重（默认0.0，不使用蒸馏）
    
    Returns:
        cfg: CfgNode 配置对象
    """
    cfg = CfgNode()
    
    # 基本配置
    cfg.INPUT = CfgNode()
    cfg.INPUT.SIZE = [224, 224]
    
    cfg.OPTIM = CfgNode()
    cfg.OPTIM.MAX_EPOCH = epochs
    
    cfg.MODEL = CfgNode()
    cfg.MODEL.INIT_WEIGHTS = None
    
    # BiomedCoOp 配置（虽然当前 CustomCLIP 不使用这些，但保留以兼容接口）
    cfg.TRAINER = CfgNode()
    # BiomedCoOp 配置
    cfg.TRAINER.BIOMEDCOOP = CfgNode()
    cfg.TRAINER.BIOMEDCOOP.N_CTX = n_ctx
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = ctx_init
    cfg.TRAINER.BIOMEDCOOP.CSC = csc
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = class_token_position
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = sccm_lambda
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = kdsp_lambda
    cfg.TRAINER.BIOMEDCOOP.TAU = tau
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = n_prompts
    cfg.TRAINER.BIOMEDCOOP.USE_FOCAL_LOSS = use_focal_loss
    cfg.TRAINER.BIOMEDCOOP.FOCAL_ALPHA = focal_alpha
    cfg.TRAINER.BIOMEDCOOP.FOCAL_GAMMA = focal_gamma
    cfg.TRAINER.BIOMEDCOOP.CLASS_TEXTS_FILE = class_texts_file
    cfg.TRAINER.BIOMEDCOOP.PREC = "amp" if use_amp else "fp32"  # 精度设置
    cfg.TRAINER.BIOMEDCOOP.CLASSIFICATION_LOSS_WEIGHT = classification_loss_weight
    cfg.TRAINER.BIOMEDCOOP.CONTRASTIVE_LOSS_WEIGHT = contrastive_loss_weight
    cfg.TRAINER.BIOMEDCOOP.DISTILLATION_LOSS_WEIGHT = distillation_loss_weight
    cfg.TRAINER.BIOMEDCOOP.USE_ORIGINAL_CLIP_RESNET50 = use_original_clip_resnet50
    
    # CoOp 配置（用于 CLIP、PMC-CLIP、PubMedCLIP）
    cfg.TRAINER.COOP = CfgNode()
    cfg.TRAINER.COOP.N_CTX = n_ctx
    cfg.TRAINER.COOP.CTX_INIT = ctx_init
    cfg.TRAINER.COOP.CSC = csc
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = class_token_position
    cfg.TRAINER.COOP.CLASS_TEXTS_FILE = class_texts_file
    cfg.TRAINER.COOP.PREC = "amp" if use_amp else "fp32"
    
    return cfg


def create_folds_from_dataset(dataset, n_splits=5, shuffle=True, random_state=42):
    """
    从数据集创建K折交叉验证的folds
    
    Args:
        dataset: CLIPDataset 实例
        n_splits: 折数
        shuffle: 是否打乱
        random_state: 随机种子
    
    Returns:
        folds: [(train_indices, val_indices), ...] 列表
    """
    labels = [label for _, label, _ in dataset.samples]
    
    # 处理 n_splits=1 的情况（用于测试）
    if n_splits == 1:
        train_idx, val_idx = train_test_split(
            range(len(dataset)), 
            test_size=0.2, 
            shuffle=shuffle, 
            random_state=random_state,
            stratify=labels
        )
        return [(train_idx, val_idx)]
    
    # 使用分层K折（StratifiedKFold）确保每折中类别分布相同
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []
    for train_idx, val_idx in skf.split(range(len(dataset)), labels):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    
    return folds


def train_epoch(model, dataloader, optimizer, device, epoch, use_amp=True, scaler=None):
    """
    训练一个epoch
    
    Args:
        model: CustomCLIP 模型（直接使用，不通过适配器）
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        use_amp: 是否使用混合精度
        scaler: GradScaler（如果使用AMP）
    
    Returns:
        epoch_loss: 平均损失
        epoch_acc: 准确率
        loss_ce: 分类损失
        contrastive_loss: 对比损失（或 loss_sccm）
        loss_distill: 蒸馏损失（或 loss_kdsp）
    """
    model.train()
    running_loss = 0.0
    running_loss_ce = 0.0
    running_contrastive_loss = 0.0
    running_loss_distill = 0.0
    running_loss_sccm = 0.0
    running_loss_kdsp = 0.0
    correct = 0
    total = 0
    
    # 检测模型类型：biomedcoop_pmcclip 返回 (logits, loss_ce, loss_sccm, loss_kdsp)
    # 其他模型返回 (logits, loss_ce, contrastive_loss, loss_distill)
    # 处理 DataParallel 包装的情况
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 检测模型类型
    is_biomedcoop_pmcclip = False
    is_hybrid_coop_sccm = False
    if hasattr(actual_model, '__class__'):
        model_class_name = actual_model.__class__.__name__
        model_module = actual_model.__class__.__module__
        # 检查是否是 biomedcoop_pmcclip 中的 CustomCLIP
        if 'biomedcoop_pmcclip' in model_module and model_class_name == 'CustomCLIP':
            is_biomedcoop_pmcclip = True
        # 或者通过检查关键属性
        elif (hasattr(actual_model, 'prompt_learner') and 
              hasattr(actual_model.prompt_learner, 'fixed_embeddings') and
              hasattr(actual_model, 'cfg') and
              hasattr(actual_model.cfg, 'TRAINER') and
              hasattr(actual_model.cfg.TRAINER, 'BIOMEDCOOP')):
            is_biomedcoop_pmcclip = True
        # 检查是否是 hybrid_coop_sccm
        elif 'hybrid_pmcclip_biomedclip_coop_sccm' in model_module and model_class_name == 'HybridCLIPWithCoOpSCCM':
            is_hybrid_coop_sccm = True
        elif hasattr(actual_model, 'sccm_lambda') and hasattr(actual_model, 'prompt_learner'):
            is_hybrid_coop_sccm = True
    
    # 获取损失权重（在循环外部获取，提高效率）
    if hasattr(model, 'classification_loss_weight') and hasattr(model, 'contrastive_loss_weight'):
        classification_weight = model.classification_loss_weight
        contrastive_weight = model.contrastive_loss_weight
    else:
        # 默认权重（如果模型没有这些属性，使用默认值）
        classification_weight = 0.5
        contrastive_weight = 0.5
    
    # 获取蒸馏损失权重（如果模型有这个属性）
    if hasattr(model, 'distillation_loss_weight'):
        distillation_weight = model.distillation_loss_weight
        use_distill_loss = distillation_weight > 0
    else:
        distillation_weight = 0.0
        use_distill_loss = False
    
    # 对于 biomedcoop_pmcclip，从配置中获取权重
    if is_biomedcoop_pmcclip and hasattr(model, 'cfg'):
        sccm_lambda = model.cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA
        kdsp_lambda = model.cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA
        use_sccm_loss = sccm_lambda > 0
        use_kdsp_loss = kdsp_lambda > 0
    else:
        sccm_lambda = 0.0
        kdsp_lambda = 0.0
        use_sccm_loss = False
        use_kdsp_loss = False
    
    # 根据权重决定损失计算方式（避免在循环内重复判断）
    use_ce_loss = classification_weight > 0
    use_contrastive_loss = contrastive_weight > 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                # 调用模型forward，根据模型类型处理不同的返回值
                forward_output = model(images, labels)
                
                if is_biomedcoop_pmcclip:
                    # biomedcoop_pmcclip 返回 (logits, loss_ce, loss_sccm, loss_kdsp)
                    logits, loss_ce, loss_sccm, loss_kdsp = forward_output
                    # 计算总损失
                    loss = loss_ce + loss_sccm + loss_kdsp
                    # 统计损失（用于显示）
                    contrastive_loss = torch.tensor(0.0, device=device)  # 占位符
                    loss_distill = torch.tensor(0.0, device=device)  # 占位符
                elif is_hybrid_coop_sccm:
                    # hybrid_coop_sccm 返回 (logits, loss_ce, contrastive_loss, loss_sccm, loss_distill)
                    logits, loss_ce, contrastive_loss, loss_sccm, loss_distill = forward_output
                    # 计算总损失
                    loss = (classification_weight * loss_ce +
                           contrastive_weight * contrastive_loss +
                           loss_sccm +
                           distillation_weight * loss_distill)
                    loss_kdsp = torch.tensor(0.0, device=device)  # 占位符
                else:
                    # 其他模型返回 (logits, loss_ce, contrastive_loss, loss_distill)
                    logits, loss_ce, contrastive_loss, loss_distill = forward_output
                    # 根据权重计算总损失（如果权重为0，对应的损失不会被计算）
                    loss = 0.0
                    if use_ce_loss:
                        loss += classification_weight * loss_ce
                    if use_contrastive_loss:
                        loss += contrastive_weight * contrastive_loss
                    if use_distill_loss:
                        loss += distillation_weight * loss_distill
                    if loss == 0.0:  # 所有权重都为0的情况（不应该发生，但为了安全起见）
                        loss = loss_ce  # 使用分类损失作为默认
                    # 统计损失（用于显示）
                    loss_sccm = torch.tensor(0.0, device=device)  # 占位符
                    loss_kdsp = torch.tensor(0.0, device=device)  # 占位符
            
            # 检查损失是否为 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected, skipping this batch")
                continue
            
            scaler.scale(loss).backward()
            # 梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 调用模型forward，根据模型类型处理不同的返回值
            forward_output = model(images, labels)
            
            if is_biomedcoop_pmcclip:
                # biomedcoop_pmcclip 返回 (logits, loss_ce, loss_sccm, loss_kdsp)
                logits, loss_ce, loss_sccm, loss_kdsp = forward_output
                # 计算总损失
                loss = loss_ce + loss_sccm + loss_kdsp
                # 统计损失（用于显示）
                contrastive_loss = torch.tensor(0.0, device=device)  # 占位符
                loss_distill = torch.tensor(0.0, device=device)  # 占位符
            elif is_hybrid_coop_sccm:
                # hybrid_coop_sccm 返回 (logits, loss_ce, contrastive_loss, loss_sccm, loss_distill)
                logits, loss_ce, contrastive_loss, loss_sccm, loss_distill = forward_output
                # 计算总损失
                loss = (classification_weight * loss_ce +
                       contrastive_weight * contrastive_loss +
                       loss_sccm +
                       distillation_weight * loss_distill)
                loss_kdsp = torch.tensor(0.0, device=device)  # 占位符
            else:
                # 其他模型返回 (logits, loss_ce, contrastive_loss, loss_distill)
                logits, loss_ce, contrastive_loss, loss_distill = forward_output
                # 根据权重计算总损失（如果权重为0，对应的损失不会被计算）
                loss = 0.0
                if use_ce_loss:
                    loss += classification_weight * loss_ce
                if use_contrastive_loss:
                    loss += contrastive_weight * contrastive_loss
                if use_distill_loss:
                    loss += distillation_weight * loss_distill
                if loss == 0.0:  # 所有权重都为0的情况（不应该发生，但为了安全起见）
                    loss = loss_ce  # 使用分类损失作为默认
                # 统计损失（用于显示）
                loss_sccm = torch.tensor(0.0, device=device)  # 占位符
                loss_kdsp = torch.tensor(0.0, device=device)  # 占位符
            
            # 检查损失是否为 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected, skipping this batch")
                continue
            
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 统计损失
        running_loss += loss.item()
        running_loss_ce += loss_ce.item() if isinstance(loss_ce, torch.Tensor) else 0.0
        
        if is_biomedcoop_pmcclip:
            running_loss_sccm += loss_sccm.item() if isinstance(loss_sccm, torch.Tensor) else 0.0
            running_loss_kdsp += loss_kdsp.item() if isinstance(loss_kdsp, torch.Tensor) else 0.0
            running_contrastive_loss += 0.0  # biomedcoop_pmcclip 不使用对比损失
            running_loss_distill += 0.0  # biomedcoop_pmcclip 不使用蒸馏损失
        elif is_hybrid_coop_sccm:
            running_loss_sccm += loss_sccm.item() if isinstance(loss_sccm, torch.Tensor) else 0.0
            running_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0
            running_loss_distill += loss_distill.item() if isinstance(loss_distill, torch.Tensor) else 0.0
            running_loss_kdsp += 0.0  # hybrid_coop_sccm 不使用KDSP损失
        else:
            running_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0
            running_loss_distill += loss_distill.item() if isinstance(loss_distill, torch.Tensor) else running_loss_distill + 0.0
            running_loss_sccm += 0.0  # 其他模型不使用SCCM损失
            running_loss_kdsp += 0.0  # 其他模型不使用KDSP损失
        
        # 计算准确率
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions.cpu() == labels.cpu()).sum().item()
            total += labels.size(0)
        
        # 更新进度条
        acc_str = f'{100 * correct / total:.2f}%' if total > 0 else '0.00%'
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': acc_str
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_loss_ce = running_loss_ce / len(dataloader)
    epoch_contrastive_loss = running_contrastive_loss / len(dataloader)
    epoch_loss_distill = running_loss_distill / len(dataloader)
    epoch_loss_sccm = running_loss_sccm / len(dataloader)
    epoch_loss_kdsp = running_loss_kdsp / len(dataloader)
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    
    # 返回损失
    if is_biomedcoop_pmcclip:
        return epoch_loss, epoch_acc, epoch_loss_ce, epoch_loss_sccm, epoch_loss_kdsp
    elif is_hybrid_coop_sccm:
        return epoch_loss, epoch_acc, epoch_loss_ce, epoch_contrastive_loss, epoch_loss_sccm, epoch_loss_distill
    else:
        return epoch_loss, epoch_acc, epoch_loss_ce, epoch_contrastive_loss, epoch_loss_distill


def validate(model, dataloader, device, use_amp=True):
    """
    验证模型
    
    Args:
        model: CustomCLIP 模型（直接使用，不通过适配器）
        dataloader: 数据加载器
        device: 设备
        use_amp: 是否使用混合精度
    
    Returns:
        epoch_loss: 平均损失
        epoch_acc: 准确率
        all_predictions: 所有预测结果
        all_labels: 所有真实标签
        val_mAP: mAP值（如果可用）
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='[Val]'):
            images = images.to(device)
            labels = labels.to(device)
            
            # 直接调用 CustomCLIP.forward()（评估模式，不传入 labels）
            logits = model(images)
            
            # 计算损失（用于记录）
            loss = F.cross_entropy(logits, labels)
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=1)
            
            running_loss += loss.item()
            correct += (predictions.cpu() == labels.cpu()).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    
    # 计算详细指标（如果可用）
    val_mAP = 0.0
    val_precision = 0.0
    val_recall = 0.0
    val_f1 = 0.0
    if METRICS_AVAILABLE:
        try:
            num_classes = len(set(all_labels))
            metrics = calculate_classification_metrics(all_labels, all_predictions, num_classes)
            val_mAP = metrics['mAP']
            val_precision = metrics['precision_macro']
            val_recall = metrics['recall_macro']
            val_f1 = metrics['f1_macro']
        except Exception as e:
            print(f"警告: 计算指标失败: {e}")
    
    return epoch_loss, epoch_acc, all_predictions, all_labels, val_mAP, val_precision, val_recall, val_f1


def train_biomedcoop_cross_validation(
    data_dir,
    output_dir,
    class_texts_file=None,
    batch_size=32,
    epochs=100,
    learning_rate=1e-4,
    weight_decay=0.01,
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
    use_weighted_sampling=False,
    weight_method='inverse_freq',
    weight_smooth_factor=1.0,
    # 模型类型参数
    model_type='biomedclip',  # 'biomedclip', 'clip', 'pmcclip', 'pubmedclip'
    clip_backbone='ViT-B/16',  # 用于 CLIP 和 PubMedCLIP
    # BiomedCoOp 特定参数（已不使用，保留以兼容）
    n_ctx=4,
    ctx_init="a photo of a",
    csc=False,
    class_token_position="end",
    sccm_lambda=1.0,
    kdsp_lambda=1.0,
    tau=1.0,
    n_prompts=4,
    use_focal_loss=False,
    focal_alpha=0.25,
    focal_gamma=2.0,
    freeze_image_encoder=False,
    # 损失权重参数
    classification_loss_weight=0.5,
    contrastive_loss_weight=0.5,
    distillation_loss_weight=0.0,
    # Hybrid 模型特定参数
    use_original_clip_resnet50=False,
):
    """
    使用K折交叉验证训练 BiomedCoOp 模型
    
    Args:
        data_dir: 数据目录（按类别组织的文件夹）
        output_dir: 输出目录
        class_texts_file: 类别文本描述JSON文件路径
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        img_size: 图像大小
        augmentation: 数据增强类型
        num_workers: 数据加载工作进程数
        use_amp: 是否使用混合精度训练
        gpu_id: GPU ID
        n_splits: 折数（默认5折）
        random_state: 随机种子
        save_best: 是否保存最佳模型
        early_stopping_patience: 早停耐心值
        early_stopping_min_delta: 早停最小改进阈值
        early_stopping_monitor: 早停监控指标
        use_weighted_sampling: 是否使用加权采样
        weight_method: 权重计算方法
        weight_smooth_factor: 权重平滑因子
        n_ctx: 上下文token数量
        ctx_init: 上下文初始化文本
        csc: 是否使用类别特定的上下文
        class_token_position: 类别token位置
        sccm_lambda: SCCM损失权重
        kdsp_lambda: KDSP损失权重（默认1.0）
        tau: 用于选择prompt的阈值（默认1.0）
        n_prompts: 使用的prompt数量（默认4）
        use_focal_loss: 是否使用 Focal Loss
        focal_alpha: Focal Loss alpha 参数
        focal_gamma: Focal Loss gamma 参数
        freeze_image_encoder: 是否冻结图像编码器
        classification_loss_weight: 分类损失权重（默认0.5）
        contrastive_loss_weight: 对比损失权重（默认0.5）
        distillation_loss_weight: 蒸馏损失权重（默认0.0，不使用蒸馏）
    """
    import random
    from pathlib import Path
    
    # 设置全局随机种子（确保可复现性）
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已设置为: {random_state}")
    
    # 设置设备
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config = {
        'model': 'biomedcoop',
        'class_texts_file': class_texts_file,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'img_size': img_size,
        'augmentation': augmentation,
        'n_splits': n_splits,
        'random_state': random_state,
        'n_ctx': n_ctx,
        'ctx_init': ctx_init,
        'csc': csc,
        'class_token_position': class_token_position,
        'sccm_lambda': sccm_lambda,
        'use_focal_loss': use_focal_loss,
        'focal_alpha': focal_alpha,
        'focal_gamma': focal_gamma,
        'freeze_image_encoder': freeze_image_encoder,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 数据增强
    train_transform, val_transform = get_data_augmentation(augmentation, img_size)
    
    # 创建完整数据集
    full_dataset = CLIPDataset(
        data_dir, 
        transform=None, 
        text_template=None,
        class_texts_dict=None,
        class_texts_file=class_texts_file
    )
    num_classes = len(full_dataset.class_to_idx)
    classnames = list(full_dataset.class_to_idx.keys())
    
    print(f"\n数据集: {len(full_dataset)} 个样本, {num_classes} 个类别")
    print(f"类别: {classnames}")
    
    # 创建交叉验证folds
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
        'fold_val_precision': [],
        'fold_val_recall': [],
        'fold_val_f1': [],
        'fold_best_val_acc': [],
        'fold_best_val_mAP': [],
        'fold_best_val_precision': [],
        'fold_best_val_recall': [],
        'fold_best_val_f1': [],
        'fold_best_epoch': [],
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
            if fold_num == 1:
                print(f"\nFold {fold_num} 启用加权采样以处理类别不平衡...")
                print(f"权重计算方法: {weight_method}")
                print(f"平滑因子: {weight_smooth_factor}")
            
            from collections import Counter
            # 优化：直接从原始数据集获取标签，避免加载图像
            # 这样比遍历 train_subset 快得多（不需要加载和变换图像）
            if fold_num == 1:
                print("  计算类别权重（直接从数据集获取标签，不加载图像）...")
            subset_labels = [full_dataset.samples[train_indices[i]][1] for i in range(len(train_indices))]
            class_counts = Counter(subset_labels)
            total_samples = len(subset_labels)
            num_classes_fold = len(class_counts)
            
            class_weights = {}
            for class_idx, count in class_counts.items():
                if weight_method == 'inverse_freq' or weight_method == 'balanced':
                    weight = total_samples / (num_classes_fold * (count + weight_smooth_factor))
                elif weight_method == 'inverse_sqrt':
                    weight = np.sqrt(total_samples / (count + weight_smooth_factor))
                else:
                    raise ValueError(f"Unknown weight_method: {weight_method}")
                class_weights[class_idx] = weight
            
            sample_weights = [class_weights[label] for label in subset_labels]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            if fold_num == 1:
                print("\n各类别权重:")
                for class_idx, weight in sorted(class_weights.items()):
                    class_name = full_dataset.idx_to_class[class_idx]
                    class_count = class_counts[class_idx]
                    print(f"  {class_name:25s}: 权重={weight:.4f}, 样本数={class_count}")
        
        # 创建DataLoader
        if fold_num == 1:
            print(f"\n创建 DataLoader...")
        if train_sampler is not None:
            train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler,
                                      num_workers=num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        if fold_num == 1:
            print(f"✓ DataLoader 创建完成")
        
        # 创建模型（根据 model_type 选择不同的模型）
        if fold_num == 1:
            print(f"\n创建模型（模型类型: {model_type}）:")
            if model_type == 'hybrid_coop':
                print(f"  使用 CoOp prompt learning（可学习的上下文 tokens）")
            else:
                print(f"  不使用 CoOp prompt learning")
            print(f"  只训练图像编码器，冻结文本编码器")
        
        # 创建配置对象
        cfg = create_biomedcoop_config(
            epochs=epochs,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            csc=csc,
            class_token_position=class_token_position,
            sccm_lambda=sccm_lambda,
            kdsp_lambda=kdsp_lambda,
            tau=tau,
            n_prompts=n_prompts,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            class_texts_file=class_texts_file,
            use_amp=use_amp,
            classification_loss_weight=classification_loss_weight,
            contrastive_loss_weight=contrastive_loss_weight,
            distillation_loss_weight=distillation_loss_weight,
            use_original_clip_resnet50=use_original_clip_resnet50
        )
        
        # 根据 model_type 加载不同的模型
        if model_type == 'biomedclip':
            if not BIOMEDCOOP_AVAILABLE:
                raise ImportError("BiomedCoOp components not available")
            print(f"加载 BiomedCLIP 模型...")
            try:
                base_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                print("✓ 成功加载 BiomedCLIP 模型")
            except Exception as e:
                print(f"✗ 加载模型失败: {e}")
                print("提示：如果网络连接有问题，请确保模型已下载到本地缓存")
                raise
            base_model = base_model.float().eval()
            base_model = base_model.to(device)
            model = BiomedCLIPCustomCLIP(cfg, classnames, base_model)
            
        elif model_type == 'clip':
            if not CLIP_AVAILABLE:
                raise ImportError("CLIP components not available")
            print(f"加载 CLIP 模型 (backbone: {clip_backbone})...")
            # 创建临时配置用于加载 CLIP
            class TempCfg:
                MODEL = type('obj', (object,), {'BACKBONE': type('obj', (object,), {'NAME': clip_backbone})()})()
            temp_cfg = TempCfg()
            base_model = load_clip_to_cpu(temp_cfg)
            base_model = base_model.float().eval()
            base_model = base_model.to(device)
            model = CLIPCustomCLIP(cfg, classnames, base_model)
            
        elif model_type == 'pmcclip':
            if not PMCCLIP_AVAILABLE:
                raise ImportError("PMC-CLIP components not available")
            print(f"加载 PMC-CLIP 模型...")
            import os
            directory = "clip/checkpoints"
            # 检查文件是否存在
            required_files = [
                "text_encoder.pth",
                "image_encoder(resnet50).pth",
                "text_projection_layer.pth"
            ]
            for filename in required_files:
                filepath = os.path.join(directory, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"PMC-CLIP checkpoint not found: {filepath}. Please run download_coop_models.py first.")
            
            from models.coop_pmcclip import ModifiedResNet, PMCCLIP
            from transformers import AutoModel
            image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
            image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'), weights_only=True))
            text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
            text_encoder.load_state_dict(torch.load(os.path.join(directory,'text_encoder.pth'), weights_only=True))
            text_projection_layer = torch.load(os.path.join(directory,'text_projection_layer.pth'), weights_only=True)
            text_projection_layer = nn.Parameter(text_projection_layer)
            
            image_encoder = image_encoder.to(device).eval()
            text_encoder = text_encoder.to(device).eval()
            text_projection_layer = text_projection_layer.to(device)
            
            base_model = PMCCLIP(image_encoder, text_encoder, text_projection_layer).to(device).eval()
            print("✓ 成功加载 PMC-CLIP 模型")
            # 验证使用的类
            print(f"创建模型使用的类: {PMCCLIPCustomCLIP.__module__}.{PMCCLIPCustomCLIP.__name__}")
            if 'biomedcoop_pmcclip' in PMCCLIPCustomCLIP.__module__:
                print("✓ 使用 biomedcoop_pmcclip.CustomCLIP")
            else:
                print(f"⚠️  警告: 使用 {PMCCLIPCustomCLIP.__module__}.CustomCLIP (不是 biomedcoop_pmcclip)")
            model = PMCCLIPCustomCLIP(cfg, classnames, base_model)
            
        elif model_type == 'pubmedclip':
            if not PUBMEDCLIP_AVAILABLE:
                raise ImportError("PubMedCLIP components not available")
            print(f"加载 PubMedCLIP 模型 (backbone: {clip_backbone})...")
            # 创建临时配置用于加载 PubMedCLIP
            class TempCfg:
                MODEL = type('obj', (object,), {'BACKBONE': type('obj', (object,), {'NAME': clip_backbone})()})()
            temp_cfg = TempCfg()
            base_model = load_pubmedclip_to_cpu(temp_cfg)
            base_model = base_model.float().eval()
            base_model = base_model.to(device)
            model = PubMedCLIPCustomCLIP(cfg, classnames, base_model)

        elif model_type == 'hybrid':
            if not HYBRID_AVAILABLE:
                raise ImportError("Hybrid PMC-CLIP + BiomedCLIP components not available")
            
            # 确保 create_model_from_pretrained 可用
            try:
                from open_clip.src.open_clip import create_model_from_pretrained
            except ImportError:
                raise ImportError("无法导入 open_clip。请安装: pip install open-clip-torch")
            
            use_original_clip = cfg.TRAINER.BIOMEDCOOP.USE_ORIGINAL_CLIP_RESNET50
            if use_original_clip:
                print(f"加载混合模型：原始 CLIP ResNet50 + BiomedCLIP 文本编码器...")
            else:
                print(f"加载混合模型：PMC-CLIP ResNet50 + BiomedCLIP 文本编码器...")

            # 加载图像编码器（PMC-CLIP 或原始 CLIP 的 ResNet50）
            if use_original_clip:
                # 使用原始 CLIP 的 ResNet50
                try:
                    import clip
                except ImportError:
                    raise ImportError("无法导入 clip 库。请安装: pip install git+https://github.com/openai/CLIP.git")
                
                print("加载原始 CLIP ResNet50 图像编码器...")
                clip_model, _ = clip.load('RN50', device='cpu')
                image_encoder = clip_model.visual  # 获取视觉编码器（ResNet50）
                print("✓ 原始 CLIP ResNet50 图像编码器加载完成")
                print(f"  输出维度: 1024")
            else:
                # 使用 PMC-CLIP 的 ResNet50
                import os
                directory = "clip/checkpoints"
                required_files = [
                    "text_encoder.pth",  # PMC-CLIP 的（不会使用）
                    "image_encoder(resnet50).pth",  # PMC-CLIP 的图像编码器
                    "text_projection_layer.pth"  # PMC-CLIP 的（不会使用）
                ]
                for filename in required_files:
                    filepath = os.path.join(directory, filename)
                    if not os.path.exists(filepath):
                        raise FileNotFoundError(f"PMC-CLIP checkpoint not found: {filepath}. Please run download_coop_models.py first.")

                # 加载 PMC-CLIP 的 ResNet50 图像编码器
                from models.coop_pmcclip import ModifiedResNet
                image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
                image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'), weights_only=True))
                print("✓ PMC-CLIP ResNet50 图像编码器加载完成")
                print(f"  输出维度: 768")

            # 加载 BiomedCLIP 文本编码器
            try:
                biomedclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                print("✓ 成功加载 BiomedCLIP 模型")
            except Exception as e:
                print(f"✗ 加载 BiomedCLIP 失败: {e}")
                print("提示：如果网络连接有问题，请确保模型已下载到本地缓存")
                raise

            if cfg.TRAINER.BIOMEDCOOP.PREC == "fp32" or cfg.TRAINER.BIOMEDCOOP.PREC == "amp":
                biomedclip_model.float()

            image_encoder = image_encoder.to(device).eval()
            biomedclip_model = biomedclip_model.to(device).eval()

            model = HybridCLIPCustomCLIP(cfg, classnames, image_encoder, biomedclip_model, use_original_clip=use_original_clip)
            print("✓ 成功加载混合模型")

        elif model_type == 'hybrid_coop':
            if not HYBRID_COOP_AVAILABLE:
                raise ImportError("Hybrid PMC-CLIP + BiomedCLIP + CoOp components not available")
            
            # 确保 create_model_from_pretrained 可用
            try:
                from open_clip.src.open_clip import create_model_from_pretrained
            except ImportError:
                raise ImportError("无法导入 open_clip。请安装: pip install open-clip-torch")
            
            print(f"加载混合模型（带 CoOp）：PMC-CLIP ResNet50 + BiomedCLIP 文本编码器 + CoOp Prompt Learning...")

            # 检查 PMC-CLIP 文件
            import os
            directory = "clip/checkpoints"
            required_files = [
                "text_encoder.pth",  # PMC-CLIP 的（不会使用）
                "image_encoder(resnet50).pth",  # PMC-CLIP 的图像编码器
                "text_projection_layer.pth"  # PMC-CLIP 的（不会使用）
            ]
            for filename in required_files:
                filepath = os.path.join(directory, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"PMC-CLIP checkpoint not found: {filepath}. Please run download_coop_models.py first.")

            # 加载 PMC-CLIP 的 ResNet50 图像编码器
            from models.coop_pmcclip import ModifiedResNet
            pmc_image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
            pmc_image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'), weights_only=True))

            # 加载 BiomedCLIP 文本编码器
            try:
                biomedclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                print("✓ 成功加载 BiomedCLIP 模型")
            except Exception as e:
                print(f"✗ 加载 BiomedCLIP 失败: {e}")
                print("提示：如果网络连接有问题，请确保模型已下载到本地缓存")
                raise

            if cfg.TRAINER.BIOMEDCOOP.PREC == "fp32" or cfg.TRAINER.BIOMEDCOOP.PREC == "amp":
                biomedclip_model.float()

            pmc_image_encoder = pmc_image_encoder.to(device).eval()
            biomedclip_model = biomedclip_model.to(device).eval()

            model = HybridCLIPWithCoOpCustomCLIP(cfg, classnames, pmc_image_encoder, biomedclip_model)
            print("✓ 成功加载混合模型（带 CoOp）")

        elif model_type == 'pmcclip_full':
            if not PMCCLIP_FULL_AVAILABLE:
                raise ImportError("PMC-CLIP Full components not available")
            print(f"加载 PMC-CLIP 完整模型（图像编码器 + 文本编码器）...")
            
            # 加载 PMC-CLIP 模型
            pmcclip_model = load_pmcclip_model(device=device)
            
            # 加载 BiomedCLIP 作为 teacher（用于蒸馏）
            teacher_model = None
            if distillation_loss_weight > 0:
                print("加载 BiomedCLIP 作为 teacher 模型（用于蒸馏）...")
                try:
                    from open_clip.src.open_clip import create_model_from_pretrained
                    teacher_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                    teacher_model = teacher_model.float().to(device).eval()
                    print("✓ 成功加载 BiomedCLIP teacher 模型")
                except Exception as e:
                    print(f"⚠️  加载 BiomedCLIP teacher 失败: {e}")
                    print("   将不使用蒸馏损失")
                    teacher_model = None
            
            model = HybridPMCCLIPOnly(cfg, classnames, pmcclip_model, teacher_model)
            print("✓ 成功加载 PMC-CLIP 完整模型")

        elif model_type == 'hybrid_coop_sccm':
            if not HYBRID_COOP_SCCM_AVAILABLE:
                raise ImportError("Hybrid PMC-CLIP + BiomedCLIP + CoOp + SCCM components not available")
            print(f"加载混合模型（带 CoOp + SCCM）：PMC-CLIP ResNet50 + BiomedCLIP 文本编码器 + CoOp + SCCM...")
            
            # 检查 PMC-CLIP 文件
            import os
            directory = "clip/checkpoints"
            required_files = [
                "image_encoder(resnet50).pth",
            ]
            for filename in required_files:
                filepath = os.path.join(directory, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"PMC-CLIP checkpoint not found: {filepath}. Please run download_coop_models.py first.")
            
            # 加载 PMC-CLIP ResNet50
            from models.coop_pmcclip import ModifiedResNet
            pmc_image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
            pmc_image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'), weights_only=True))
            
            # 加载 BiomedCLIP
            try:
                from open_clip.src.open_clip import create_model_from_pretrained
                biomedclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                print("✓ 成功加载 BiomedCLIP 模型")
            except Exception as e:
                print(f"✗ 加载 BiomedCLIP 失败: {e}")
                raise
            
            if cfg.TRAINER.BIOMEDCOOP.PREC == "fp32" or cfg.TRAINER.BIOMEDCOOP.PREC == "amp":
                biomedclip_model.float()
            
            pmc_image_encoder = pmc_image_encoder.to(device).eval()
            biomedclip_model = biomedclip_model.to(device).eval()
            
            model = HybridCLIPWithCoOpSCCM(cfg, classnames, pmc_image_encoder, biomedclip_model)
            print("✓ 成功加载混合模型（带 CoOp + SCCM）")

        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose from: 'biomedclip', 'clip', 'pmcclip', 'pubmedclip', 'hybrid', 'hybrid_coop', 'pmcclip_full', 'hybrid_coop_sccm'")
        
        model = model.to(device)
        
        # 冻结文本编码器（根据模型类型）
        if fold_num == 1:
            print("\n冻结文本编码器...")
        
        # 根据模型类型冻结文本编码器
        if model_type == 'biomedclip':
            # 冻结完整模型的所有参数（包括文本编码器）
            for param in model.biomedclip_model.parameters():
                param.requires_grad = False
        elif model_type == 'clip':
            # 冻结 CLIP 模型的文本编码器
            for param in model.clip_model.parameters():
                param.requires_grad = False
        elif model_type == 'pmcclip':
            # 冻结 PMC-CLIP 模型的文本编码器
            for param in model.pmcclip_model.parameters():
                param.requires_grad = False
        elif model_type == 'pubmedclip':
            # 冻结 PubMedCLIP 模型的文本编码器
            for param in model.clip_model.parameters():
                param.requires_grad = False
        elif model_type == 'hybrid' or model_type == 'hybrid_coop':
            # 冻结 BiomedCLIP 文本编码器
            for param in model.biomedclip_model.parameters():
                param.requires_grad = False
            # 文本投影层保持可训练（用于对齐图像和文本特征维度）
            # 投影层在 HybridCLIP 类中，会在后面统一处理图像编码器时一起处理
            # 对于 hybrid_coop，CoOp prompts 也是可训练的
        elif model_type == 'hybrid_coop_sccm':
            # 冻结 BiomedCLIP 文本编码器
            for param in model.biomedclip_model.parameters():
                param.requires_grad = False
            # CoOp prompts 和图像编码器将在后面设置为可训练
        elif model_type == 'pmcclip_full':
            # 冻结 PMC-CLIP 文本编码器
            for param in model.text_encoder.parameters():
                param.requires_grad = False
            # 冻结文本投影层（使用 requires_grad_() 方法）
            if hasattr(model, 'text_projection_layer'):
                if isinstance(model.text_projection_layer, torch.nn.Parameter):
                    model.text_projection_layer.requires_grad_(False)
            # 冻结 teacher 模型（如果存在）
            if hasattr(model, 'teacher_model') and model.teacher_model is not None:
                for param in model.teacher_model.parameters():
                    param.requires_grad = False
        
        # 冻结 logit_scale（如果是 Parameter）
        if hasattr(model, 'logit_scale'):
            if isinstance(model.logit_scale, torch.nn.Parameter):
                model.logit_scale.requires_grad = False
            # 如果是 float 或其他类型，不需要冻结（PMC-CLIP 的 logit_scale 是 float）
        
        # 冻结图像编码器（如果指定）
        if freeze_image_encoder:
            if fold_num == 1:
                print("冻结图像编码器...")
            for param in model.image_encoder.parameters():
                param.requires_grad = False
        else:
            # 默认情况下，只训练图像编码器（文本编码器已冻结）
            if fold_num == 1:
                print("只训练图像编码器（文本编码器已冻结）...")
            # 确保图像编码器参数可训练
            for param in model.image_encoder.parameters():
                param.requires_grad = True
            # 对于混合模型，还需要训练文本投影层（用于对齐维度）
            if model_type in ['hybrid', 'hybrid_coop'] and hasattr(model, 'text_projection'):
                if fold_num == 1:
                    print("  同时训练文本投影层（对齐图像和文本特征维度）...")
                for param in model.text_projection.parameters():
                    param.requires_grad = True
                if hasattr(model, 'distill_projection'):
                    if fold_num == 1:
                        print("  同时训练蒸馏投影层（用于知识蒸馏）...")
                    for param in model.distill_projection.parameters():
                        param.requires_grad = True
            # 对于 pmcclip_full 模型，训练蒸馏投影层（如果存在）
            if model_type == 'pmcclip_full' and hasattr(model, 'distill_projection'):
                if fold_num == 1:
                    print("  同时训练蒸馏投影层（768->512，用于知识蒸馏）...")
                for param in model.distill_projection.parameters():
                    param.requires_grad = True
            # 对于 hybrid_coop，还需要训练 CoOp prompts
            if model_type == 'hybrid_coop' and hasattr(model, 'prompt_learner'):
                if fold_num == 1:
                    print("  同时训练 CoOp prompts（可学习的上下文 tokens）...")
                # 只训练 ctx 参数，不训练其他参数（如 biomedclip_model_temp）
                for name, param in model.prompt_learner.named_parameters():
                    if name == 'ctx':
                        param.requires_grad = True
                        if fold_num == 1:
                            print(f"    训练参数: prompt_learner.{name} ({param.numel():,} 参数)")
                    else:
                        param.requires_grad = False
            # 对于 hybrid_coop_sccm，训练 CoOp prompts 和图像投影层
            if model_type == 'hybrid_coop_sccm':
                if fold_num == 1:
                    print("  同时训练 CoOp prompts（可学习的上下文 tokens）...")
                # 训练 CoOp ctx 参数
                if hasattr(model, 'prompt_learner'):
                    for name, param in model.prompt_learner.named_parameters():
                        if name == 'ctx':
                            param.requires_grad = True
                            if fold_num == 1:
                                print(f"    训练参数: prompt_learner.{name} ({param.numel():,} 参数)")
                        else:
                            param.requires_grad = False
                # 训练图像投影层
                if hasattr(model, 'image_projection'):
                    if fold_num == 1:
                        print("  同时训练图像投影层（768->512）...")
                    for param in model.image_projection.parameters():
                        param.requires_grad = True
        
        # 对于使用 BiomedCoOp 的模型（pmcclip），启用 prompt_learner.ctx 的训练
        # 这是 BiomedCoOp 的核心：可学习的 prompt tokens
        if model_type == 'pmcclip' and hasattr(model, 'prompt_learner'):
            if fold_num == 1:
                print("  启用 BiomedCoOp prompt learning（可学习的上下文 tokens）...")
            # 只训练 ctx 参数，不训练其他参数（如 token_prefix, token_suffix, fixed_embeddings）
            for name, param in model.prompt_learner.named_parameters():
                if name == 'ctx':
                    param.requires_grad = True
                    if fold_num == 1:
                        print(f"    训练参数: prompt_learner.{name} ({param.numel():,} 参数)")
                else:
                    param.requires_grad = False
        
        # 优化器 - 使用分层学习率
        # prompt_learner.ctx 使用较大学习率（从头学习）
        # image_encoder 使用较小学习率（预训练微调）
        param_groups = []
        
        # 收集 prompt_learner.ctx 参数（如果存在）
        prompt_params = []
        image_encoder_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'prompt_learner.ctx' in name:
                prompt_params.append(param)
            elif 'image_encoder' in name:
                image_encoder_params.append(param)
            else:
                other_params.append(param)
        
        # 设置分层学习率
        # prompt_learner: 使用 10x 基础学习率
        # image_encoder: 使用 0.1x 基础学习率（更保守的微调）
        if prompt_params:
            prompt_lr = learning_rate * 10  # 例如 1e-4 * 10 = 1e-3
            param_groups.append({'params': prompt_params, 'lr': prompt_lr, 'name': 'prompt_learner'})
            if fold_num == 1:
                print(f"  prompt_learner 学习率: {prompt_lr}")
        
        if image_encoder_params:
            image_lr = learning_rate * 0.1  # 例如 1e-4 * 0.1 = 1e-5
            param_groups.append({'params': image_encoder_params, 'lr': image_lr, 'name': 'image_encoder'})
            if fold_num == 1:
                print(f"  image_encoder 学习率: {image_lr}")
        
        if other_params:
            param_groups.append({'params': other_params, 'lr': learning_rate, 'name': 'other'})
            if fold_num == 1:
                print(f"  其他参数学习率: {learning_rate}")
        
        optimizer = optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
        
        # 打印参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params_count = total_params - trainable_params_count
        
        if fold_num == 1:
            print(f"\n参数统计:")
            print(f"  总参数: {total_params:,}")
            print(f"  可训练参数: {trainable_params_count:,} ({100*trainable_params_count/total_params:.2f}%)")
            print(f"  冻结参数: {frozen_params_count:,} ({100*frozen_params_count/total_params:.2f}%)")
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # AMP scaler
        scaler = None
        if use_amp and device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        
        # 早停相关
        use_early_stopping = early_stopping_patience is not None and early_stopping_patience > 0
        early_stopping_counter = 0
        # 最佳指标（用于模型保存和早停）
        best_val_acc = 0.0
        best_val_mAP = 0.0  # 主要指标：用于保存最佳模型
        best_val_precision = 0.0
        best_val_recall = 0.0
        best_val_f1 = 0.0
        best_val_loss = float('inf')
        best_epoch = 0
        
        # 打印训练策略说明
        if fold_num == 1:
            print(f"\n训练策略:")
            print(f"  - 模型保存: 基于验证集 mAP（始终保存最优 mAP 模型）")
            print(f"  - 早停监控: {early_stopping_monitor}")
            if use_early_stopping:
                print(f"  - 早停耐心值: {early_stopping_patience} epochs")
                print(f"  - 早停最小改进: {early_stopping_min_delta}")
            else:
                print(f"  - 早停: 未启用")
        
        # 检测模型类型（处理 DataParallel 包装的情况）
        actual_model = model.module if hasattr(model, 'module') else model
        
        # 检测是否是 biomedcoop_pmcclip 模型
        is_biomedcoop_pmcclip = False
        is_hybrid_coop_sccm = False
        if hasattr(actual_model, '__class__'):
            model_class_name = actual_model.__class__.__name__
            model_module = actual_model.__class__.__module__
            # 检查是否是 biomedcoop_pmcclip 中的 CustomCLIP
            if 'biomedcoop_pmcclip' in model_module and model_class_name == 'CustomCLIP':
                is_biomedcoop_pmcclip = True
            # 或者通过检查关键属性
            elif (hasattr(actual_model, 'prompt_learner') and 
                  hasattr(actual_model.prompt_learner, 'fixed_embeddings') and
                  hasattr(actual_model, 'cfg') and
                  hasattr(actual_model.cfg, 'TRAINER') and
                  hasattr(actual_model.cfg.TRAINER, 'BIOMEDCOOP')):
                is_biomedcoop_pmcclip = True
            # 检查是否是 hybrid_coop_sccm
            elif 'hybrid_pmcclip_biomedclip_coop_sccm' in model_module and model_class_name == 'HybridCLIPWithCoOpSCCM':
                is_hybrid_coop_sccm = True
            elif hasattr(actual_model, 'sccm_lambda') and hasattr(actual_model, 'prompt_learner'):
                is_hybrid_coop_sccm = True
        
        # 调试信息（仅第一次）
        if fold_num == 1:
            print(f"\n模型类型检测:")
            print(f"  - 模型类型: {type(actual_model).__name__}")
            print(f"  - 模型模块: {actual_model.__class__.__module__}")
            print(f"  - 是否 biomedcoop_pmcclip: {is_biomedcoop_pmcclip}")
            print(f"  - 是否 hybrid_coop_sccm: {is_hybrid_coop_sccm}")
            if hasattr(actual_model, 'cfg') and hasattr(actual_model.cfg, 'TRAINER') and hasattr(actual_model.cfg.TRAINER, 'BIOMEDCOOP'):
                print(f"  - SCCM_LAMBDA: {actual_model.cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA}")
                print(f"  - KDSP_LAMBDA: {actual_model.cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA}")
                print(f"  - N_PROMPTS: {actual_model.cfg.TRAINER.BIOMEDCOOP.N_PROMPTS}")
            elif hasattr(actual_model, 'sccm_lambda'):
                print(f"  - SCCM_LAMBDA: {actual_model.sccm_lambda}")
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_loss_ce': [],
            'train_contrastive_loss': [],
            'train_loss_distill': [],
            'train_loss_sccm': [],
            'train_loss_kdsp': [],
            'val_loss': [],
            'val_acc': [],
            'val_mAP': [],
        }
        
        # 训练循环
        print(f"\n开始训练 Fold {fold_num}...")
        for epoch in range(epochs):
            # 训练
            train_epoch_result = train_epoch(
                model, train_loader, optimizer, device, epoch+1, use_amp=use_amp, scaler=scaler
            )
            
            if is_biomedcoop_pmcclip:
                # biomedcoop_pmcclip 返回 (loss, acc, loss_ce, loss_sccm, loss_kdsp)
                train_loss, train_acc, train_loss_ce, train_loss_sccm, train_loss_kdsp = train_epoch_result
                train_contrastive_loss = 0.0
                train_loss_distill = 0.0
            elif is_hybrid_coop_sccm:
                # hybrid_coop_sccm 返回 (loss, acc, loss_ce, contrastive_loss, loss_sccm, loss_distill)
                train_loss, train_acc, train_loss_ce, train_contrastive_loss, train_loss_sccm, train_loss_distill = train_epoch_result
                train_loss_kdsp = 0.0
            else:
                # 其他模型返回 (loss, acc, loss_ce, contrastive_loss, loss_distill)
                train_loss, train_acc, train_loss_ce, train_contrastive_loss, train_loss_distill = train_epoch_result
                train_loss_sccm = 0.0
                train_loss_kdsp = 0.0
            
            # 验证
            val_loss, val_acc, val_predictions, val_labels, val_mAP, val_precision, val_recall, val_f1 = validate(
                model, val_loader, device, use_amp=use_amp
            )
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_loss_ce'].append(train_loss_ce)
            history['train_contrastive_loss'].append(train_contrastive_loss)
            history['train_loss_distill'].append(train_loss_distill)
            history['train_loss_sccm'].append(train_loss_sccm)
            history['train_loss_kdsp'].append(train_loss_kdsp)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_mAP'].append(val_mAP)
            
            # 保存最佳模型（始终基于 mAP）
            # 无论 early_stopping_monitor 设置为什么，都基于 mAP 保存最佳模型
            improved_mAP = False
            if val_mAP > best_val_mAP + early_stopping_min_delta:
                best_val_mAP = val_mAP
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_val_precision = val_precision
                best_val_recall = val_recall
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                improved_mAP = True
            
            # 早停判断（根据 early_stopping_monitor 设置）
            improved_early_stop = False
            if early_stopping_monitor == 'val_acc':
                if val_acc > best_val_acc + early_stopping_min_delta:
                    improved_early_stop = True
            elif early_stopping_monitor == 'val_loss':
                if val_loss < best_val_loss - early_stopping_min_delta:
                    improved_early_stop = True
            elif early_stopping_monitor == 'val_mAP':
                improved_early_stop = improved_mAP  # 如果监控 mAP，则与保存逻辑一致
            
            # 保存最佳模型（基于 mAP）
            if improved_mAP and save_best:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_mAP': val_mAP,
                    'val_loss': val_loss,
                    'class_to_idx': full_dataset.class_to_idx,
                }, fold_dir / 'best_model.pth')
                print(f"  ✓ 保存最佳 mAP 模型 (mAP: {val_mAP:.2f}%, Epoch {epoch+1})")
            
            # 早停计数器（根据 early_stopping_monitor）
            if improved_early_stop:
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # 打印结果
            print(f"Fold {fold_num}, Epoch {epoch+1}/{epochs}")
            if is_biomedcoop_pmcclip:
                print(f"  Train Loss: {train_loss:.4f} (CE: {train_loss_ce:.4f}, SCCM: {train_loss_sccm:.4f}, KDSP: {train_loss_kdsp:.4f}), Train Acc: {train_acc:.2f}%")
            elif is_hybrid_coop_sccm:
                print(f"  Train Loss: {train_loss:.4f} (CE: {train_loss_ce:.4f}, Contrastive: {train_contrastive_loss:.4f}, SCCM: {train_loss_sccm:.4f}, Distill: {train_loss_distill:.4f}), Train Acc: {train_acc:.2f}%")
            elif hasattr(model, 'distillation_loss_weight') and model.distillation_loss_weight > 0:
                print(f"  Train Loss: {train_loss:.4f} (CE: {train_loss_ce:.4f}, Contrastive: {train_contrastive_loss:.4f}, Distill: {train_loss_distill:.4f}), Train Acc: {train_acc:.2f}%")
            else:
                print(f"  Train Loss: {train_loss:.4f} (CE: {train_loss_ce:.4f}, Contrastive: {train_contrastive_loss:.4f}), Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val mAP: {val_mAP:.2f}%")
            print(f"  Best Val mAP: {best_val_mAP:.2f}% (Epoch {best_epoch}) - 模型保存基于 mAP")
            print(f"  Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
            if use_early_stopping:
                print(f"  早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            print("-" * 80)
            
            # 保存训练日志
            log_file = fold_dir / 'train.log'
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Fold {fold_num}, Epoch {epoch+1}/{epochs}\n")
                f.write(f"{'='*80}\n")
                if is_biomedcoop_pmcclip:
                    f.write(f"Train Loss: {train_loss:.4f} (CE: {train_loss_ce:.4f}, SCCM: {train_loss_sccm:.4f}, KDSP: {train_loss_kdsp:.4f}), Train Acc: {train_acc:.2f}%\n")
                elif is_hybrid_coop_sccm:
                    f.write(f"Train Loss: {train_loss:.4f} (CE: {train_loss_ce:.4f}, Contrastive: {train_contrastive_loss:.4f}, SCCM: {train_loss_sccm:.4f}, Distill: {train_loss_distill:.4f}), Train Acc: {train_acc:.2f}%\n")
                elif hasattr(model, 'distillation_loss_weight') and model.distillation_loss_weight > 0:
                    f.write(f"Train Loss: {train_loss:.4f} (CE: {train_loss_ce:.4f}, Contrastive: {train_contrastive_loss:.4f}, Distill: {train_loss_distill:.4f}), Train Acc: {train_acc:.2f}%\n")
                else:
                    f.write(f"Train Loss: {train_loss:.4f} (CE: {train_loss_ce:.4f}, Contrastive: {train_contrastive_loss:.4f}), Train Acc: {train_acc:.2f}%\n")
                f.write(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val mAP: {val_mAP:.2f}%\n")
                f.write(f"Best Val mAP: {best_val_mAP:.2f}% (Epoch {best_epoch}) - 模型保存基于 mAP\n")
                f.write(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})\n")
                if use_early_stopping:
                    f.write(f"早停计数器: {early_stopping_counter}/{early_stopping_patience}\n")
                f.flush()
            
            # 早停检查
            if use_early_stopping and early_stopping_counter >= early_stopping_patience:
                print(f"\n早停触发! 验证指标在 {early_stopping_patience} 个epoch内没有改善")
                break
        
        # 保存训练历史
        with open(fold_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # 获取最终验证指标（最后一个epoch的指标）
        final_val_precision = val_precision
        final_val_recall = val_recall
        final_val_f1 = val_f1
        
        # 记录fold结果
        all_fold_results['fold_train_loss'].append(history['train_loss'][-1])
        all_fold_results['fold_train_acc'].append(history['train_acc'][-1])
        all_fold_results['fold_val_loss'].append(history['val_loss'][-1])
        all_fold_results['fold_val_acc'].append(history['val_acc'][-1])
        all_fold_results['fold_val_mAP'].append(history['val_mAP'][-1])
        all_fold_results['fold_val_precision'].append(final_val_precision)
        all_fold_results['fold_val_recall'].append(final_val_recall)
        all_fold_results['fold_val_f1'].append(final_val_f1)
        all_fold_results['fold_best_val_acc'].append(best_val_acc)
        all_fold_results['fold_best_val_mAP'].append(best_val_mAP)
        all_fold_results['fold_best_val_precision'].append(best_val_precision)
        all_fold_results['fold_best_val_recall'].append(best_val_recall)
        all_fold_results['fold_best_val_f1'].append(best_val_f1)
        all_fold_results['fold_best_epoch'].append(best_epoch)
        
        print(f"\nFold {fold_num} 训练完成!")
        print(f"  最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"  最佳验证mAP: {best_val_mAP:.2f}% (Epoch {best_epoch})")
    
    # 计算交叉验证平均结果
    cv_summary = {
        'mode': 'cv',
        'n_splits': n_splits,
        'random_state': random_state,
        'average_best_val_acc': np.mean(all_fold_results['fold_best_val_acc']),
        'std_best_val_acc': np.std(all_fold_results['fold_best_val_acc']),
        'average_best_val_mAP': np.mean(all_fold_results['fold_best_val_mAP']),
        'std_best_val_mAP': np.std(all_fold_results['fold_best_val_mAP']),
        'average_best_val_precision': np.mean(all_fold_results['fold_best_val_precision']),
        'std_best_val_precision': np.std(all_fold_results['fold_best_val_precision']),
        'average_best_val_recall': np.mean(all_fold_results['fold_best_val_recall']),
        'std_best_val_recall': np.std(all_fold_results['fold_best_val_recall']),
        'average_best_val_f1': np.mean(all_fold_results['fold_best_val_f1']),
        'std_best_val_f1': np.std(all_fold_results['fold_best_val_f1']),
        'average_val_acc': np.mean(all_fold_results['fold_val_acc']),
        'std_val_acc': np.std(all_fold_results['fold_val_acc']),
        'average_val_mAP': np.mean(all_fold_results['fold_val_mAP']),
        'std_val_mAP': np.std(all_fold_results['fold_val_mAP']),
        'average_val_precision': np.mean(all_fold_results['fold_val_precision']),
        'std_val_precision': np.std(all_fold_results['fold_val_precision']),
        'average_val_recall': np.mean(all_fold_results['fold_val_recall']),
        'std_val_recall': np.std(all_fold_results['fold_val_recall']),
        'average_val_f1': np.mean(all_fold_results['fold_val_f1']),
        'std_val_f1': np.std(all_fold_results['fold_val_f1']),
        'average_val_loss': np.mean(all_fold_results['fold_val_loss']),
        'fold_results': all_fold_results,
    }
    
    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    # 按照参考日志格式输出结果
    print(f"\n{'='*80}")
    print("交叉验证完成！")
    print(f"{'='*80}\n")
    print("详细结果:")
    
    # 输出每个fold的详细结果
    for fold_num in range(1, n_splits + 1):
        idx = fold_num - 1
        print(f"  Fold {fold_num}:")
        print(f"    最佳验证mAP: {all_fold_results['fold_best_val_mAP'][idx]:.2f}% (Epoch {all_fold_results['fold_best_epoch'][idx]})")
        print(f"    最佳验证准确率: {all_fold_results['fold_best_val_acc'][idx]:.2f}% (Epoch {all_fold_results['fold_best_epoch'][idx]})")
        print(f"    mAP: {all_fold_results['fold_best_val_mAP'][idx]:.2f}%")
        print(f"    Precision: {all_fold_results['fold_best_val_precision'][idx]:.2f}%")
        print(f"    Recall: {all_fold_results['fold_best_val_recall'][idx]:.2f}%")
        print(f"    F1 Score: {all_fold_results['fold_best_val_f1'][idx]:.2f}%")
    
    # 输出平均结果
    print("\n平均结果:")
    print(f"  平均最佳验证mAP: {cv_summary['average_best_val_mAP']:.2f}% ± {cv_summary['std_best_val_mAP']:.2f}%")
    print(f"  平均最佳验证准确率: {cv_summary['average_best_val_acc']:.2f}% ± {cv_summary['std_best_val_acc']:.2f}%")
    print(f"  平均mAP: {cv_summary['average_best_val_mAP']:.2f}% ± {cv_summary['std_best_val_mAP']:.2f}%")
    print(f"  平均Precision: {cv_summary['average_best_val_precision']:.2f}% ± {cv_summary['std_best_val_precision']:.2f}%")
    print(f"  平均Recall: {cv_summary['average_best_val_recall']:.2f}% ± {cv_summary['std_best_val_recall']:.2f}%")
    print(f"  平均F1 Score: {cv_summary['average_best_val_f1']:.2f}% ± {cv_summary['std_best_val_f1']:.2f}%")
    print(f"  平均最终验证准确率: {cv_summary['average_val_acc']:.2f}%")
    print(f"  平均最终验证损失: {cv_summary['average_val_loss']:.4f}")
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"  - cv_summary.json: 交叉验证汇总结果")
    print(f"  - fold_N/: 各折的训练历史和最佳模型")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='BiomedCoOp 模型训练脚本')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, required=True, help='数据目录（按类别组织的文件夹）')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--class-texts-file', type=str, default=None, help='类别文本描述JSON文件路径')
    
    # 模型类型参数
    parser.add_argument('--model-type', type=str, default='biomedclip',
                       choices=['biomedclip', 'clip', 'pmcclip', 'pubmedclip', 'hybrid', 'hybrid_coop', 'pmcclip_full', 'hybrid_coop_sccm'],
                       help='模型类型: biomedclip, clip, pmcclip, pubmedclip, hybrid, hybrid_coop, pmcclip_full, hybrid_coop_sccm (默认: biomedclip)')
    parser.add_argument('--clip-backbone', type=str, default='ViT-B/16',
                       choices=['ViT-B/16', 'ViT-B/32', 'RN50', 'RN101'],
                       help='CLIP/PubMedCLIP 的 backbone (默认: ViT-B/16)')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--img-size', type=int, default=224, help='图像大小')
    parser.add_argument('--augmentation', type=str, default='standard', 
                       choices=['none', 'minimal', 'standard'],
                       help='数据增强类型')
    
    # 其他参数
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度训练')
    
    # 交叉验证参数
    parser.add_argument('--n-splits', type=int, default=5, help='交叉验证折数（默认5折）')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子')
    
    # 早停参数
    parser.add_argument('--early-stopping-patience', type=int, default=None, 
                       help='早停耐心值，如果设置为None则不使用早停（默认None）')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                       help='早停最小改进阈值（默认0.0）')
    parser.add_argument('--early-stopping-monitor', type=str, default='val_loss',
                       choices=['val_acc', 'val_loss', 'val_mAP'],
                       help='早停监控指标')
    
    # 加权采样参数
    parser.add_argument('--use-weighted-sampling', action='store_true', 
                       help='启用加权采样以处理类别不平衡问题')
    parser.add_argument('--weight-method', type=str, default='inverse_freq',
                       choices=['inverse_freq', 'inverse_sqrt', 'balanced'],
                       help='权重计算方法')
    parser.add_argument('--weight-smooth-factor', type=float, default=1.0,
                       help='权重平滑因子')
    
    # BiomedCoOp 特定参数
    parser.add_argument('--n-ctx', type=int, default=4, help='上下文token数量（默认4）')
    parser.add_argument('--ctx-init', type=str, default="a photo of a", help='上下文初始化文本')
    parser.add_argument('--csc', action='store_true', help='使用类别特定的上下文')
    parser.add_argument('--class-token-position', type=str, default='end',
                       choices=['end', 'middle', 'front'],
                       help='类别token位置（默认end）')
    parser.add_argument('--sccm-lambda', type=float, default=1.0, help='SCCM损失权重（默认1.0）')
    parser.add_argument('--kdsp-lambda', type=float, default=1.0, help='KDSP损失权重（默认1.0）')
    parser.add_argument('--tau', type=float, default=1.0, help='用于选择prompt的阈值（默认1.0）')
    parser.add_argument('--n-prompts', type=int, default=4, help='使用的prompt数量（默认4）')
    parser.add_argument('--use-focal-loss', action='store_true', help='使用 Focal Loss')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal Loss alpha 参数')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal Loss gamma 参数')
    parser.add_argument('--freeze-image-encoder', action='store_true', help='冻结图像编码器')
    
    # 损失权重参数
    parser.add_argument('--classification-loss-weight', type=float, default=0.5,
                       help='分类损失权重（默认0.5）')
    parser.add_argument('--contrastive-loss-weight', type=float, default=0.5,
                       help='对比损失权重（默认0.5）')
    parser.add_argument('--distillation-loss-weight', type=float, default=0.0,
                       help='蒸馏损失权重（默认0.0，不使用蒸馏）')
    
    # Hybrid 模型特定参数
    parser.add_argument('--use-original-clip-resnet50', action='store_true',
                       help='使用原始 CLIP 的 ResNet50 而不是 PMC-CLIP 的 ResNet50（仅对 hybrid 模型有效）')
    
    args = parser.parse_args()
    
    train_biomedcoop_cross_validation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        class_texts_file=args.class_texts_file,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        augmentation=args.augmentation,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        gpu_id=args.gpu_id,
        n_splits=args.n_splits,
        random_state=args.random_state,
        save_best=True,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_monitor=args.early_stopping_monitor,
        use_weighted_sampling=args.use_weighted_sampling,
        weight_method=args.weight_method,
        weight_smooth_factor=args.weight_smooth_factor,
        model_type=args.model_type,
        clip_backbone=args.clip_backbone,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        csc=args.csc,
        class_token_position=args.class_token_position,
        sccm_lambda=args.sccm_lambda,
        kdsp_lambda=args.kdsp_lambda,
        tau=args.tau,
        n_prompts=args.n_prompts,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        freeze_image_encoder=args.freeze_image_encoder,
        classification_loss_weight=args.classification_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        distillation_loss_weight=args.distillation_loss_weight,
        use_original_clip_resnet50=args.use_original_clip_resnet50,
    )


if __name__ == "__main__":
    main()
