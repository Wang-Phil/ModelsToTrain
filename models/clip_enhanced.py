"""
增强版CLIP模型 - 支持多种损失函数融合
融合了biomedcoop的多种损失：分类损失、对比损失、SCCM损失、KDSP损失

基于clip.py，添加了：
1. 分类损失（Classification Loss）
2. 对比损失（Contrastive Loss）- 原有功能
3. SCCM损失（Semantic Consistency Constraint）- 比较学习的文本特征与类别文本描述
4. KDSP损失（Knowledge Distillation with Soft Predictions）- 知识蒸馏

使用方法：
    from clip_enhanced import EnhancedCLIPModel
    
    model = EnhancedCLIPModel(
        image_encoder_name='resnet50',
        text_encoder_name='pubmedbert',
        embed_dim=512,
        temperature=0.07,
        class_texts=None,  # 类别文本描述列表（用于SCCM损失）
        class_texts_file=None,  # 或从JSON文件加载类别文本描述
        teacher_model=None,  # Teacher模型（用于KDSP损失）
        use_classification_loss=True,
        use_contrastive_loss=True,
        use_sccm_loss=False,
        use_kdsp_loss=False,
        classification_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        sccm_loss_weight=1.0,
        kdsp_loss_weight=1.0
    )
"""

import copy
import os.path as osp
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

# 导入基础CLIP模型组件
# 注意：需要确保clip.py在同一目录下
try:
    from clip import ImageEncoder, TextEncoder, CLIP_AVAILABLE
except ImportError:
    # 如果导入失败，尝试相对导入
    from .clip import ImageEncoder, TextEncoder, CLIP_AVAILABLE

# 如果设置了镜像环境变量，在导入transformers之前设置
if 'HF_ENDPOINT' not in os.environ:
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


class EnhancedCLIPModel(nn.Module):
    """
    增强版CLIP模型 - 支持多种损失函数
    
    支持的损失函数：
    1. Classification Loss: 直接优化分类准确性
    2. Contrastive Loss: 保持图像-文本对齐
    3. SCCM Loss: 语义一致性约束（需要类别文本描述）
    4. KDSP Loss: 知识蒸馏（需要teacher模型）
    """
    
    def __init__(
        self,
        image_encoder_name='resnet50',
        text_encoder_name='pubmedbert',
        embed_dim=512,
        temperature=0.07,
        class_texts=None,
        class_texts_file=None,
        teacher_model=None,
        use_classification_loss=True,
        use_contrastive_loss=True,
        use_sccm_loss=False,
        use_kdsp_loss=False,
        classification_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        sccm_loss_weight=1.0,
        kdsp_loss_weight=1.0,
        logit_scale=None
    ):
        """
        Args:
            image_encoder_name: 图像编码器名称
            text_encoder_name: 文本编码器名称
            embed_dim: embedding维度
            temperature: 温度参数（如果为None，则使用logit_scale）
            class_texts: 类别文本描述列表（用于SCCM损失），例如 ['class1 description', 'class2 description']
            class_texts_file: 类别文本描述JSON文件路径（替代class_texts）
            teacher_model: Teacher模型（用于KDSP损失），应该是冻结的CLIP模型
            use_classification_loss: 是否使用分类损失
            use_contrastive_loss: 是否使用对比损失
            use_sccm_loss: 是否使用SCCM损失
            use_kdsp_loss: 是否使用KDSP损失
            classification_loss_weight: 分类损失权重
            contrastive_loss_weight: 对比损失权重
            sccm_loss_weight: SCCM损失权重
            kdsp_loss_weight: KDSP损失权重
            logit_scale: logit scale参数（如果为None，则使用temperature作为初始值）
        """
        super(EnhancedCLIPModel, self).__init__()
        self.embed_dim = embed_dim
        self.use_classification_loss = use_classification_loss
        self.use_contrastive_loss = use_contrastive_loss
        self.use_sccm_loss = use_sccm_loss
        self.use_kdsp_loss = use_kdsp_loss
        self.classification_loss_weight = classification_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.sccm_loss_weight = sccm_loss_weight
        self.kdsp_loss_weight = kdsp_loss_weight
        
        # 设置logit_scale（用于缩放相似度）
        if logit_scale is not None:
            if isinstance(logit_scale, torch.Tensor):
                self.logit_scale = logit_scale
            else:
                self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        else:
            # 使用temperature初始化logit_scale
            # CLIP通常使用logit_scale = exp(t)，其中t是可学习参数
            init_value = math.log(1.0 / temperature) if temperature > 0 else 0.0
            self.logit_scale = nn.Parameter(torch.tensor(init_value))
        
        # 图像和文本编码器
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_name,
            embed_dim=embed_dim
        )
        
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
            embed_dim=embed_dim
        )
        
        # ========== 类别文本特征预编码（用于分类和SCCM损失）==========
        self.class_texts = class_texts
        self.class_text_features = None  # 预编码的类别文本特征 [n_cls, embed_dim]
        self.class_text_description_features = None  # 用于SCCM的类别文本描述特征 [n_cls, embed_dim]
        
        if class_texts is not None:
            # 从列表加载类别文本
            self._load_class_texts_from_list(class_texts)
        elif class_texts_file is not None:
            # 从JSON文件加载类别文本
            self._load_class_texts_from_file(class_texts_file)
        
        # ========== Teacher模型（用于KDSP损失）==========
        self.teacher_model = teacher_model
        if teacher_model is not None:
            # 冻结teacher模型
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
        
        # ========== 打印配置信息 ==========
        print("=" * 80)
        print("Enhanced CLIP Model Configuration:")
        print(f"  Image Encoder: {image_encoder_name}")
        print(f"  Text Encoder: {text_encoder_name}")
        print(f"  Embed Dim: {embed_dim}")
        print(f"  Loss Functions:")
        print(f"    - Classification Loss: {use_classification_loss} (weight={classification_loss_weight})")
        print(f"    - Contrastive Loss: {use_contrastive_loss} (weight={contrastive_loss_weight})")
        print(f"    - SCCM Loss: {use_sccm_loss} (weight={sccm_loss_weight})")
        print(f"    - KDSP Loss: {use_kdsp_loss} (weight={kdsp_loss_weight})")
        if self.class_text_features is not None:
            print(f"  Pre-encoded Class Text Features: {self.class_text_features.shape}")
        if self.teacher_model is not None:
            print(f"  Teacher Model: Enabled (for KDSP loss)")
        print("=" * 80)
    
    def _load_class_texts_from_list(self, class_texts):
        """从列表加载类别文本并预编码"""
        if not isinstance(class_texts, list) or len(class_texts) == 0:
            print("Warning: class_texts is empty or not a list, skipping pre-encoding")
            return
        
        print(f"Pre-encoding {len(class_texts)} class texts...")
        self.eval()
        with torch.no_grad():
            # 预编码类别文本特征（用于分类）
            self.class_text_features = self.text_encoder(texts=class_texts)  # [n_cls, embed_dim]
        
        # 用于SCCM的类别文本描述特征（默认与类别文本特征相同，后续可以通过JSON文件覆盖）
        self.class_text_description_features = self.class_text_features.clone()
        print(f"✓ Pre-encoded {len(class_texts)} class text features: {self.class_text_features.shape}")
    
    def _load_class_texts_from_file(self, class_texts_file):
        """从JSON文件加载类别文本描述并预编码"""
        if not osp.exists(class_texts_file):
            print(f"Warning: Class texts file not found: {class_texts_file}")
            return
        
        print(f"Loading class texts from: {class_texts_file}")
        with open(class_texts_file, 'r', encoding='utf-8') as f:
            class_texts_dict = json.load(f)
        
        # 提取类别名称和描述
        class_names = []
        class_descriptions = []
        for key, value in class_texts_dict.items():
            class_names.append(key)
            # 格式: "类别名: 描述文本"
            class_descriptions.append(f"{key}: {value}")
        
        # 预编码类别名称（用于分类）
        print(f"Pre-encoding {len(class_names)} class names...")
        self.eval()
        with torch.no_grad():
            self.class_text_features = self.text_encoder(texts=class_names)  # [n_cls, embed_dim]
        
        # 预编码类别描述（用于SCCM）
        if self.use_sccm_loss:
            print(f"Pre-encoding {len(class_descriptions)} class descriptions for SCCM loss...")
            with torch.no_grad():
                self.class_text_description_features = self.text_encoder(texts=class_descriptions)  # [n_cls, embed_dim]
        else:
            self.class_text_description_features = self.class_text_features.clone()
        
        self.class_texts = class_names  # 保存类别名称
        print(f"✓ Loaded {len(class_names)} classes from {class_texts_file}")
        print(f"  Class text features: {self.class_text_features.shape}")
        if self.use_sccm_loss:
            print(f"  Class description features: {self.class_text_description_features.shape}")
    
    def forward(self, images, labels=None, class_texts=None, return_features=False):
        """
        前向传播
        
        Args:
            images: 图像tensor [batch_size, 3, H, W]
            labels: 标签 [batch_size]（训练时需要）
            class_texts: 类别文本列表（可选，如果未提供则使用预编码的特征）
            return_features: 是否返回特征（用于调试）
        
        Returns:
            训练模式 (labels不为None):
                (logits, loss_dict)
                其中 loss_dict 包含:
                - classification_loss (如果启用)
                - contrastive_loss (如果启用)
                - sccm_loss (如果启用)
                - kdsp_loss (如果启用)
                - total_loss
            
            评估模式 (labels为None):
                logits [batch_size, n_cls]
        """
        device = images.device
        
        # 编码图像
        image_features = self.image_encoder(images)  # [batch_size, embed_dim]
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 获取logit_scale
        if isinstance(self.logit_scale, nn.Parameter) or isinstance(self.logit_scale, torch.Tensor):
            logit_scale = self.logit_scale.exp() if self.logit_scale.requires_grad else torch.exp(self.logit_scale)
        else:
            logit_scale = torch.exp(torch.tensor(self.logit_scale, device=device))
        
        # 获取类别文本特征
        if class_texts is not None:
            # 使用提供的类别文本
            class_text_features = self.text_encoder(texts=class_texts)  # [n_cls, embed_dim]
            class_text_features = class_text_features / (class_text_features.norm(dim=-1, keepdim=True) + 1e-8)
        elif self.class_text_features is not None:
            # 使用预编码的类别文本特征
            class_text_features = self.class_text_features.to(device)
            class_text_features = class_text_features / (class_text_features.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            raise ValueError("Either class_texts must be provided in forward() or class_text_features must be pre-encoded in __init__()")
        
        # 计算logits
        logits = logit_scale * image_features @ class_text_features.t()  # [batch_size, n_cls]
        logits = torch.clamp(logits, min=-100, max=100)  # 防止softmax溢出
        
        if self.training and labels is not None:
            # ========== 训练模式：计算损失 ==========
            loss_dict = {}
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_size = image_features.shape[0]
            
            # 1. 分类损失
            if self.use_classification_loss:
                loss_ce = F.cross_entropy(logits, labels)
                if torch.isnan(loss_ce):
                    print("Warning: NaN detected in classification loss, using 0")
                    loss_ce = torch.tensor(0.0, device=device, requires_grad=True)
                loss_dict['classification_loss'] = loss_ce
                total_loss = total_loss + self.classification_loss_weight * loss_ce
            
            # 2. 对比损失
            if self.use_contrastive_loss:
                # 为每个样本选择对应的类别文本特征
                batch_text_features = class_text_features[labels]  # [batch_size, embed_dim]
                batch_text_features = batch_text_features / (batch_text_features.norm(dim=-1, keepdim=True) + 1e-8)
                
                # 计算相似度矩阵
                logits_per_image = logit_scale * image_features @ batch_text_features.t()
                logits_per_text = logit_scale * batch_text_features @ image_features.t()
                
                # 创建对比学习标签（对角线匹配）
                contrastive_labels = torch.arange(batch_size, device=device)
                
                # 双向对比损失
                contrastive_loss = (
                    F.cross_entropy(logits_per_image, contrastive_labels) +
                    F.cross_entropy(logits_per_text, contrastive_labels)
                ) / 2
                
                if torch.isnan(contrastive_loss):
                    print("Warning: NaN detected in contrastive loss, using 0")
                    contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                loss_dict['contrastive_loss'] = contrastive_loss
                total_loss = total_loss + self.contrastive_loss_weight * contrastive_loss
            
            # 3. SCCM损失（语义一致性约束）
            # SCCM损失比较类别文本特征（从class_texts编码）与类别文本描述特征（从JSON描述编码）
            # 这确保模型学习的文本表示与语义描述保持一致
            if self.use_sccm_loss and self.class_text_description_features is not None:
                # class_text_features: 从类别名称或提供的文本编码得到的特征 [n_cls, embed_dim]
                # class_text_description_features: 从类别描述（JSON）编码得到的特征 [n_cls, embed_dim]
                # 直接比较所有类别的特征（与biomedcoop的逻辑一致）
                class_text_features_norm = class_text_features / (class_text_features.norm(dim=-1, keepdim=True) + 1e-8)
                description_features_norm = self.class_text_description_features.to(device) / (
                    self.class_text_description_features.to(device).norm(dim=-1, keepdim=True) + 1e-8
                )
                
                # 计算MSE损失（所有类别）
                loss_sccm = F.mse_loss(class_text_features_norm, description_features_norm)
                
                if torch.isnan(loss_sccm):
                    print("Warning: NaN detected in SCCM loss, using 0")
                    loss_sccm = torch.tensor(0.0, device=device, requires_grad=True)
                
                loss_dict['sccm_loss'] = loss_sccm
                total_loss = total_loss + self.sccm_loss_weight * loss_sccm
            
            # 4. KDSP损失（知识蒸馏）
            if self.use_kdsp_loss and self.teacher_model is not None:
                # Student logits（当前模型的logits）
                student_logits = logits
                
                # Teacher logits（零样本logits）
                with torch.no_grad():
                    # 获取teacher的图像特征
                    teacher_image_features = self.teacher_model.image_encoder(images)
                    teacher_image_features = teacher_image_features / (teacher_image_features.norm(dim=-1, keepdim=True) + 1e-8)
                    
                    # 获取teacher的类别文本特征
                    if hasattr(self.teacher_model, 'class_text_features') and self.teacher_model.class_text_features is not None:
                        teacher_class_text_features = self.teacher_model.class_text_features.to(device)
                    else:
                        # 如果teacher没有预编码的特征，使用当前模型的类别文本特征
                        teacher_class_text_features = class_text_features
                    
                    teacher_class_text_features = teacher_class_text_features / (teacher_class_text_features.norm(dim=-1, keepdim=True) + 1e-8)
                    
                    # 获取teacher的logit_scale
                    if hasattr(self.teacher_model, 'logit_scale'):
                        if isinstance(self.teacher_model.logit_scale, nn.Parameter) or isinstance(self.teacher_model.logit_scale, torch.Tensor):
                            teacher_logit_scale = self.teacher_model.logit_scale.exp() if self.teacher_model.logit_scale.requires_grad else torch.exp(self.teacher_model.logit_scale)
                        else:
                            teacher_logit_scale = torch.exp(torch.tensor(self.teacher_model.logit_scale, device=device))
                    else:
                        teacher_logit_scale = logit_scale
                    
                    # 计算teacher logits
                    teacher_logits = teacher_logit_scale * teacher_image_features @ teacher_class_text_features.t()
                    teacher_logits = torch.clamp(teacher_logits, min=-100, max=100)
                
                # 计算KL散度损失
                log_probs_student = F.log_softmax(student_logits, dim=1)
                log_probs_teacher = F.log_softmax(teacher_logits, dim=1)
                
                if torch.isnan(log_probs_student).any() or torch.isnan(log_probs_teacher).any():
                    print("Warning: NaN detected in log_probs for KDSP loss, using 0")
                    loss_kdsp = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    loss_kdsp = F.kl_div(
                        log_probs_student,
                        log_probs_teacher,
                        reduction='batchmean',
                        log_target=True
                    )
                    
                    if torch.isnan(loss_kdsp):
                        print("Warning: NaN detected in KDSP loss, using 0")
                        loss_kdsp = torch.tensor(0.0, device=device, requires_grad=True)
                
                loss_dict['kdsp_loss'] = loss_kdsp
                total_loss = total_loss + self.kdsp_loss_weight * loss_kdsp
            
            loss_dict['total_loss'] = total_loss
            
            if return_features:
                return logits, loss_dict, image_features, class_text_features
            else:
                return logits, loss_dict
        else:
            # ========== 评估模式：只返回logits ==========
            if return_features:
                return logits, image_features, class_text_features
            else:
                return logits
    
    def predict(self, images, class_texts=None, return_probs=False):
        """
        预测图像的类别
        
        Args:
            images: 图像tensor [batch_size, 3, H, W]
            class_texts: 类别文本列表（可选）
            return_probs: 是否返回概率
        
        Returns:
            predictions: 预测的类别索引 [batch_size]
            probabilities: 每个类别的概率 [batch_size, n_cls]（如果return_probs=True）
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, labels=None, class_texts=class_texts)
            predictions = torch.argmax(logits, dim=1)
            
            if return_probs:
                probabilities = F.softmax(logits, dim=1)
                return predictions, probabilities
            else:
                return predictions


def create_enhanced_model(config):
    """根据配置创建增强版CLIP模型"""
    # 提取基础配置
    image_encoder_name = config.get('image_encoder', 'resnet50')
    text_encoder_name = config.get('text_encoder', 'pubmedbert')
    embed_dim = config.get('embed_dim', 512)
    temperature = config.get('temperature', 0.07)
    logit_scale = config.get('logit_scale', None)
    
    # 提取类别文本配置
    class_texts = config.get('class_texts', None)
    class_texts_file = config.get('class_texts_file', None)
    
    # 提取teacher模型配置
    teacher_model = config.get('teacher_model', None)
    
    # 提取损失函数配置
    use_classification_loss = config.get('use_classification_loss', True)
    use_contrastive_loss = config.get('use_contrastive_loss', True)
    use_sccm_loss = config.get('use_sccm_loss', False)
    use_kdsp_loss = config.get('use_kdsp_loss', False)
    classification_loss_weight = config.get('classification_loss_weight', 1.0)
    contrastive_loss_weight = config.get('contrastive_loss_weight', 1.0)
    sccm_loss_weight = config.get('sccm_loss_weight', 1.0)
    kdsp_loss_weight = config.get('kdsp_loss_weight', 1.0)
    
    model = EnhancedCLIPModel(
        image_encoder_name=image_encoder_name,
        text_encoder_name=text_encoder_name,
        embed_dim=embed_dim,
        temperature=temperature,
        class_texts=class_texts,
        class_texts_file=class_texts_file,
        teacher_model=teacher_model,
        use_classification_loss=use_classification_loss,
        use_contrastive_loss=use_contrastive_loss,
        use_sccm_loss=use_sccm_loss,
        use_kdsp_loss=use_kdsp_loss,
        classification_loss_weight=classification_loss_weight,
        contrastive_loss_weight=contrastive_loss_weight,
        sccm_loss_weight=sccm_loss_weight,
        kdsp_loss_weight=kdsp_loss_weight,
        logit_scale=logit_scale
    )
    
    return model

