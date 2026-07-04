"""
PMC-CLIP 完整模型：使用 PMC-CLIP 的图像编码器 + 文本编码器
支持分类损失、对比损失、蒸馏损失（使用 BiomedCLIP 图像编码器作为 teacher）
"""

import os
import os.path as osp
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
from tqdm import tqdm

# 导入 PMC-CLIP 的 ModifiedResNet
import sys
local_clip_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'clip')
if local_clip_path not in sys.path:
    sys.path.insert(0, local_clip_path)

try:
    from clip.pmcclip import ModifiedResNet
except ImportError:
    try:
        from pmcclip import ModifiedResNet
    except ImportError:
        import importlib.util
        pmcclip_path = os.path.join(local_clip_path, 'pmcclip.py')
        if os.path.exists(pmcclip_path):
            spec = importlib.util.spec_from_file_location("pmcclip", pmcclip_path)
            pmcclip_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pmcclip_module)
            ModifiedResNet = pmcclip_module.ModifiedResNet
        else:
            raise ImportError("Cannot find ModifiedResNet. Please ensure clip/pmcclip.py exists.")

from transformers import AutoTokenizer, AutoModel

# Directory where PMC-CLIP files should be located
directory = "clip/checkpoints"

# PMC-CLIP File URLs
files = {
    "text_encoder.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth",
    "image_encoder(resnet50).pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth",
}


def download_file(url, filepath):
    """下载文件"""
    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, "wb") as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))
        print(f"{filepath} downloaded successfully.")
    else:
        print(f"Failed to download {filepath}. HTTP Status Code: {response.status_code}")


class PMCCLIP(nn.Module):
    """PMC-CLIP 基础模型"""
    def __init__(self, image_encoder, text_encoder, projection_layer):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_projection_layer = projection_layer
        self.logit_scale = 4.4292  # PMC-CLIP 默认的 logit_scale
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
    
    def encode_text(self, text):
        """编码文本"""
        if isinstance(text, str):
            text = [text]
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        device = next(self.text_encoder.parameters()).device
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        text_feature = self.text_encoder(input_ids, attention_mask)
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        return text_feature
    
    def encode_image(self, image):
        """编码图像"""
        image_feature = self.image_encoder(image)
        if isinstance(image_feature, dict):
            image_feature = image_feature['image_features']
        return image_feature


class HybridPMCCLIPOnly(nn.Module):
    """
    PMC-CLIP 完整模型：使用 PMC-CLIP 的图像编码器 + 文本编码器
    
    特征维度：
    - 图像编码器输出：768维
    - 文本编码器输出：768维（通过 text_projection_layer）
    - Teacher 图像编码器输出：512维（BiomedCLIP）
    
    训练策略：
    - 训练图像编码器
    - 冻结文本编码器
    - 使用分类损失 + 对比损失 + 蒸馏损失
    """
    def __init__(self, cfg, classnames, pmcclip_model, teacher_model=None):
        super().__init__()
        self.n_cls = len(classnames)
        self.dtype = torch.float32
        
        # ========== PMC-CLIP 模型 ==========
        self.pmcclip_model = pmcclip_model
        self.image_encoder = pmcclip_model.image_encoder
        self.text_encoder = pmcclip_model.text_encoder
        self.text_projection_layer = pmcclip_model.text_projection_layer
        self.tokenizer = pmcclip_model.tokenizer
        self.logit_scale = pmcclip_model.logit_scale
        
        self.image_embed_dim = 768  # PMC-CLIP 图像特征维度
        self.text_embed_dim = 768   # PMC-CLIP 文本特征维度
        
        # ========== Teacher 模型（用于蒸馏）==========
        self.teacher_model = teacher_model
        if teacher_model is not None:
            # 冻结 teacher 模型
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
            self.teacher_image_embed_dim = 512  # BiomedCLIP 图像特征维度
            
            # 蒸馏投影层：将 PMC-CLIP 768维投影到 BiomedCLIP 512维
            self.distill_projection = nn.Linear(self.image_embed_dim, self.teacher_image_embed_dim)
            nn.init.xavier_uniform_(self.distill_projection.weight)
            nn.init.zeros_(self.distill_projection.bias)
        
        # 存储类别名称
        self.classnames = [name.replace("_", " ") for name in classnames]
        
        # 加载类别文本描述
        class_texts_file = getattr(cfg.TRAINER.BIOMEDCOOP, 'CLASS_TEXTS_FILE', None)
        if class_texts_file is None:
            default_path = 'class_texts_hip_prosthesis.json'
            if osp.exists(default_path):
                class_texts_file = default_path
        
        # 为每个类别获取文本描述
        self.class_prompts = []
        if class_texts_file and osp.exists(class_texts_file):
            print(f"加载类别文本描述: {class_texts_file}")
            with open(class_texts_file, 'r', encoding='utf-8') as f:
                class_texts_dict = json.load(f)
            
            for classname in self.classnames:
                description = None
                for key, value in class_texts_dict.items():
                    if (key.lower() == classname.lower() or
                        key.replace(" ", "_").lower() == classname.replace(" ", "_").lower() or
                        key.replace("_", " ").lower() == classname.lower()):
                        description = value
                        break
                
                if description is None:
                    print(f"警告: 未找到类别 '{classname}' 的文本描述，使用默认 prompt")
                    description = f"a photo of {classname}."
                else:
                    description = f"{classname}: {description}"
                    print(f"  {classname}: 使用文本描述")
                
                self.class_prompts.append(description)
        else:
            print(f"警告: 未找到类别文本描述文件，使用默认 prompt")
            self.class_prompts = [f"a photo of {name}." for name in self.classnames]
        
        # 预编码所有类别的文本特征
        print(f"预编码 {len(self.class_prompts)} 个类别的文本特征（使用 PMC-CLIP）...")
        with torch.no_grad():
            device = next(self.text_encoder.parameters()).device
            class_text_features_list = []
            for i, prompt in enumerate(self.class_prompts):
                text_feat = self.pmcclip_model.encode_text(prompt)  # [1, 768]
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                class_text_features_list.append(text_feat)
            print(f"✓ 完成文本特征预编码")
            self.register_buffer('class_text_features', torch.cat(class_text_features_list, dim=0))  # [n_cls, 768]
        
        # 损失函数权重配置
        self.classification_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'CLASSIFICATION_LOSS_WEIGHT', 0.5)
        self.contrastive_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'CONTRASTIVE_LOSS_WEIGHT', 0.5)
        self.distillation_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'DISTILLATION_LOSS_WEIGHT', 0.0)
        
        print(f"损失权重: 分类={self.classification_loss_weight}, 对比={self.contrastive_loss_weight}, 蒸馏={self.distillation_loss_weight}")
    
    def forward(self, image, label=None):
        """
        前向传播
        
        Args:
            image: 图像tensor [batch_size, 3, H, W]
            label: 标签 [batch_size]
        
        Returns:
            训练模式: (logits, loss_ce, contrastive_loss, loss_distill)
            评估模式: logits
        """
        device = image.device
        eps = 1e-8
        
        # 限制 logit_scale 范围
        logit_scale_clamped = min(self.logit_scale, 4.6052)
        logit_scale = math.exp(logit_scale_clamped)
        
        # 获取图像特征
        image_features = self.image_encoder(image.type(self.dtype))
        if isinstance(image_features, dict):
            image_features = image_features['image_features']
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + eps)  # [batch_size, 768]
        
        # 检查 NaN
        if torch.isnan(image_features).any():
            print("Warning: NaN detected in image_features")
            image_features = torch.nan_to_num(image_features, nan=0.0)
        
        # 获取文本特征
        text_features = self.class_text_features  # [n_cls, 768]
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + eps)
        
        # 计算 logits
        logits = logit_scale * image_features @ text_features.t()  # [batch_size, n_cls]
        logits = torch.clamp(logits, min=-100, max=100)
        
        if self.training and label is not None:
            batch_size = image_features.shape[0]
            
            # 1. 分类损失
            if self.classification_loss_weight > 0:
                loss_ce = F.cross_entropy(logits, label)
                if torch.isnan(loss_ce):
                    print("Warning: NaN in loss_ce")
                    loss_ce = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss_ce = torch.tensor(0.0, device=device)
            
            # 2. 对比损失
            if self.contrastive_loss_weight > 0:
                batch_text_features = text_features[label]  # [batch_size, 768]
                batch_text_features = batch_text_features / (batch_text_features.norm(dim=-1, keepdim=True) + eps)
                
                logits_per_image = logit_scale * image_features @ batch_text_features.t()
                logits_per_text = logit_scale * batch_text_features @ image_features.t()
                logits_per_image = torch.clamp(logits_per_image, min=-100, max=100)
                logits_per_text = torch.clamp(logits_per_text, min=-100, max=100)
                
                contrastive_labels = torch.arange(batch_size, device=device)
                contrastive_loss = (
                    F.cross_entropy(logits_per_image, contrastive_labels) +
                    F.cross_entropy(logits_per_text, contrastive_labels)
                ) / 2
                
                if torch.isnan(contrastive_loss):
                    print("Warning: NaN in contrastive_loss")
                    contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                contrastive_loss = torch.tensor(0.0, device=device)
            
            # 3. 蒸馏损失（如果有 teacher 模型）
            if self.distillation_loss_weight > 0 and self.teacher_model is not None:
                # Student 特征：投影到 teacher 维度
                student_features = self.distill_projection(image_features)  # [batch_size, 512]
                student_features = student_features / (student_features.norm(dim=-1, keepdim=True) + eps)
                
                # Teacher 特征
                with torch.no_grad():
                    teacher_dtype = next(self.teacher_model.parameters()).dtype
                    teacher_features = self.teacher_model.encode_image(image.type(teacher_dtype))
                    if isinstance(teacher_features, dict):
                        teacher_features = teacher_features.get('image_features', teacher_features)
                    teacher_features = teacher_features / (teacher_features.norm(dim=-1, keepdim=True) + eps)
                
                loss_distill = F.mse_loss(student_features, teacher_features)
                
                if torch.isnan(loss_distill):
                    print("Warning: NaN in loss_distill")
                    loss_distill = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss_distill = torch.tensor(0.0, device=device)
            
            return logits, loss_ce, contrastive_loss, loss_distill
        else:
            return logits


def load_pmcclip_model(device='cuda'):
    """加载 PMC-CLIP 模型"""
    # 检查并下载文件
    os.makedirs(directory, exist_ok=True)
    for filename, url in files.items():
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            print(f"{filename} 未找到，正在下载...")
            download_file(url, filepath)
        else:
            print(f"✓ {filename} 已存在")
    
    # 加载图像编码器
    print("加载 PMC-CLIP ResNet50 图像编码器...")
    image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
    image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'), weights_only=True))
    
    # 加载文本编码器
    print("加载 PMC-CLIP 文本编码器...")
    text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
    text_encoder.load_state_dict(torch.load(os.path.join(directory,'text_encoder.pth'), weights_only=True))
    
    # 加载文本投影层
    text_projection_layer = torch.load(os.path.join(directory,'text_projection_layer.pth'), weights_only=True)
    text_projection_layer = nn.Parameter(text_projection_layer)
    
    image_encoder = image_encoder.to(device).eval()
    text_encoder = text_encoder.to(device).eval()
    text_projection_layer = text_projection_layer.to(device)
    
    pmcclip_model = PMCCLIP(image_encoder, text_encoder, text_projection_layer).to(device).eval()
    print("✓ PMC-CLIP 模型加载完成")
    
    return pmcclip_model

