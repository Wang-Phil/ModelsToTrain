"""
MetaCLIP 适配器
将 MetaCLIP 预训练模型集成到现有的 CLIP 训练框架中
支持在医学图像数据集上进行微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 添加 MetaCLIP 路径
METACLIP_PATH = "/home/ln/wangweicheng/MetaCLIP"
if METACLIP_PATH not in sys.path:
    sys.path.insert(0, METACLIP_PATH)

try:
    from src.mini_clip.factory import create_model_and_transforms, get_tokenizer
    from src.mini_clip.model import CLIP
    METACLIP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MetaCLIP not available: {e}")
    print("Please ensure MetaCLIP is installed or set METACLIP_PATH correctly")
    METACLIP_AVAILABLE = False


class MetaCLIPImageEncoder(nn.Module):
    """MetaCLIP 图像编码器适配器"""
    
    def __init__(self, metaclip_model, embed_dim=512):
        """
        Args:
            metaclip_model: MetaCLIP 模型实例
            embed_dim: 目标嵌入维度（如果与 MetaCLIP 不同，需要投影）
        """
        super(MetaCLIPImageEncoder, self).__init__()
        self.metaclip_model = metaclip_model
        self.visual = metaclip_model.visual
        
        # 获取 MetaCLIP 的嵌入维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.visual(dummy_input)
            metaclip_embed_dim = dummy_output.shape[1]
        
        # 如果维度不匹配，添加投影层
        if metaclip_embed_dim != embed_dim:
            self.projection = nn.Linear(metaclip_embed_dim, embed_dim)
        else:
            self.projection = nn.Identity()
        
        self.embed_dim = embed_dim
    
    def forward(self, x):
        """
        Args:
            x: 图像tensor [batch_size, 3, H, W]
        Returns:
            features: 特征向量 [batch_size, embed_dim]
        """
        # 使用 MetaCLIP 的视觉编码器
        x = self.visual(x)
        # 投影到目标维度
        x = self.projection(x)
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x


class MetaCLIPTextEncoder(nn.Module):
    """MetaCLIP 文本编码器适配器"""
    
    def __init__(self, metaclip_model, embed_dim=512):
        """
        Args:
            metaclip_model: MetaCLIP 模型实例
            embed_dim: 目标嵌入维度（如果与 MetaCLIP 不同，需要投影）
        """
        super(MetaCLIPTextEncoder, self).__init__()
        self.metaclip_model = metaclip_model
        self.transformer = metaclip_model.transformer
        
        # 获取 MetaCLIP 的嵌入维度
        # MetaCLIP 的文本编码器输出维度在 text_projection 中
        if hasattr(metaclip_model, 'text_projection'):
            metaclip_embed_dim = metaclip_model.text_projection.shape[1]
        else:
            # 默认使用视觉编码器的维度
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = metaclip_model.visual(dummy_input)
                metaclip_embed_dim = dummy_output.shape[1]
        
        # 如果维度不匹配，添加投影层
        if metaclip_embed_dim != embed_dim:
            self.projection = nn.Linear(metaclip_embed_dim, embed_dim)
        else:
            self.projection = nn.Identity()
        
        self.embed_dim = embed_dim
        self.token_embedding = metaclip_model.token_embedding
        self.positional_embedding = metaclip_model.positional_embedding
        self.ln_final = metaclip_model.ln_final
        self.text_projection = metaclip_model.text_projection if hasattr(metaclip_model, 'text_projection') else None
    
    def forward(self, input_ids=None, texts=None):
        """
        Args:
            input_ids: tokenized input ids (可选)
            texts: 原始文本列表（可选，如果提供 input_ids 则不需要）
        Returns:
            features: 文本特征 [batch_size, embed_dim]
        """
        if texts is not None:
            # 如果提供了原始文本，需要先 tokenize
            # MetaCLIP 使用特定的 tokenizer
            from src.mini_clip.tokenizer import tokenize
            input_ids = tokenize(texts)
        
        if input_ids is None:
            raise ValueError("Either input_ids or texts must be provided")
        
        device = next(self.transformer.parameters()).device
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).to(device)
        elif not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(device)
        
        # MetaCLIP 的文本编码流程
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        
        # 取最后一个非padding token的表示
        # CLIP 使用最后一个非padding token
        x = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)]
        
        # 应用 text_projection（如果存在）
        if self.text_projection is not None:
            x = x @ self.text_projection
        
        # 投影到目标维度
        x = self.projection(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x


class MetaCLIPAdapter(nn.Module):
    """MetaCLIP 适配器 - 将 MetaCLIP 模型适配到现有 CLIP 训练框架"""
    
    def __init__(
        self,
        model_name='ViT-B-32-quickgelu',
        pretrained='metaclip_400m',
        embed_dim=512,
        temperature=0.07,
        device='cpu'
    ):
        """
        Args:
            model_name: MetaCLIP 模型名称，例如 'ViT-B-32-quickgelu' 或 'ViT-H-14-quickgelu-worldwide@WorldWideCLIP'
            pretrained: 预训练权重标识，例如 'metaclip_400m', 'metaclip2_worldwide'
            embed_dim: 目标嵌入维度
            temperature: 温度参数
            device: 设备
        """
        super(MetaCLIPAdapter, self).__init__()
        
        if not METACLIP_AVAILABLE:
            raise ImportError("MetaCLIP is not available. Please install MetaCLIP or set METACLIP_PATH correctly.")
        
        self.embed_dim = embed_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.model_name = model_name
        self.pretrained = pretrained
        
        # 加载 MetaCLIP 模型
        print(f"Loading MetaCLIP model: {model_name}, pretrained: {pretrained}")
        self.metaclip_model, _, _ = create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            precision='fp32',
            device=torch.device(device)
        )
        
        # 创建适配的图像和文本编码器
        self.image_encoder = MetaCLIPImageEncoder(self.metaclip_model, embed_dim=embed_dim)
        self.text_encoder = MetaCLIPTextEncoder(self.metaclip_model, embed_dim=embed_dim)
        
        # 保存 tokenizer 引用（如果需要）
        self._tokenizer = None
    
    def get_tokenizer(self):
        """获取 tokenizer"""
        if self._tokenizer is None:
            # 根据模型类型选择 tokenizer
            if 'worldwide' in self.model_name.lower():
                # MetaCLIP 2 使用 XLM-V tokenizer
                self._tokenizer = get_tokenizer("facebook/xlm-v-base")
            else:
                # MetaCLIP 1 使用标准 CLIP tokenizer
                self._tokenizer = get_tokenizer(None)
        return self._tokenizer
    
    def forward(self, images, texts=None, text_features=None):
        """
        Forward pass
        Args:
            images: 图像tensor [batch_size, 3, H, W]
            texts: 文本列表（可选，如果提供text_features则不需要）
            text_features: 预计算的文本特征 [num_classes, embed_dim] 或 [batch_size, embed_dim]（可选）
        Returns:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [num_classes, embed_dim] 或 [batch_size, embed_dim]
        """
        # 编码图像
        image_features = self.image_encoder(images)
        
        # 编码文本
        if text_features is None:
            if texts is None:
                raise ValueError("Either texts or text_features must be provided")
            
            # 直接传递文本列表给 text_encoder，它会内部处理 tokenization
            text_features = self.text_encoder(texts=texts)
        
        return image_features, text_features
    
    def compute_similarity(self, image_features, text_features):
        """
        计算图像特征和文本特征的相似度
        Args:
            image_features: [batch_size, embed_dim]
            text_features: [num_classes, embed_dim] 或 [batch_size, embed_dim]
        Returns:
            similarity: [batch_size, num_classes] 或 [batch_size, batch_size]
        """
        # 计算余弦相似度（已经归一化，所以直接矩阵乘法）
        similarity = image_features @ text_features.T  # [batch_size, num_classes]
        
        # 应用温度参数
        similarity = similarity / self.temperature
        return similarity
    
    def predict(self, images, class_texts=None, text_features=None):
        """
        预测图像的类别
        Args:
            images: 图像tensor [batch_size, 3, H, W]
            class_texts: 类别文本描述列表（可选，如果提供text_features则不需要）
            text_features: 预计算的类别文本特征 [num_classes, embed_dim]（可选，如果提供则不需要class_texts）
        Returns:
            predictions: 预测的类别索引 [batch_size]
            probabilities: 每个类别的概率 [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            # 编码图像
            image_features = self.image_encoder(images)
            
            # 编码所有类别的文本描述（如果未提供预计算的文本特征）
            if text_features is None:
                if class_texts is None:
                    raise ValueError("Either class_texts or text_features must be provided")
                
                # 直接传递文本列表给 text_encoder
                text_features = self.text_encoder(texts=class_texts)
            
            # 计算相似度
            similarity = self.compute_similarity(image_features, text_features)
            
            # 转换为概率
            probabilities = F.softmax(similarity, dim=1)
            
            # 获取预测类别
            predictions = torch.argmax(similarity, dim=1)
        
        return predictions, probabilities
    
    def precompute_class_text_features(self, class_texts):
        """
        预计算所有类别的文本特征（用于推理加速）
        Args:
            class_texts: 类别文本描述列表
        Returns:
            text_features: 预计算的文本特征 [num_classes, embed_dim]
        """
        self.eval()
        with torch.no_grad():
            text_features = self.text_encoder(texts=class_texts)
        return text_features


def create_metaclip_model(
    model_name='ViT-B-32-quickgelu',
    pretrained='metaclip_400m',
    embed_dim=512,
    temperature=0.07,
    device='cpu'
):
    """
    创建 MetaCLIP 适配模型
    
    Args:
        model_name: MetaCLIP 模型名称
            - MetaCLIP 1: 'ViT-B-32-quickgelu', 'ViT-B-16-quickgelu', 'ViT-L-14-quickgelu', 'ViT-H-14-quickgelu'
            - MetaCLIP 2: 'ViT-H-14-quickgelu-worldwide@WorldWideCLIP', 'ViT-B-32-worldwide@WorldWideCLIP'
        pretrained: 预训练权重标识
            - MetaCLIP 1: 'metaclip_400m', 'metaclip_2_5b'
            - MetaCLIP 2: 'metaclip2_worldwide'
        embed_dim: 嵌入维度
        temperature: 温度参数
        device: 设备
    
    Returns:
        model: MetaCLIPAdapter 实例
    """
    return MetaCLIPAdapter(
        model_name=model_name,
        pretrained=pretrained,
        embed_dim=embed_dim,
        temperature=temperature,
        device=device
    )

