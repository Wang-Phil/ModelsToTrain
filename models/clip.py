"""
CLIP风格的医学图像分类模型
实现图像-文本对齐的零样本分类

模型架构说明：
- 本模型采用CLIP（Contrastive Language-Image Pre-training）的对比学习架构思想
- 图像编码器：使用预训练的视觉模型（ResNet/ViT/EfficientNet/ConvNeXt/StarNet等）作为backbone
- 文本编码器：使用预训练的语言模型（BERT中文或CLIP文本编码器）作为backbone
- 投影层：将图像和文本特征投影到统一的嵌入空间（embed_dim）
- 温度参数：可学习的温度参数用于对比学习中的相似度缩放
- 训练方式：通过图像-文本对比学习进行端到端微调，学习跨模态对齐

注意：这不是直接使用OpenAI的预训练CLIP模型，而是采用CLIP架构思想，
使用独立的预训练编码器组合，并通过对比学习进行微调。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# 如果设置了镜像环境变量，在导入transformers之前设置
if 'HF_ENDPOINT' not in os.environ:
    # 默认使用镜像站点（如果无法访问 Hugging Face）
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from transformers import AutoModel, AutoTokenizer
from torchvision import models

# CLIP 是可选的，只在需要时导入
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# 导入 starnet_dual_pyramid_rcf 模型
try:
    from .starnet_dual_pyramid_rcf import StarNet_DualPyramid_RCF, starnet_dual_pyramid_rcf
    STARNET_DUAL_PYRAMID_AVAILABLE = True
except ImportError:
    try:
        from starnet_dual_pyramid_rcf import StarNet_DualPyramid_RCF, starnet_dual_pyramid_rcf
        STARNET_DUAL_PYRAMID_AVAILABLE = True
    except ImportError:
        STARNET_DUAL_PYRAMID_AVAILABLE = False

# 导入标准 StarNet 模型
try:
    from .starnet import StarNet, starnet_s1, starnet_s2, starnet_s3, starnet_s4, starnet_s050, starnet_s100, starnet_s150
    STARNET_AVAILABLE = True
except ImportError:
    try:
        from starnet import StarNet, starnet_s1, starnet_s2, starnet_s3, starnet_s4, starnet_s050, starnet_s100, starnet_s150
        STARNET_AVAILABLE = True
    except ImportError:
        STARNET_AVAILABLE = False


class StarNetFeatureExtractor(nn.Module):
    """StarNet Dual-Pyramid RCF 特征提取器 - 移除分类层，仅提取特征"""
    
    def __init__(self, base=24, depths=[2, 2, 8, 3], global_depths=[1, 1, 1, 1], 
                 mlp_ratio=4, drop_path=0.1, use_attn=0, dropout_rate=0.1, num_classes=1000):
        super(StarNetFeatureExtractor, self).__init__()
        
        if not STARNET_DUAL_PYRAMID_AVAILABLE:
            raise ImportError("starnet_dual_pyramid_rcf 模型不可用。请确保 starnet_dual_pyramid_rcf.py 在同一目录下。")
        
        # 创建完整的 StarNet 模型
        self.model = StarNet_DualPyramid_RCF(
            base=base,
            depths=depths,
            num_classes=num_classes,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            global_depths=global_depths,
            use_attn=use_attn,
            dropout_rate=dropout_rate
        )
        
        # 移除分类层，保留特征提取部分
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        提取特征
        Args:
            x: 输入图像 [batch_size, 3, H, W]
        Returns:
            features: 特征向量 [batch_size, feature_dim]
        """
        # 复制 StarNet 的前向传播逻辑，但不进行最终分类
        # 1. Global Pyramid 独立运行
        G = self.model.global_pyr(x)
        
        # 2. Local Pyramid 初始化
        L_current = self.model.local.stem(x)
        fused_features = []
        
        # 3. 逐层迭代，执行 RCF 逻辑
        for i in range(4):
            # Downsample
            L_proj = self.model.local.downsamples[i](L_current)
            
            # Local Pyramid Stage i 的输出
            L_i_out = self.model.local.blocks_list[i](L_proj)
            
            # 融合特征: F_i = L_i * (1-w) + Adapter(G_i) * w
            A = self.model.adapters[i](G[i])
            w = torch.sigmoid(self.model.fuse_weights[i])
            F_i = L_i_out * (1 - w) + A * w
            
            fused_features.append(F_i)
            
            # 残差级联融合
            if i < 3:
                gamma = self.model.gamma_weights[i]
                L_current = L_i_out + gamma * F_i
            else:
                L_current = L_i_out
        
        # 4. 使用最后一个 Stage 的融合特征
        x = fused_features[-1]
        
        # 5. 池化和归一化（不使用分类层）
        x = self.pool(self.model.norm(x))
        x = torch.flatten(x, 1)
        
        return x


class StandardStarNetFeatureExtractor(nn.Module):
    """标准 StarNet 特征提取器 - 移除分类层，仅提取特征"""
    
    def __init__(self, model_name='starnet_s1', pretrained=False):
        super(StandardStarNetFeatureExtractor, self).__init__()
        
        if not STARNET_AVAILABLE:
            raise ImportError("StarNet 模型不可用。请确保 starnet.py 在同一目录下。")
        
        # 支持的 StarNet 模型映射
        starnet_models = {
            'starnet_s1': starnet_s1,
            'starnet_s2': starnet_s2,
            'starnet_s3': starnet_s3,
            'starnet_s4': starnet_s4,
            'starnet_s050': starnet_s050,
            'starnet_s100': starnet_s100,
            'starnet_s150': starnet_s150,
        }
        
        if model_name not in starnet_models:
            raise ValueError(f"不支持的 StarNet 模型: {model_name}. "
                           f"支持的模型: {list(starnet_models.keys())}")
        
        # 创建模型（移除分类头）
        self.model = starnet_models[model_name](pretrained=pretrained, num_classes=1000)
        
        # 移除分类层，保留特征提取部分
        # StarNet 的结构: stem -> stages -> norm -> avgpool -> head
        # 我们需要保留到 avgpool 之前的部分
        self.stem = self.model.stem
        self.stages = self.model.stages
        self.norm = self.model.norm
        self.avgpool = self.model.avgpool
        
    def forward(self, x):
        """
        提取特征
        Args:
            x: 输入图像 [batch_size, 3, H, W]
        Returns:
            features: 特征向量 [batch_size, feature_dim]
        """
        # 前向传播到分类层之前
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(self.norm(x))
        x = torch.flatten(x, 1)
        return x


class ImageEncoder(nn.Module):
    """图像编码器 - 支持多种预训练模型"""
    
    def __init__(self, model_name='resnet50', embed_dim=512):
        super(ImageEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.model_name = model_name
        
        # ResNet系列
        if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            resnet = getattr(models, model_name)(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            # ResNet的feature dim: resnet18/34是512, resnet50/101/152是2048
            feature_dim = 512 if model_name in ['resnet18', 'resnet34'] else 2048
            self.projection = nn.Linear(feature_dim, embed_dim)
            self.forward_fn = self._forward_resnet
        
        # ViT
        elif model_name == 'vit':
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
            self.projection = nn.Linear(768, embed_dim)
            self.forward_fn = self._forward_vit
        
        # EfficientNet系列
        elif model_name.startswith('efficientnet'):
            try:
                import timm
                self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0移除分类头
                # EfficientNet的feature dim需要根据模型确定
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    dummy_output = self.backbone(dummy_input)
                    feature_dim = dummy_output.shape[1]
                self.projection = nn.Linear(feature_dim, embed_dim)
                self.forward_fn = self._forward_efficientnet
            except ImportError:
                raise ImportError("EfficientNet需要安装timm库: pip install timm")
        
        # ConvNeXt系列
        elif model_name.startswith('convnext'):
            try:
                import timm
                # timm中ConvNeXt的模型名称映射
                convnext_name_map = {
                    'convnext-tiny': 'convnext_tiny',
                    'convnext-small': 'convnext_small', 
                    'convnext-base': 'convnext_base',
                    'convnext-large': 'convnext_large'
                }
                # 如果使用连字符格式，转换为下划线格式
                timm_model_name = convnext_name_map.get(model_name, model_name.replace('-', '_'))
                
                # 尝试创建模型
                try:
                    self.backbone = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
                except RuntimeError:
                    # 如果失败，尝试原始名称
                    self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
                
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    dummy_output = self.backbone(dummy_input)
                    feature_dim = dummy_output.shape[1]
                self.projection = nn.Linear(feature_dim, embed_dim)
                self.forward_fn = self._forward_convnext
            except ImportError:
                raise ImportError("ConvNeXt需要安装timm库: pip install timm")
        
        # StarNet Dual-Pyramid RCF 模型
        elif model_name == 'starnet_dual_pyramid_rcf':
            if not STARNET_DUAL_PYRAMID_AVAILABLE:
                raise ImportError("starnet_dual_pyramid_rcf 模型不可用。请确保 starnet_dual_pyramid_rcf.py 在同一目录下。")
            
            # 创建特征提取包装器
            self.backbone = StarNetFeatureExtractor()
            # 动态获取特征维度
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = self.backbone(dummy_input)
                feature_dim = dummy_output.shape[1]
            self.projection = nn.Linear(feature_dim, embed_dim)
            self.forward_fn = self._forward_starnet
        
        # 标准 StarNet 模型系列
        elif model_name.startswith('starnet_') and model_name != 'starnet_dual_pyramid_rcf':
            if not STARNET_AVAILABLE:
                raise ImportError("StarNet 模型不可用。请确保 starnet.py 在同一目录下。")
            
            # 支持的 StarNet 模型列表
            supported_starnet_models = [
                'starnet_s1', 'starnet_s2', 'starnet_s3', 'starnet_s4',
                'starnet_s050', 'starnet_s100', 'starnet_s150'
            ]
            
            # 检查是否指定了预训练权重（通过 model_name:pretrained 格式）
            pretrained = False
            original_model_name = model_name
            if ':' in model_name:
                model_name, pretrained_str = model_name.split(':', 1)
                pretrained = pretrained_str.lower() in ['true', '1', 'yes', 'pretrained']
            
            if model_name not in supported_starnet_models:
                raise ValueError(f"不支持的 StarNet 模型: {original_model_name}. "
                               f"支持的模型: {supported_starnet_models}")
            
            # 创建特征提取包装器
            self.backbone = StandardStarNetFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained
            )
            
            # 动态获取特征维度
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = self.backbone(dummy_input)
                feature_dim = dummy_output.shape[1]
            self.projection = nn.Linear(feature_dim, embed_dim)
            self.forward_fn = self._forward_starnet
        
        else:
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"支持的模型: resnet18/34/50/101/152, vit, efficientnet-b0~b7, "
                           f"convnext-tiny/small/base/large, starnet_dual_pyramid_rcf, "
                           f"starnet_s1/s2/s3/s4/s050/s100/s150")
        
    def _forward_resnet(self, x):
        """ResNet前向传播"""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.projection(x)
        return x
    
    def _forward_vit(self, x):
        """ViT前向传播"""
        outputs = self.backbone(x)
        x = outputs.last_hidden_state[:, 0]  # CLS token
        x = self.projection(x)
        return x
    
    def _forward_efficientnet(self, x):
        """EfficientNet前向传播"""
        x = self.backbone(x)
        x = self.projection(x)
        return x
    
    def _forward_convnext(self, x):
        """ConvNeXt前向传播"""
        x = self.backbone(x)
        x = self.projection(x)
        return x
    
    def _forward_starnet(self, x):
        """StarNet 前向传播（支持标准 StarNet 和 Dual-Pyramid RCF）"""
        x = self.backbone(x)  # 提取特征 [batch_size, feature_dim]
        x = self.projection(x)  # 投影到 embed_dim
        return x
        
    def forward(self, x):
        x = self.forward_fn(x)
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x


class TextEncoder(nn.Module):
    """文本编码器 - 使用BERT或CLIP的文本编码器"""
    
    def __init__(self, model_name='bert-base-chinese', embed_dim=512):
        super(TextEncoder, self).__init__()
        self.embed_dim = embed_dim
        
        if model_name.startswith('bert'):
            # 使用BERT
            # 检查是否设置了镜像环境变量，如果没有且无法连接，则使用镜像
            import os
            hf_endpoint = os.environ.get('HF_ENDPOINT', None)
            
            # 尝试加载模型
            try:
                self.backbone = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                if 'Connection' in str(e) or 'refused' in str(e).lower():
                    print(f"⚠ 无法连接到 Hugging Face，尝试使用镜像站点...")
                    # 使用 Hugging Face 镜像站点
                    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                    try:
                        # 重新尝试加载
                        self.backbone = AutoModel.from_pretrained(model_name)
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        print("✓ 成功从镜像站点加载模型")
                    except Exception as e2:
                        print(f"✗ 从镜像站点加载也失败: {e2}")
                        print("\n解决方案：")
                        print("1. 检查网络连接")
                        print("2. 设置代理: export HTTP_PROXY=your_proxy")
                        print("3. 或手动下载模型到本地")
                        raise e2
                else:
                    raise e
            hidden_dim = self.backbone.config.hidden_size
            self.projection = nn.Linear(hidden_dim, embed_dim)
        elif model_name.startswith('clip'):
            # 使用CLIP的文本编码器（注意：CLIP主要支持英文，中文效果可能不佳）
            if not CLIP_AVAILABLE:
                raise ImportError(
                    "CLIP is not installed. Install it with: pip install git+https://github.com/openai/CLIP.git\n"
                    "Or use 'bert-base-chinese' for Chinese text encoding."
                )
            try:
                # 支持指定CLIP模型版本，例如 'clip:ViT-B/32' 或 'clip:RN50'
                # 默认使用 ViT-B/32
                clip_model_name = "ViT-B/32"
                if ':' in model_name:
                    clip_model_name = model_name.split(':', 1)[1]
                    print(f"使用CLIP模型: {clip_model_name}")
                
                clip_model, _ = clip.load(clip_model_name, device='cpu')
                self.backbone = clip_model.transformer
                self.token_embedding = clip_model.token_embedding
                self.positional_embedding = clip_model.positional_embedding
                self.ln_final = clip_model.ln_final
                self.text_projection = clip_model.text_projection
                hidden_dim = clip_model.text_projection.shape[0]
                if embed_dim != clip_model.text_projection.shape[1]:
                    self.projection = nn.Linear(clip_model.text_projection.shape[1], embed_dim)
                else:
                    self.projection = nn.Identity()
            except Exception as e:
                raise ValueError(f"Failed to load CLIP model: {e}. Consider using 'bert-base-chinese' for Chinese text.")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model_name = model_name
        
    def tokenize(self, texts):
        """对文本进行tokenize"""
        if self.model_name.startswith('bert'):
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
        else:
            # CLIP tokenizer
            if not CLIP_AVAILABLE:
                raise ImportError("CLIP is not installed. Install it with: pip install git+https://github.com/openai/CLIP.git")
            return clip.tokenize(texts)
    
    def forward(self, input_ids=None, attention_mask=None, texts=None):
        """
        Forward pass
        Args:
            input_ids: tokenized input ids (for BERT)
            attention_mask: attention mask (for BERT)
            texts: raw text strings (will be tokenized if input_ids not provided)
        """
        if self.model_name.startswith('bert'):
            if texts is not None:
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(next(self.parameters()).device)
                attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
            
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            # 使用[CLS] token的表示
            x = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_dim]
            x = self.projection(x)
        else:
            # CLIP text encoder
            if not CLIP_AVAILABLE:
                raise ImportError("CLIP is not installed. Install it with: pip install git+https://github.com/openai/CLIP.git")
            if texts is not None:
                input_ids = clip.tokenize(texts, truncate=True).to(next(self.parameters()).device)
            
            x = self.token_embedding(input_ids)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.backbone(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            # 取最后一个token的表示（EOS token）
            # CLIP使用最后一个非padding token
            x = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)]
            x = x @ self.text_projection
            x = self.projection(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x


class CLIPModel(nn.Module):
    """CLIP模型 - 图像和文本编码器的组合"""
    
    def __init__(
        self,
        image_encoder_name='resnet50',
        text_encoder_name='bert-base-chinese',
        embed_dim=512,
        temperature=0.07
    ):
        super(CLIPModel, self).__init__()
        self.embed_dim = embed_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        self.image_encoder = ImageEncoder(
            model_name=image_encoder_name,
            embed_dim=embed_dim
        )
        
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
            embed_dim=embed_dim
        )
    
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


def create_model(config):
    """根据配置创建模型"""
    model = CLIPModel(
        image_encoder_name=config.get('image_encoder', 'resnet50'),
        text_encoder_name=config.get('text_encoder', 'bert-base-chinese'),
        embed_dim=config.get('embed_dim', 512),
        temperature=config.get('temperature', 0.07)
    )
    return model

