"""
经典分类模型集合
包含：ResNet, Inception, DenseNet, MobileNet, GoogleNet, EfficientNet等
统一接口，方便在训练脚本中使用
"""

import torch
import torch.nn as nn
import torchvision.models as models
from .starnet import starnet_s050, starnet_s100, starnet_s150
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("警告: timm未安装，EfficientNet V2将不可用")


def get_resnet18(num_classes=9, pretrained=False):
    """
    ResNet18模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        ResNet18模型
    """
    # 兼容新版本 PyTorch (使用 weights 参数)
    try:
        # PyTorch 1.13+ 使用 weights 参数
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=None)
    except (AttributeError, TypeError):
        # 旧版本 PyTorch 使用 pretrained 参数
        model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_resnet50(num_classes=9, pretrained=False):
    """
    ResNet50模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        ResNet50模型
    """
    model = models.resnet50(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_resnet101(num_classes=9, pretrained=False):
    """
    ResNet101模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        ResNet101模型
    """
    model = models.resnet101(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_inceptionv3(num_classes=9, pretrained=False):
    """
    InceptionV3模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        InceptionV3模型
    """
    # InceptionV3需要先创建模型（aux_logits=True用于预训练权重）
    # 然后禁用辅助分类器
    model = models.inception_v3(pretrained=pretrained, aux_logits=True)
    # 禁用辅助分类器
    model.AuxLogits = None
    # 修改主分类器
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_efficientnet_b0(num_classes=9, pretrained=False):
    """
    EfficientNet-B0 模型

    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重

    Returns:
        EfficientNet-B0 模型
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model


def get_densenet121(num_classes=9, pretrained=False):
    """
    DenseNet121模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        DenseNet121模型
    """
    model = models.densenet121(pretrained=pretrained)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    return model


def get_densenet161(num_classes=9, pretrained=False):
    """
    DenseNet161模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        DenseNet161模型
    """
    model = models.densenet161(pretrained=pretrained)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    return model


def get_densenet201(num_classes=9, pretrained=False):
    """
    DenseNet201模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        DenseNet201模型
    """
    model = models.densenet201(pretrained=pretrained)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    return model


def get_mobilenetv2(num_classes=9, pretrained=False):
    """
    MobileNetV2模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        MobileNetV2模型
    """
    model = models.mobilenet_v2(pretrained=pretrained)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model


def get_mobilenetv3_small(num_classes=9, pretrained=False):
    """
    MobileNetV3-Small模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        MobileNetV3-Small模型
    """
    model = models.mobilenet_v3_small(pretrained=pretrained)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, num_classes)
    return model


def get_mobilenetv3_large(num_classes=9, pretrained=False):
    """
    MobileNetV3-Large模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        MobileNetV3-Large模型
    """
    model = models.mobilenet_v3_large(pretrained=pretrained)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, num_classes)
    return model


def get_googlenet(num_classes=9, pretrained=False):
    """
    GoogleNet模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        GoogleNet模型
    """
    model = models.googlenet(pretrained=pretrained, aux_logits=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_efficientnet_v2_s(num_classes=9, pretrained=False):
    """
    EfficientNet V2-S模型（使用timm库）
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        EfficientNet V2-S模型
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm库未安装，无法使用EfficientNet V2。请安装: pip install timm")
    
    model = timm.create_model('efficientnetv2_s', pretrained=pretrained, num_classes=num_classes)
    return model


def get_efficientnet_v2_m(num_classes=9, pretrained=False):
    """
    EfficientNet V2-M模型（使用timm库）
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        EfficientNet V2-M模型
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm库未安装，无法使用EfficientNet V2。请安装: pip install timm")
    
    model = timm.create_model('efficientnetv2_m', pretrained=pretrained, num_classes=num_classes)
    return model


def get_efficientnet_v2_l(num_classes=9, pretrained=False):
    """
    EfficientNet V2-L模型（使用timm库）
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        EfficientNet V2-L模型
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm库未安装，无法使用EfficientNet V2。请安装: pip install timm")
    
    model = timm.create_model('efficientnetv2_l', pretrained=pretrained, num_classes=num_classes)
    return model


# UNet实现（用于分割任务，但也可以用于分类）
class UNet(nn.Module):
    """
    UNet网络（用于图像分割，也可以用于分类）
    这里实现一个简化版本用于分类任务
    """
    def __init__(self, num_classes=9, in_channels=3):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self._conv_block(1024 + 512, 512)
        self.dec3 = self._conv_block(512 + 256, 256)
        self.dec2 = self._conv_block(256 + 128, 128)
        self.dec1 = self._conv_block(128 + 64, 64)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upsample(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upsample(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upsample(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Classification
        x = self.global_pool(dec1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def get_unet(num_classes=9, pretrained=False):
    """
    UNet模型（用于分类）
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重（UNet不支持预训练）
    
    Returns:
        UNet模型
    """
    if pretrained:
        print("警告: UNet不支持预训练权重，将使用随机初始化")
    return UNet(num_classes=num_classes)


# TransUNet实现（简化版本，用于分类）
class TransUNet(nn.Module):
    """
    TransUNet网络（结合Transformer和UNet）
    这里实现一个简化版本用于分类任务
    """
    def __init__(self, num_classes=9, in_channels=3, img_size=224, patch_size=16, embed_dim=768):
        super(TransUNet, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        
        # UNet-like decoder
        self.dec1 = self._conv_block(embed_dim, 512)
        self.dec2 = self._conv_block(512, 256)
        self.dec3 = self._conv_block(256, 128)
        self.dec4 = self._conv_block(128, 64)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        B = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        B, C, H, W = x.size()
        
        # Flatten for transformer
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        
        # Add positional encoding (learnable)
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.transformer(x)  # [B, H*W, embed_dim]
        
        # Reshape back
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, embed_dim, H, W]
        
        # Decoder
        x = self.dec1(x)
        x = self.upsample(x)
        x = self.dec2(x)
        x = self.upsample(x)
        x = self.dec3(x)
        x = self.upsample(x)
        x = self.dec4(x)
        
        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def get_transunet(num_classes=9, pretrained=False, img_size=224):
    """
    TransUNet模型（用于分类）
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重（TransUNet不支持预训练）
        img_size: 输入图像大小
    
    Returns:
        TransUNet模型
    """
    if pretrained:
        print("警告: TransUNet不支持预训练权重，将使用随机初始化")
    return TransUNet(num_classes=num_classes, img_size=img_size)


# 模型注册表，方便统一调用
MODEL_REGISTRY = {
    'resnet18': get_resnet18,
    'resnet50': get_resnet50,
    'resnet101': get_resnet101,
    'inceptionv3': get_inceptionv3,
    'densenet121': get_densenet121,
    'densenet161': get_densenet161,
    'densenet201': get_densenet201,
    'mobilenetv2': get_mobilenetv2,
    'mobilenetv3_small': get_mobilenetv3_small,
    'mobilenetv3_large': get_mobilenetv3_large,
    'googlenet': get_googlenet,
    'efficientnet_b0': get_efficientnet_b0,
    'efficientnetv2_s': get_efficientnet_v2_s,
    'efficientnetv2_m': get_efficientnet_v2_m,
    'efficientnetv2_l': get_efficientnet_v2_l,
    'starnet_s050': lambda num_classes, pretrained=False, **kwargs: starnet_s050(pretrained=pretrained, **kwargs),
    'starnet_s100': lambda num_classes, pretrained=False, **kwargs: starnet_s100(pretrained=pretrained, **kwargs),
    'starnet_s150': lambda num_classes, pretrained=False, **kwargs: starnet_s150(pretrained=pretrained, **kwargs),
    'unet': get_unet,
    'transunet': get_transunet,
}


def create_model(model_name, num_classes=9, pretrained=False, **kwargs):
    """
    统一的模型创建接口
    
    Args:
        model_name: 模型名称（小写）
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        **kwargs: 其他模型特定参数
    
    Returns:
        模型实例
    
    Examples:
        model = create_model('resnet50', num_classes=9, pretrained=False)
        model = create_model('efficientnetv2_s', num_classes=9, pretrained=True)
        model = create_model('transunet', num_classes=9, img_size=224)
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_REGISTRY:
        available_models = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"未知模型: {model_name}\n"
            f"可用模型: {available_models}"
        )
    
    model_fn = MODEL_REGISTRY[model_name]
    
    # 处理特殊参数
    if model_name == 'transunet':
        img_size = kwargs.get('img_size', 224)
        return model_fn(num_classes=num_classes, pretrained=pretrained, img_size=img_size)
    else:
        return model_fn(num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    # 测试所有模型
    print("测试模型创建...")
    
    test_models = [
        'resnet18', 'resnet50', 'resnet101', 'inceptionv3', 'densenet121', 'densenet161', 'densenet201',
        'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large', 'googlenet', 'efficientnet_b0', 'unet', 'transunet'
    ]
    
    if TIMM_AVAILABLE:
        test_models.extend(['efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l'])
    
    for model_name in test_models:
        try:
            print(f"\n创建模型: {model_name}")
            model = create_model(model_name, num_classes=9, pretrained=False)
            print(f"  ✓ 成功创建 {model_name}")
            print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"  ✗ 创建 {model_name} 失败: {e}")

