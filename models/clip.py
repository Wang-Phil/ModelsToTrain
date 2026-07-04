"""
CLIP风格的医学图像分类模型
实现图像-文本对齐的零样本分类

模型架构说明：
- 本模型采用CLIP（Contrastive Language-Image Pre-training）的对比学习架构思想
- 图像编码器：使用预训练的视觉模型（ResNet/ViT/BiomedCLIP等）作为backbone
- 文本编码器：使用预训练的语言模型（PubMedBERT/BiomedCLIP/PMC-CLIP/CLIP文本编码器）作为backbone
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
from pathlib import Path
# 如果设置了镜像环境变量，在导入transformers之前设置
if 'HF_ENDPOINT' not in os.environ:
    # 默认使用镜像站点（如果无法访问 Hugging Face）
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from transformers import AutoModel, AutoTokenizer
from torchvision import models

# CLIP 是可选的，只在需要时导入
# 注意：需要确保导入的是系统安装的 OpenAI CLIP 库，而不是本地的 clip 目录
try:
    import sys
    import importlib
    
    # 保存当前的 sys.path
    original_path = sys.path.copy()
    
    # 临时移除可能包含本地 clip 目录的路径
    # 查找可能包含本地 clip 目录的路径（通常是项目根目录）
    script_dir = Path(__file__).parent.parent  # models/ -> ModelsTotrain/
    local_clip_dir = script_dir / 'clip'
    
    # 如果本地 clip 目录在 sys.path 中，临时移除
    paths_to_remove = []
    for i, path in enumerate(sys.path):
        if str(local_clip_dir) == path or str(local_clip_dir.parent) == path:
            paths_to_remove.append((i, path))
    
    # 临时移除这些路径
    for i, path in reversed(paths_to_remove):
        sys.path.pop(i)
    
    try:
        # 如果本地clip目录存在，需要确保它不在sys.path中
        if local_clip_dir.exists():
            # 确保本地clip目录不在sys.path中
            current_path = [p for p in sys.path if str(local_clip_dir) not in p and str(local_clip_dir.parent) != p]
            sys.path[:] = current_path
        
        # 尝试导入系统安装的 OpenAI CLIP 库
        # 使用importlib来确保导入正确的模块
        import importlib
        import importlib.util
        
        # 尝试从系统路径导入clip
        spec = None
        for path in sys.path:
            if path and os.path.exists(path):
                clip_path = os.path.join(path, 'clip')
                if os.path.exists(clip_path) and os.path.isdir(clip_path):
                    # 检查是否是OpenAI CLIP（有load函数）
                    init_file = os.path.join(clip_path, '__init__.py')
                    if os.path.exists(init_file):
                        # 读取文件检查是否有load函数
                        with open(init_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if 'def load' in content or 'load =' in content:
                                spec = importlib.util.spec_from_file_location('clip', init_file)
                                break
        
        if spec is None:
            # 尝试直接导入
            import clip
        else:
            clip = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(clip)
        
        # 验证是否是 OpenAI CLIP 库（检查是否有 load 函数）
        if not hasattr(clip, 'load'):
            raise ImportError("导入的 clip 模块不是 OpenAI CLIP 库（缺少 load 函数）")
        CLIP_AVAILABLE = True
    finally:
        # 恢复原始路径
        sys.path[:] = original_path
        
except ImportError as e:
    CLIP_AVAILABLE = False
    clip = None
    print(f"Warning: Cannot import OpenAI CLIP library: {e}")


class _StarNetVisual(nn.Module):
    """StarNet 特征提取（去掉分类头），供 CLIP 图像编码器使用。"""

    def __init__(self, stem, stages, norm, avgpool):
        super().__init__()
        self.stem = stem
        self.stages = stages
        self.norm = norm
        self.avgpool = avgpool

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return x


class ImageEncoder(nn.Module):
    """图像编码器 - 支持多种预训练模型"""
    
    def __init__(self, model_name='resnet50', embed_dim=512):
        super(ImageEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.model_name = model_name
        
        # ResNet系列
        # 检查是否指定了预训练权重（通过 model_name:pretrained 格式，例如 resnet50:false, resnet50:pmcclip, resnet50:clip）
        resnet_pretrained = True  # 默认使用预训练权重
        resnet_model_name = model_name
        use_pmcclip = False  # 是否使用 PMC-CLIP 预训练权重
        use_clip = False  # 是否使用原始CLIP预训练权重
        
        if ':' in model_name and model_name.split(':')[0] in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            resnet_model_name, pretrained_str = model_name.split(':', 1)
            if pretrained_str.lower() == 'pmcclip':
                use_pmcclip = True
                resnet_pretrained = False  # 不使用 ImageNet 预训练
            elif pretrained_str.lower() == 'clip':
                use_clip = True
                resnet_pretrained = False  # 不使用 ImageNet 预训练
            else:
                resnet_pretrained = pretrained_str.lower() in ['true', '1', 'yes', 'pretrained']
        
        if resnet_model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            # 检查是否使用原始CLIP预训练权重
            if use_clip:
                # 使用原始CLIP预训练权重
                if not CLIP_AVAILABLE or clip is None:
                    raise ImportError(
                        "CLIP is not installed. Install it with: pip install git+https://github.com/openai/CLIP.git"
                    )
                
                # 确保 clip 有 load 函数
                if not hasattr(clip, 'load'):
                    raise ImportError(
                        "导入的 clip 模块不是 OpenAI CLIP 库（缺少 load 函数）。\n"
                        "可能是本地 clip 目录覆盖了系统库。请确保已安装: pip install git+https://github.com/openai/CLIP.git"
                    )
                
                # CLIP支持的ResNet模型映射
                clip_model_map = {
                    'resnet50': 'RN50',
                    'resnet101': 'RN101',
                }
                
                if resnet_model_name not in clip_model_map:
                    raise ValueError(
                        f"CLIP不支持 {resnet_model_name}。CLIP支持的ResNet模型: RN50 (ResNet50), RN101 (ResNet101)。\n"
                        f"如果使用 ResNet18，请使用 ImageNet 预训练权重（不指定 :clip，或使用 resnet18:true）"
                    )
                
                clip_model_name = clip_model_map[resnet_model_name]
                print(f"加载原始CLIP {clip_model_name} 预训练权重...")
                
                try:
                    # 加载CLIP模型
                    clip_model, _ = clip.load(clip_model_name, device='cpu')
                    
                    # 提取图像编码器（visual）
                    self.backbone = clip_model.visual
                    
                    # CLIP的ResNet输出维度是1024（RN50和RN101都是1024）
                    feature_dim = 1024
                    self.projection = nn.Linear(feature_dim, embed_dim)
                    self.forward_fn = self._forward_clip_resnet
                    
                    print(f"✓ 原始CLIP {clip_model_name} 权重加载成功，输出维度: {feature_dim}")
                except Exception as e:
                    raise ValueError(f"加载原始CLIP {clip_model_name} 失败: {e}")
            
            # 检查是否使用 PMC-CLIP 预训练权重
            elif use_pmcclip and resnet_model_name == 'resnet50':
                # 使用 PMC-CLIP 的 ModifiedResNet
                try:
                    import sys
                    import os
                    # 添加 clip 目录到路径
                    script_dir = Path(__file__).parent.parent  # models/ -> ModelsTotrain/
                    clip_dir = script_dir / 'clip'
                    if clip_dir.exists():
                        # 将 clip 目录添加到 sys.path，这样可以直接导入 pmcclip
                        if str(clip_dir) not in sys.path:
                            sys.path.insert(0, str(clip_dir))
                    
                    # 直接导入 pmcclip（因为 clip_dir 已经在 sys.path 中）
                    from pmcclip import ModifiedResNet
                    
                    # PMC-CLIP 检查点目录
                    checkpoint_dir = script_dir / 'clip' / 'checkpoints'
                    checkpoint_path = checkpoint_dir / 'image_encoder(resnet50).pth'
                    
                    # 如果检查点不存在，提示下载
                    if not checkpoint_path.exists():
                        print(f"警告: PMC-CLIP 权重文件不存在: {checkpoint_path}")
                        print("请从以下 URL 下载:")
                        print("https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth")
                        print(f"保存到: {checkpoint_dir}")
                        raise FileNotFoundError(f"PMC-CLIP 权重文件不存在: {checkpoint_path}")
                    
                    print(f"加载 PMC-CLIP ResNet50 预训练权重: {checkpoint_path}")
                    # 创建 ModifiedResNet 模型
                    # PMC-CLIP ResNet50 配置: layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64
                    self.backbone = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
                    # 加载预训练权重
                    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                    self.backbone.load_state_dict(state_dict)
                    print("✓ PMC-CLIP ResNet50 权重加载成功")
                    
                    # ModifiedResNet 的输出维度是 output_dim (768)
                    feature_dim = 768
                    self.projection = nn.Linear(feature_dim, embed_dim)
                    self.forward_fn = self._forward_pmcclip_resnet
                except ImportError as e:
                    raise ImportError(f"无法导入 PMC-CLIP 的 ModifiedResNet。请确保 clip/pmcclip.py 存在。错误: {e}")
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"PMC-CLIP 权重文件未找到: {e}")
                except Exception as e:
                    raise ValueError(f"加载 PMC-CLIP ResNet50 失败: {e}")
            else:
                # 使用标准的 torchvision ResNet（ImageNet 预训练或随机初始化）
                resnet = getattr(models, resnet_model_name)(pretrained=resnet_pretrained)
                self.backbone = nn.Sequential(*list(resnet.children())[:-1])
                # ResNet的feature dim: resnet18/34是512, resnet50/101/152是2048
                feature_dim = 512 if resnet_model_name in ['resnet18', 'resnet34'] else 2048
                self.projection = nn.Linear(feature_dim, embed_dim)
                self.forward_fn = self._forward_resnet
        
        # ViT
        elif model_name == 'vit':
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
            self.projection = nn.Linear(768, embed_dim)
            self.forward_fn = self._forward_vit
        
        # BiomedCLIP 图像编码器
        elif model_name == 'biomedclip' or model_name.startswith('biomedclip:'):
            try:
                # 尝试导入 open_clip
                import sys
                import os
                # 添加 open_clip 路径（优先使用当前目录下的 open_clip）
                script_dir = Path(__file__).parent.parent  # models/ -> ModelsTotrain/
                open_clip_base_path = script_dir / 'open_clip'
                
                # 如果当前目录下没有，尝试 BiomedCoOp 路径
                if not open_clip_base_path.exists():
                    biomedcoop_path = os.path.join(os.path.dirname(__file__), '..', '..', 'BiomedCoOp', 'open_clip')
                    if os.path.exists(biomedcoop_path):
                        open_clip_base_path = Path(biomedcoop_path)
                
                if open_clip_base_path.exists():
                    # 将 open_clip 的父目录添加到 sys.path，这样 open_clip.src.open_clip 可以正常工作
                    sys.path.insert(0, str(open_clip_base_path.parent))
                    print(f"使用 open_clip 路径: {open_clip_base_path}")
                
                # open_clip 内部使用 open_clip.src.open_clip，所以需要这样导入
                from open_clip.src.open_clip import create_model_from_pretrained
                
                # 解析模型名称（支持 biomedclip 或 biomedclip:hf-hub:model_name）
                if ':' in model_name:
                    pretrained_name = model_name.split(':', 1)[1]
                else:
                    pretrained_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                
                print(f"加载 BiomedCLIP 图像编码器: {pretrained_name}")
                biomedclip_model, _ = create_model_from_pretrained(pretrained_name)
                
                # 获取图像编码器（visual）
                self.backbone = biomedclip_model.visual
                
                # 动态获取 BiomedCLIP 的输出维度
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    # 获取模型的数据类型
                    try:
                        dtype = next(self.backbone.parameters()).dtype
                    except:
                        dtype = torch.float32
                    dummy_output = self.backbone(dummy_input.type(dtype))
                    # 处理输出形状
                    if len(dummy_output.shape) == 3:
                        dummy_output = dummy_output[:, 0]  # CLS token
                    elif len(dummy_output.shape) == 2:
                        pass
                    else:
                        dummy_output = dummy_output.view(dummy_output.shape[0], -1)
                    feature_dim = dummy_output.shape[1]
                    print(f"BiomedCLIP 图像编码器输出维度: {feature_dim}")
                
                self.projection = nn.Linear(feature_dim, embed_dim)
                self.forward_fn = self._forward_biomedclip
                
            except ImportError as e:
                raise ImportError(f"BiomedCLIP 需要 open_clip 库。请确保已安装或 BiomedCoOp/open_clip 路径正确。错误: {e}")
            except Exception as e:
                raise ValueError(f"加载 BiomedCLIP 失败: {e}")

        # StarNet 系列（如 starnet_s1, starnet_s1:pretrained，见 CLIP_TRAINING_GUIDE.md）
        elif model_name.split(":", 1)[0].startswith("starnet_"):
            from models import starnet as starnet_mod

            if ":" in model_name:
                arch, flag = model_name.split(":", 1)
                use_pretrained = flag.lower() in ("pretrained", "true", "1", "yes")
            else:
                arch = model_name
                use_pretrained = False

            factory = getattr(starnet_mod, arch, None)
            if factory is None:
                raise ValueError(
                    f"未知的 StarNet 架构: {arch}。请在 models/starnet.py 中确认已注册该构建函数。"
                )

            full = factory(pretrained=use_pretrained, num_classes=1000)
            self.backbone = _StarNetVisual(full.stem, full.stages, full.norm, full.avgpool)
            feature_dim = full.head.in_features
            self.projection = nn.Linear(feature_dim, embed_dim)
            self.forward_fn = self._forward_starnet
            print(f"✓ StarNet 图像编码器: {model_name}, feature_dim={feature_dim}, pretrained={use_pretrained}")

        # CasGNet（默认骨干见 models/casgnet.py；可替换为你的结构或加载权重）
        elif model_name == "casgnet" or model_name.startswith("casgnet:"):
            from models.casgnet import build_casgnet_visual

            ckpt_path = None
            pretrained = False
            if model_name.startswith("casgnet:"):
                suffix = model_name[len("casgnet:") :]
                low = suffix.lower()
                if low in ("pretrained", "true", "1", "yes"):
                    pretrained = True
                elif suffix:
                    ckpt_path = suffix

            self.backbone, feature_dim = build_casgnet_visual(
                checkpoint_path=ckpt_path, pretrained=pretrained
            )
            self.projection = nn.Linear(feature_dim, embed_dim)
            self.forward_fn = self._forward_casgnet
            print(
                f"✓ CasGNet 图像编码器: {model_name}, feature_dim={feature_dim}, "
                f"checkpoint={ckpt_path}, pretrained_flag={pretrained}"
            )

        else:
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"支持的模型: resnet18/34/50/101/152, "
                           f"resnet50:clip (使用原始CLIP预训练权重), "
                           f"resnet101:clip (使用原始CLIP预训练权重), "
                           f"resnet50:pmcclip (使用PMC-CLIP预训练权重), "
                           f"vit, biomedclip, starnet_s1[:pretrained], casgnet[:path.pth]")
        
    def _forward_resnet(self, x):
        """ResNet前向传播"""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.projection(x)
        return x
    
    def _forward_pmcclip_resnet(self, x):
        """PMC-CLIP ModifiedResNet 前向传播"""
        # ModifiedResNet 的 forward 返回字典，包含 'image_features'
        output = self.backbone(x)
        if isinstance(output, dict):
            x = output['image_features']  # [batch_size, output_dim]
        else:
            x = output
        # 投影到目标维度
        x = self.projection(x)
        return x
    
    def _forward_vit(self, x):
        """ViT前向传播"""
        outputs = self.backbone(x)
        x = outputs.last_hidden_state[:, 0]  # CLS token
        x = self.projection(x)
        return x
    
    def _forward_clip_resnet(self, x):
        """原始CLIP ResNet前向传播"""
        # CLIP的ResNet编码器
        # 获取模型的数据类型
        try:
            dtype = next(self.backbone.parameters()).dtype
        except:
            dtype = torch.float32
        
        # 转换输入数据类型
        x = x.type(dtype)
        
        # 前向传播
        x = self.backbone(x)
        
        # CLIP ResNet输出是 [batch_size, feature_dim]
        # 确保是2D tensor
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        
        # 投影到目标维度
        x = self.projection(x)
        return x
    
    def _forward_starnet(self, x):
        """StarNet 特征 + 投影"""
        x = self.backbone(x)
        return self.projection(x)

    def _forward_casgnet(self, x):
        """CasGNet 特征 + 投影"""
        x = self.backbone(x)
        return self.projection(x)

    def _forward_biomedclip(self, x):
        """BiomedCLIP 图像编码器前向传播"""
        # BiomedCLIP 的 visual 编码器（ViT-B/16）
        # 获取模型的数据类型（ViT 没有 conv1，需要从其他层获取）
        try:
            # 尝试从第一个参数获取 dtype
            dtype = next(self.backbone.parameters()).dtype
        except:
            # 如果失败，使用 float32
            dtype = torch.float32
        
        # 转换输入数据类型
        x = x.type(dtype)
        
        # 前向传播
        x = self.backbone(x)
        
        # 处理输出形状
        # ViT 输出可能是 [batch_size, seq_len, dim] 或 [batch_size, dim]
        if len(x.shape) == 3:
            # [batch_size, seq_len, dim] -> 取 CLS token (第一个token)
            x = x[:, 0]
        elif len(x.shape) == 2:
            # [batch_size, dim] -> 直接使用
            pass
        else:
            # 其他形状，尝试 flatten
            x = x.view(x.shape[0], -1)
        
        # 投影到目标维度
        x = self.projection(x)
        return x
        
    def forward(self, x):
        x = self.forward_fn(x)
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        return x


class TextEncoder(nn.Module):
    """文本编码器 - 使用PubMedBERT/BiomedCLIP/PMC-CLIP/CLIP的文本编码器"""
    
    def __init__(self, model_name='pubmedbert', embed_dim=512):
        """
        Args:
            model_name: 模型名称
            embed_dim: embedding维度
        """
        super(TextEncoder, self).__init__()
        self.embed_dim = embed_dim
        
        if model_name.startswith('pubmedbert') or model_name == 'biomedclip_text':
            # 使用PubMedBERT或BiomedCLIP
            # 检查是否设置了镜像环境变量，如果没有且无法连接，则使用镜像
            import os
            hf_endpoint = os.environ.get('HF_ENDPOINT', None)
            
            # 处理 BiomedCLIP 文本编码器
            if model_name == 'biomedclip_text':
                # 尝试从 open_clip 加载 BiomedCLIP 的文本编码器
                try:
                    import sys
                    # 添加 open_clip 路径（优先使用当前目录下的 open_clip）
                    script_dir = Path(__file__).parent.parent  # models/ -> ModelsTotrain/
                    open_clip_base_path = script_dir / 'open_clip'
                    
                    # 如果当前目录下没有，尝试 BiomedCoOp 路径
                    if not open_clip_base_path.exists():
                        biomedcoop_path = os.path.join(os.path.dirname(__file__), '..', '..', 'BiomedCoOp', 'open_clip')
                        if os.path.exists(biomedcoop_path):
                            open_clip_base_path = Path(biomedcoop_path)
                    
                    if open_clip_base_path.exists():
                        # 将 open_clip 的父目录添加到 sys.path，这样 open_clip.src.open_clip 可以正常工作
                        sys.path.insert(0, str(open_clip_base_path.parent))
                        print(f"使用 open_clip 路径: {open_clip_base_path}")
                    
                    # open_clip 内部使用 open_clip.src.open_clip，所以需要这样导入
                    from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
                    
                    pretrained_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                    print(f"加载 BiomedCLIP 文本编码器: {pretrained_name}")
                    biomedclip_model, _ = create_model_from_pretrained(pretrained_name)
                    
                    # 保存完整的模型以便调用 encode_text
                    self.biomedclip_model = biomedclip_model
                    # 获取文本编码器（用于直接调用）
                    self.backbone = biomedclip_model.text
                    self.tokenizer = get_tokenizer(pretrained_name)
                    
                    # 动态检测 BiomedCLIP 文本编码器的输出维度
                    # 使用一个虚拟输入来获取实际输出维度
                    try:
                        with torch.no_grad():
                            # 创建一个虚拟的 tokenized 输入
                            dummy_texts = ["dummy text"]
                            dummy_tokenized = self.tokenizer(dummy_texts)
                            if isinstance(dummy_tokenized, torch.Tensor):
                                dummy_input_ids = dummy_tokenized
                            else:
                                dummy_input_ids = dummy_tokenized['input_ids'] if 'input_ids' in dummy_tokenized else dummy_tokenized
                            
                            # 获取输出维度
                            dummy_output = self.biomedclip_model.encode_text(dummy_input_ids, normalize=False)
                            hidden_dim = dummy_output.shape[-1]
                            print(f"BiomedCLIP 文本编码器实际输出维度: {hidden_dim}")
                    except Exception as e:
                        print(f"无法动态检测输出维度，使用默认值 512: {e}")
                        # BiomedCLIP 的文本编码器输出通常是 512（与图像编码器对齐）
                        hidden_dim = 512
                    
                    self.projection = nn.Linear(hidden_dim, embed_dim)
                    self.model_name = 'biomedclip_text'
                    self._use_biomedclip_text = True
                except ImportError as e:
                    raise ImportError(f"BiomedCLIP 文本编码器需要 open_clip 库。错误: {e}")
                except Exception as e:
                    raise ValueError(f"加载 BiomedCLIP 文本编码器失败: {e}")
            else:
                # PubMedBERT 文本编码器
                # 如果是 pubmedbert，使用 microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
                if model_name.startswith('pubmedbert'):
                    if model_name == 'pubmedbert':
                        model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
                    elif model_name.startswith('pubmedbert:'):
                        model_name = model_name.split(':', 1)[1]
                
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
            # 只有在不是 BiomedCLIP 时才创建投影层
            # 如果之前已经创建了（BiomedCLIP），不要覆盖
            if not hasattr(self, 'projection') or self.projection is None:
                self.projection = nn.Linear(hidden_dim, embed_dim)
            # 只有在不是 BiomedCLIP 时才设置为 False
            # 如果之前已经设置为 True（BiomedCLIP），不要覆盖
            if not hasattr(self, '_use_biomedclip_text') or not self._use_biomedclip_text:
                self._use_biomedclip_text = False
            # 保存原始模型名称用于 forward 判断
            self.model_name = 'pubmedbert'  # 保存原始名称
        elif model_name == 'pmcclip_text':
            # 使用 PMC-CLIP 的文本编码器（BiomedBERT + text_projection_layer）
            try:
                import os
                import sys
                # 添加 clip 目录到路径
                script_dir = Path(__file__).parent.parent  # models/ -> ModelsTotrain/
                clip_dir = script_dir / 'clip'
                checkpoints_dir = clip_dir / 'checkpoints'
                
                if clip_dir.exists():
                    sys.path.insert(0, str(clip_dir))
                    print(f"使用 PMC-CLIP 路径: {clip_dir}")
                
                # 检查并下载 PMC-CLIP 预训练权重
                text_encoder_path = checkpoints_dir / 'text_encoder.pth'
                text_projection_path = checkpoints_dir / 'text_projection_layer.pth'
                
                # 文件 URL
                files_to_download = {
                    'text_encoder.pth': 'https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth',
                    'text_projection_layer.pth': 'https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth',
                }
                
                # 如果文件不存在，下载
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                for filename, url in files_to_download.items():
                    filepath = checkpoints_dir / filename
                    if not filepath.exists():
                        print(f"下载 PMC-CLIP {filename}...")
                        import requests
                        from tqdm import tqdm
                        response = requests.get(url, stream=True)
                        if response.status_code == 200:
                            total_size = int(response.headers.get('content-length', 0))
                            with open(filepath, 'wb') as f:
                                with tqdm(total=total_size, unit='B', unit_scale=True, desc=str(filepath)) as pbar:
                                    for chunk in response.iter_content(chunk_size=1024):
                                        f.write(chunk)
                                        pbar.update(len(chunk))
                            print(f"✓ {filename} 下载完成")
                        else:
                            raise RuntimeError(f"下载失败: {filename}, HTTP {response.status_code}")
                
                # 加载文本编码器（BiomedBERT）
                print(f"加载 PMC-CLIP 文本编码器...")
                self.backbone = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
                self.backbone.load_state_dict(torch.load(str(text_encoder_path), weights_only=True))
                
                # 加载文本投影层
                text_projection_layer = torch.load(str(text_projection_path), weights_only=True)
                self.text_projection_layer = nn.Parameter(text_projection_layer)
                
                # 获取 tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
                
                # PMC-CLIP 输出维度是 768
                hidden_dim = 768
                self.projection = nn.Linear(hidden_dim, embed_dim)
                self.model_name = 'pmcclip_text'
                self._use_pmcclip_text = True
                self._use_biomedclip_text = False
                print(f"✓ PMC-CLIP 文本编码器加载完成，输出维度: {hidden_dim}")
            except Exception as e:
                raise ValueError(f"加载 PMC-CLIP 文本编码器失败: {e}")
        
        elif model_name.startswith('clip'):
            # 使用CLIP的文本编码器（注意：CLIP主要支持英文，中文效果可能不佳）
            if not CLIP_AVAILABLE or clip is None:
                raise ImportError(
                    "CLIP is not installed. Install it with: pip install git+https://github.com/openai/CLIP.git"
                )
            
            # 确保 clip 有 load 函数（验证是否是 OpenAI CLIP 库）
            if not hasattr(clip, 'load'):
                raise ImportError(
                    "导入的 clip 模块不是 OpenAI CLIP 库（缺少 load 函数）。\n"
                    "可能是本地 clip 目录覆盖了系统库。请确保已安装: pip install git+https://github.com/openai/CLIP.git"
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
                raise ValueError(f"Failed to load CLIP model: {e}")
        else:
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"支持的模型: pubmedbert, biomedclip_text, pmcclip_text, clip:ViT-B/32")
        
        if not hasattr(self, '_use_biomedclip_text'):
            self._use_biomedclip_text = False
        if not hasattr(self, '_use_pmcclip_text'):
            self._use_pmcclip_text = False
        if not hasattr(self, 'model_name'):
            self.model_name = model_name
        
    def tokenize(self, texts):
        """对文本进行tokenize"""
        if self._use_biomedclip_text:
            # BiomedCLIP tokenizer
            if isinstance(texts, str):
                texts = [texts]
            return self.tokenizer(texts)
        elif hasattr(self, '_use_pmcclip_text') and self._use_pmcclip_text:
            # PMC-CLIP tokenizer (BiomedBERT)
            return self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors='pt'
            )
        elif self.model_name.startswith('pubmedbert'):
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
        # 确保 _use_biomedclip_text 被正确初始化
        if not hasattr(self, '_use_biomedclip_text'):
            self._use_biomedclip_text = False
        
        # 优先检查是否是 BiomedCLIP（使用 model_name 作为备用判断）
        is_biomedclip = self._use_biomedclip_text or (hasattr(self, 'model_name') and self.model_name == 'biomedclip_text')
        
        if is_biomedclip:
            # BiomedCLIP 文本编码器
            device = next(self.parameters()).device
            
            # 处理输入：优先使用 texts，如果提供了 input_ids 则使用 input_ids
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                # 使用 tokenizer 对文本进行编码
                tokenized = self.tokenizer(texts)
                if isinstance(tokenized, torch.Tensor):
                    input_ids = tokenized.to(device)
                else:
                    # 如果是字典，提取 input_ids
                    input_ids = tokenized['input_ids'].to(device) if 'input_ids' in tokenized else tokenized.to(device)
            elif input_ids is not None:
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids).to(device)
                else:
                    input_ids = input_ids.to(device)
            else:
                raise ValueError("BiomedCLIP 文本编码器需要提供 texts 或 input_ids")
            
            # 通过 text 子模块处理（避免 encode_text 的兼容性问题）
            try:
                if hasattr(self.biomedclip_model, 'text'):
                    # 获取 text 编码器的输出
                    # 注意：BiomedCLIP 的 text 子模块期望 input_ids 是 2D tensor [batch, seq_len]
                    if len(input_ids.shape) == 1:
                        input_ids = input_ids.unsqueeze(0)  # 添加 batch 维度
                    
                    outputs = self.biomedclip_model.text(input_ids)
                    # 处理输出：通常取 CLS token 或最后一个 token
                    if isinstance(outputs, tuple):
                        x = outputs[0]  # 取第一个元素（通常是 hidden states）
                    else:
                        x = outputs
                    # 处理输出形状
                    if len(x.shape) == 3:  # [batch, seq_len, hidden_dim]
                        x = x[:, 0]  # 取 CLS token
                    elif len(x.shape) == 2:
                        pass  # 已经是 [batch, hidden_dim]
                    else:
                        x = x.view(x.shape[0], -1)
                else:
                    raise ValueError("BiomedCLIP 模型没有 text 子模块")
            except Exception as e:
                raise NotImplementedError(f"BiomedCLIP 文本编码失败: {e}")
            
            # 投影到目标维度
            x = self.projection(x)
        elif hasattr(self, '_use_pmcclip_text') and self._use_pmcclip_text:
            # PMC-CLIP 文本编码器（BiomedBERT + text_projection_layer）
            device = next(self.parameters()).device
            
            # 处理输入
            if input_ids is None:
                if texts is not None:
                    # 使用 PMC-CLIP 的 tokenizer (BiomedBERT)
                    if isinstance(texts, str):
                        texts = [texts]
                    encoded = self.tokenizer(
                        texts,
                        padding='max_length',
                        truncation=True,
                        max_length=77,
                        return_tensors='pt'
                    )
                    input_ids = encoded['input_ids'].to(device)
                    attention_mask = encoded['attention_mask'].to(device)
                else:
                    raise ValueError("PMC-CLIP 文本编码器需要提供 texts 或 input_ids")
            else:
                input_ids = input_ids.to(device) if hasattr(input_ids, 'to') else input_ids
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device) if hasattr(attention_mask, 'to') else attention_mask
            
            # PMC-CLIP 文本编码器前向传播
            # 使用 BiomedBERT 编码，然后通过 text_projection_layer
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooler_output = outputs.pooler_output  # [batch_size, 768]
            
            # 通过 text_projection_layer 投影
            x = pooler_output @ self.text_projection_layer  # [batch_size, 768]
            
            # 投影到目标维度
            x = self.projection(x)
        elif (self.model_name == 'pubmedbert' or (hasattr(self, 'backbone') and hasattr(self.backbone, 'config'))) and (not hasattr(self, '_use_biomedclip_text') or not self._use_biomedclip_text) and (not hasattr(self, 'model_name') or self.model_name != 'biomedclip_text') and (not hasattr(self, '_use_pmcclip_text') or not self._use_pmcclip_text):
            # PubMedBERT 文本编码器
            # 注意：排除 BiomedCLIP（即使它有 config 属性）
            # 检查是否有 config 属性来区分 BERT 和 CLIP
            device = next(self.parameters()).device
            
            # 传统方式：使用文本提示
            if texts is not None:
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
            
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            # 使用[CLS] token的表示
            x = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_dim]
            x = self.projection(x)
        elif self.model_name.startswith('clip') or hasattr(self, 'token_embedding'):
            # CLIP text encoder（只有在有 token_embedding 属性时才使用）
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
        text_encoder_name='pubmedbert',
        embed_dim=512,
        temperature=0.07
    ):
        """
        Args:
            image_encoder_name: 图像编码器名称
            text_encoder_name: 文本编码器名称
            embed_dim: embedding维度
            temperature: 温度参数
        """
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
        image_encoder_name=config.get('image_encoder_name', config.get('image_encoder', 'resnet50')),
        text_encoder_name=config.get('text_encoder_name', config.get('text_encoder', 'pubmedbert')),
        embed_dim=config.get('embed_dim', 512),
        temperature=config.get('temperature', 0.07)
    )
    return model