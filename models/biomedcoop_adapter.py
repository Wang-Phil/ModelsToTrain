"""
BiomedCoOp 模型适配器
将 BiomedCoOp 的 CustomCLIP 包装成兼容 train_clip.py 的接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp

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

# 延迟导入 BiomedCoOp 模型（避免 dassl 依赖问题）
# 在需要时才导入，这样可以避免在导入时就需要 dassl
def _import_biomedcoop_components():
    """延迟导入 BiomedCoOp 组件"""
    try:
        from models.biomedcoop_biomedclip import CustomCLIP
        # PromptLearner 已被注释，不再需要导入
        # from models.biomedcoop_biomedclip import CustomCLIP, PromptLearner
        return CustomCLIP, None  # 返回 None 作为 PromptLearner 的占位符
    except ImportError as e:
        raise ImportError(
            f"无法导入 BiomedCoOp 组件: {e}\n"
            f"请确保已安装所有依赖（dassl, gdown 等）"
        )

# 导入 open_clip
from open_clip.src.open_clip import create_model_from_pretrained
import os

# 设置环境变量，优先使用本地缓存的模型
# 增加超时时间（从默认的10秒增加到300秒）
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
# 设置重试次数
os.environ.setdefault('HF_HUB_DOWNLOAD_MAX_RETRIES', '10')
# 如果网络有问题，可以设置离线模式（需要确保模型已缓存）
# os.environ.setdefault('HF_HUB_OFFLINE', '1')  # 取消注释以启用离线模式


class BiomedCoOpAdapter(nn.Module):
    """
    将 BiomedCoOp 的 CustomCLIP 适配为 CLIPModel 接口
    使其可以在 train_clip.py 中使用
    
    注意：BiomedCoOp 使用自己的损失函数（分类损失 + 对比损失 + SCCM损失），
    而不是 train_clip.py 的标准 CLIP 损失。适配器会返回特征，但实际的损失计算
    在模型内部完成（通过 CustomCLIP.forward）。
    """
    
    def __init__(
        self,
        classnames,
        class_texts_file=None,
        embed_dim=512,
        temperature=0.07,
        n_ctx=4,
        ctx_init="a photo of a",
        csc=False,
        class_token_position="end",
        sccm_lambda=1.0,
        use_focal_loss=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        device='cuda'
    ):
        """
        Args:
            classnames: 类别名称列表
            class_texts_file: 类别文本描述JSON文件路径
            embed_dim: 嵌入维度（BiomedCoOp 使用固定的 512）
            temperature: 温度参数（BiomedCoOp 使用可学习的 logit_scale）
            n_ctx: 上下文token数量
            ctx_init: 上下文初始化文本
            csc: 是否使用类别特定的上下文
            class_token_position: 类别token位置 ("end", "middle", "front")
            sccm_lambda: SCCM损失权重
            use_focal_loss: 是否使用 Focal Loss
            focal_alpha: Focal Loss alpha 参数
            focal_gamma: Focal Loss gamma 参数
            device: 设备
        """
        super().__init__()
        
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.embed_dim = embed_dim
        self.device = device
        
        # 创建配置对象
        cfg = self._create_config(
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            csc=csc,
            class_token_position=class_token_position,
            sccm_lambda=sccm_lambda,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            class_texts_file=class_texts_file
        )
        
        # 加载 BiomedCLIP 模型
        # 延迟导入 BiomedCoOp 组件
        CustomCLIP, PromptLearner = _import_biomedcoop_components()
        
        print(f"Loading BiomedCLIP for BiomedCoOp adapter...")
        # 确保模型加载到正确的设备上
        import torch
        if isinstance(device, str):
            if device.startswith('cuda:'):
                device_obj = torch.device(device)
            else:
                device_obj = torch.device(device)
        else:
            device_obj = device
        
        biomedclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        biomedclip_model = biomedclip_model.float().eval()
        # 立即将模型移动到指定设备
        biomedclip_model = biomedclip_model.to(device_obj)
        
        # 创建 CustomCLIP 模型
        self.biomedcoop_model = CustomCLIP(cfg, classnames, biomedclip_model)
        # 确保整个模型在正确的设备上
        self.biomedcoop_model = self.biomedcoop_model.to(device_obj)
        
        # 为了兼容 train_clip.py，需要提供这些属性
        self.image_encoder = self.biomedcoop_model.image_encoder
        # ========== 原始 BiomedCLIP：不使用单独的 text_encoder ==========
        # self.text_encoder = self.biomedcoop_model.text_encoder  # 已注释，使用完整模型
        # 创建一个占位符以保持兼容性
        self.text_encoder = None  # 原始 BiomedCLIP 使用完整模型进行文本编码
        
        # 温度参数（BiomedCoOp 使用 logit_scale，这里创建一个兼容的 temperature）
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # 存储类别文本（用于预测）
        self.class_texts = None
        self._setup_class_texts(class_texts_file)
    
    def _create_config(self, n_ctx, ctx_init, csc, class_token_position, 
                       sccm_lambda, use_focal_loss, focal_alpha, focal_gamma, class_texts_file):
        """创建配置对象"""
        cfg = CfgNode()
        
        # 基本配置
        cfg.INPUT = CfgNode()
        cfg.INPUT.SIZE = [224, 224]
        
        cfg.OPTIM = CfgNode()
        cfg.OPTIM.MAX_EPOCH = 100
        
        # BiomedCoOp 配置
        cfg.TRAINER = CfgNode()
        cfg.TRAINER.BIOMEDCOOP = CfgNode()
        cfg.TRAINER.BIOMEDCOOP.N_CTX = n_ctx
        cfg.TRAINER.BIOMEDCOOP.CTX_INIT = ctx_init
        cfg.TRAINER.BIOMEDCOOP.CSC = csc
        cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = class_token_position
        cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = sccm_lambda
        cfg.TRAINER.BIOMEDCOOP.USE_FOCAL_LOSS = use_focal_loss
        cfg.TRAINER.BIOMEDCOOP.FOCAL_ALPHA = focal_alpha
        cfg.TRAINER.BIOMEDCOOP.FOCAL_GAMMA = focal_gamma
        cfg.TRAINER.BIOMEDCOOP.CLASS_TEXTS_FILE = class_texts_file
        
        return cfg
    
    def _setup_class_texts(self, class_texts_file):
        """设置类别文本（用于预测）"""
        if class_texts_file and osp.exists(class_texts_file):
            import json
            with open(class_texts_file, 'r', encoding='utf-8') as f:
                class_texts_dict = json.load(f)
            
            self.class_texts = []
            for classname in self.classnames:
                classname_clean = classname.replace("_", " ")
                if classname_clean in class_texts_dict:
                    self.class_texts.append(class_texts_dict[classname_clean])
                elif classname in class_texts_dict:
                    self.class_texts.append(class_texts_dict[classname])
                else:
                    self.class_texts.append(classname_clean)
        else:
            # 使用类别名称作为文本
            self.class_texts = [name.replace("_", " ") for name in self.classnames]
    
    def forward(self, images, texts=None, text_features=None):
        """
        前向传播（兼容 CLIPModel 接口）
        原始 BiomedCLIP：使用预编码的类别文本特征
        
        Args:
            images: 图像tensor [batch_size, 3, H, W]
            texts: 文本列表（可选，原始 BiomedCLIP 不使用）
            text_features: 文本特征（可选，原始 BiomedCLIP 使用预编码的特征）
        Returns:
            image_features: 图像特征 [batch_size, embed_dim]
            text_features: 文本特征 [num_classes, embed_dim]
        """
        # ========== 原始 BiomedCLIP：使用预编码的类别文本特征 ==========
        # 确保图像在正确的设备上（在类型转换之前）
        device = next(self.image_encoder.parameters()).device
        images = images.to(device)
        
        # 获取图像特征
        image_features = self.image_encoder(images.type(self.biomedcoop_model.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 使用预编码的类别文本特征（在 CustomCLIP.__init__ 中已预编码）
        class_text_features = self.biomedcoop_model.class_text_features
        
        # ========== 以下为 CoOp 相关代码（已注释，保留用于回溯）==========
        # # 获取所有类别的文本特征（通过 prompt learner）
        # prompts = self.biomedcoop_model.prompt_learner()
        # class_text_features = self.text_encoder(prompts, self.biomedcoop_model.tokenized_prompts)
        # class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
        
        return image_features, class_text_features
    
    def predict(self, images, class_texts=None, text_features=None):
        """
        预测图像的类别（兼容 CLIPModel 接口）
        原始 BiomedCLIP：使用预编码的类别文本特征
        
        Args:
            images: 图像tensor [batch_size, 3, H, W]
            class_texts: 类别文本列表（可选，原始 BiomedCLIP 不使用）
            text_features: 类别文本特征（可选，原始 BiomedCLIP 使用预编码的特征）
        Returns:
            predictions: 预测的类别索引 [batch_size]
            probabilities: 预测概率 [batch_size, num_classes]
        """
        self.biomedcoop_model.eval()
        
        with torch.no_grad():
            # 确保图像在正确的设备上（在类型转换之前）
            device = next(self.image_encoder.parameters()).device
            images = images.to(device)
            
            # 获取图像特征
            image_features = self.image_encoder(images.type(self.biomedcoop_model.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # ========== 原始 BiomedCLIP：使用预编码的类别文本特征 ==========
            class_text_features = self.biomedcoop_model.class_text_features
            
            # ========== 以下为 CoOp 相关代码（已注释，保留用于回溯）==========
            # # 获取所有类别的文本特征（通过 prompt learner）
            # prompts = self.biomedcoop_model.prompt_learner()
            # class_text_features = self.text_encoder(prompts, self.biomedcoop_model.tokenized_prompts)
            # class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            logit_scale = self.biomedcoop_model.logit_scale.exp()
            logits = logit_scale * image_features @ class_text_features.t()
            
            # 获取预测结果
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            return predictions, probabilities
    
    def train(self, mode=True):
        """设置训练/评估模式"""
        super().train(mode)
        if hasattr(self, 'biomedcoop_model'):
            self.biomedcoop_model.train(mode)
            # ========== 原始 BiomedCLIP：不使用 prompt_learner ==========
            # if hasattr(self.biomedcoop_model, 'prompt_learner'):
            #     self.biomedcoop_model.prompt_learner.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        return self.train(False)
    
    def compute_loss(self, images, labels):
        """
        计算 BiomedCLIP 的损失（用于训练）
        原始 BiomedCLIP 只使用对比损失
        
        Args:
            images: 图像tensor [batch_size, 3, H, W]
            labels: 标签 [batch_size]
        Returns:
            loss: 总损失（只包含对比损失）
            loss_dict: 损失字典（包含各个损失组件）
        """
        self.biomedcoop_model.train()
        # ========== 原始 BiomedCLIP：不使用 prompt_learner ==========
        # self.biomedcoop_model.prompt_learner.train()  # 已注释，不再使用 CoOp
        
        # 确保输入在正确的设备上
        device = next(self.biomedcoop_model.parameters()).device
        images = images.to(device)
        labels = labels.to(device)
        
        # 调用 BiomedCLIP 的 forward（训练模式）
        logits, loss_ce, contrastive_loss, loss_sccm = self.biomedcoop_model(images, labels)
        
        # ========== 原始 BiomedCLIP：只使用对比损失 ==========
        total_loss = contrastive_loss  # 只使用对比损失
        # ========== 以下为 CoOp 相关代码（已注释，保留用于回溯）==========
        # total_loss = loss_ce + contrastive_loss + loss_sccm
        
        loss_dict = {
            'total_loss': total_loss,
            'loss_ce': loss_ce,  # 占位符，实际为0
            'contrastive_loss': contrastive_loss,
            'loss_sccm': loss_sccm  # 占位符，实际为0
        }
        
        return total_loss, loss_dict


def create_biomedcoop_model(
    classnames,
    class_texts_file=None,
    embed_dim=512,
    temperature=0.07,
    n_ctx=4,
    ctx_init="a photo of a",
    csc=False,
    class_token_position="end",
    sccm_lambda=1.0,
    use_focal_loss=False,
    focal_alpha=0.25,
    focal_gamma=2.0,
    device='cuda'
):
    """
    创建 BiomedCoOp 适配器模型（兼容 train_clip.py）
    
    Args:
        classnames: 类别名称列表
        class_texts_file: 类别文本描述JSON文件路径
        embed_dim: 嵌入维度（BiomedCoOp 使用固定的 512）
        temperature: 温度参数
        n_ctx: 上下文token数量（默认4）
        ctx_init: 上下文初始化文本（默认"a photo of a"）
        csc: 是否使用类别特定的上下文（默认False）
        class_token_position: 类别token位置（默认"end"）
        sccm_lambda: SCCM损失权重（默认1.0）
        use_focal_loss: 是否使用 Focal Loss（默认False）
        focal_alpha: Focal Loss alpha 参数（默认0.25）
        focal_gamma: Focal Loss gamma 参数（默认2.0）
        device: 设备（默认'cuda'）
    
    Returns:
        model: BiomedCoOpAdapter 实例
    """
    model = BiomedCoOpAdapter(
        classnames=classnames,
        class_texts_file=class_texts_file,
        embed_dim=embed_dim,
        temperature=temperature,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        csc=csc,
        class_token_position=class_token_position,
        sccm_lambda=sccm_lambda,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        device=device
    )
    return model

