import os.path as osp
import json
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

# 导入系统安装的 CLIP 库（不是本地的 clip 目录）
import sys
import importlib
import os

# 先移除本地 clip（如果存在）
if 'clip' in sys.modules:
    clip_file = sys.modules['clip'].__file__ if hasattr(sys.modules['clip'], '__file__') else None
    if clip_file and 'ModelsTotrain/clip' in clip_file:
        del sys.modules['clip']
        # 也删除相关的子模块
        modules_to_remove = [k for k in list(sys.modules.keys()) if k.startswith('clip.')]
        for k in modules_to_remove:
            del sys.modules[k]

# 尝试从系统路径导入 CLIP 库
clip = None
CLIP_AVAILABLE = False

try:
    # 先确保移除本地 clip（如果存在）
    if 'clip' in sys.modules:
        clip_file = sys.modules['clip'].__file__ if hasattr(sys.modules['clip'], '__file__') else None
        if clip_file and 'ModelsTotrain/clip' in clip_file:
            del sys.modules['clip']
            # 也删除相关的子模块
            modules_to_remove = [k for k in list(sys.modules.keys()) if k.startswith('clip.')]
            for k in modules_to_remove:
                del sys.modules[k]
    
    # 方法1: 直接导入系统 CLIP 库（最简单可靠的方法）
    # 修改 sys.path，确保 site-packages 优先于当前目录
    original_path = sys.path[:]
    site_packages_paths = [p for p in sys.path if ('site-packages' in p or 'dist-packages' in p) and 'ModelsTotrain' not in p]
    other_paths = [p for p in sys.path if ('site-packages' not in p and 'dist-packages' not in p) or 'ModelsTotrain' in p]
    sys.path = site_packages_paths + other_paths
    
    try:
        import clip as _clip
        # 检查是否是有效的 CLIP 库（有 load 和 tokenize 方法）
        if hasattr(_clip, 'load') and hasattr(_clip, 'tokenize'):
            # 验证是否是本地空模块（检查文件路径）
            clip_file = getattr(_clip, '__file__', None)
            if clip_file:
                if 'ModelsTotrain/clip' not in clip_file:
                    # 这是系统 CLIP 库
                    clip = _clip
                    CLIP_AVAILABLE = True
                else:
                    # 这是本地空模块
                    CLIP_AVAILABLE = False
                    clip = None
            else:
                # 没有 __file__ 属性，但如果有 load 方法，应该是有效的
                clip = _clip
                CLIP_AVAILABLE = True
        else:
            # 没有 load 方法，不是有效的 CLIP 库
            CLIP_AVAILABLE = False
            clip = None
    finally:
        # 恢复原始 sys.path
        sys.path = original_path
        
except ImportError:
    CLIP_AVAILABLE = False
except Exception as e:
    # 如果导入过程中出现其他错误，也标记为不可用
    CLIP_AVAILABLE = False

# 如果 CLIP 库不可用，创建一个占位符
if not CLIP_AVAILABLE:
    # 创建一个占位符模块，避免后续导入错误
    class ClipPlaceholder:
        @staticmethod
        def load(*args, **kwargs):
            raise ImportError(
                "系统未安装 CLIP 库。请先安装：\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            )
        @staticmethod
        def tokenize(*args, **kwargs):
            raise ImportError("系统未安装 CLIP 库")
    
    clip = ClipPlaceholder()

# 尝试导入 SimpleTokenizer
_tokenizer = None
if CLIP_AVAILABLE:
    try:
        from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
        _tokenizer = _Tokenizer()
    except ImportError:
        try:
            from clip import SimpleTokenizer as _Tokenizer
            _tokenizer = _Tokenizer()
        except ImportError:
            # 如果没有 SimpleTokenizer，创建一个占位符
            class _Tokenizer:
                @staticmethod
                def encode(text):
                    return clip.tokenize(text)
            _tokenizer = _Tokenizer()
else:
    # 创建一个占位符 tokenizer
    class _Tokenizer:
        @staticmethod
        def encode(text):
            raise ImportError("系统未安装 CLIP 库")
    _tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    
    # 首先检查 CLIP 库是否可用
    if not CLIP_AVAILABLE or not hasattr(clip, 'load'):
        raise ImportError(
            f"无法加载 CLIP 模型 {backbone_name}：系统未安装标准 CLIP 库。\n"
            "请先安装 CLIP 库：\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "\n"
            "或者训练其他模型（PMC-CLIP 不依赖系统 CLIP 库）：\n"
            "  --model-type pmcclip"
        )
    
    # 使用 torch.hub.load 加载 CLIP 模型（更可靠的方法）
    try:
        # 方法1: 尝试使用 clip.load（推荐）
        model, preprocess = clip.load(backbone_name, device="cpu", download_root=None)
        return model
    except Exception as e1:
        print(f"使用 clip.load 失败: {e1}")
        try:
            # 方法2: 尝试使用 clip._MODELS 和手动下载
            if hasattr(clip, '_MODELS'):
                url = clip._MODELS[backbone_name]
                import urllib.request
                cache_dir = os.path.expanduser("~/.cache/clip")
                os.makedirs(cache_dir, exist_ok=True)
                filename = os.path.basename(url)
                model_path = os.path.join(cache_dir, filename)
                
                if not os.path.exists(model_path):
                    print(f"下载 CLIP 模型到 {model_path}...")
                    urllib.request.urlretrieve(url, model_path)
                
                try:
                    # loading JIT archive
                    model = torch.jit.load(model_path, map_location="cpu").eval()
                    state_dict = None
                except RuntimeError:
                    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                
                model = clip.build_model(state_dict or model.state_dict())
                return model
            else:
                raise AttributeError("clip._MODELS 不存在")
        except Exception as e2:
            print(f"使用手动下载失败: {e2}")
            # 方法3: 尝试直接使用 clip.load 的备用参数
            try:
                model, preprocess = clip.load(backbone_name, device="cpu")
                return model
            except Exception as e3:
                raise RuntimeError(f"无法加载 CLIP 模型 {backbone_name}: {e3}")


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    """
    原始 CLIP 模型（仅使用对比损失）
    不使用 CoOp prompt learning，只使用原始的 CLIP 对比学习损失
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # ========== 原始 CLIP 模型（不使用 CoOp）==========
        self.image_encoder = clip_model.visual
        self.clip_model = clip_model  # 保存完整模型引用，用于文本编码
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)
        
        # 存储类别名称（用于生成文本提示）
        self.classnames = [name.replace("_", " ") for name in classnames]
        
        # 加载类别文本描述（从 JSON 文件）
        class_texts_file = getattr(cfg.TRAINER.COOP, 'CLASS_TEXTS_FILE', None)
        if class_texts_file is None:
            # 尝试默认路径
            default_path = osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), 'class_texts_hip_prosthesis.json')
            if osp.exists(default_path):
                class_texts_file = default_path
            else:
                # 尝试相对路径
                default_path = 'class_texts_hip_prosthesis.json'
                if osp.exists(default_path):
                    class_texts_file = default_path
        
        # 为每个类别获取文本描述
        self.class_prompts = []
        if class_texts_file and osp.exists(class_texts_file):
            print(f"加载类别文本描述: {class_texts_file}")
            with open(class_texts_file, 'r', encoding='utf-8') as f:
                class_texts_dict = json.load(f)
            
            # 为每个类别匹配对应的描述文本
            for classname in self.classnames:
                description = None
                # 尝试多种匹配方式
                for key, value in class_texts_dict.items():
                    if (key.lower() == classname.lower() or 
                        key.replace(" ", "_").lower() == classname.replace(" ", "_").lower() or
                        key.replace("_", " ").lower() == classname.lower()):
                        description = value
                        break
                
                if description is None:
                    # 如果找不到匹配的描述，使用默认 prompt
                    print(f"警告: 未找到类别 '{classname}' 的文本描述，使用默认 prompt")
                    description = f"a photo of {classname}."
                else:
                    # 将类别名加入到描述文本中，格式: "类别名: 描述文本"
                    description = f"{classname}: {description}"
                    print(f"  {classname}: 使用文本描述（包含类别名）")
                
                self.class_prompts.append(description)
        else:
            # 如果没有 JSON 文件，使用默认 prompt
            print(f"警告: 未找到类别文本描述文件，使用默认 prompt")
            if class_texts_file:
                print(f"  尝试路径: {class_texts_file}")
            self.class_prompts = [f"a photo of {name}." for name in self.classnames]
        
        # 预编码所有类别的文本特征（用于推理）
        print(f"预编码 {len(self.class_prompts)} 个类别的文本特征...")
        with torch.no_grad():
            device = next(clip_model.parameters()).device
            class_text_features_list = []
            for i, prompt in enumerate(self.class_prompts):
                if i % 3 == 0:  # 每3个类别打印一次进度
                    print(f"  编码进度: {i+1}/{len(self.class_prompts)}")
                tokenized = clip.tokenize(prompt)
                text_feat = clip_model.encode_text(tokenized.to(device))
                class_text_features_list.append(text_feat)
            print(f"✓ 完成文本特征预编码")
            self.register_buffer('class_text_features', torch.cat(class_text_features_list, dim=0))  # [n_cls, dim]
        
        # Focal Loss 参数（从配置中读取，如果没有则使用默认值）
        self.use_focal_loss = getattr(cfg.TRAINER.COOP, 'USE_FOCAL_LOSS', True)  # 默认启用
        self.focal_alpha = getattr(cfg.TRAINER.COOP, 'FOCAL_ALPHA', 0.25)
        self.focal_gamma = getattr(cfg.TRAINER.COOP, 'FOCAL_GAMMA', 2.0)
        self.classification_loss_weight = getattr(cfg.TRAINER.COOP, 'CLASSIFICATION_LOSS_WEIGHT', 0.5)  # 分类损失权重
        self.contrastive_loss_weight = getattr(cfg.TRAINER.COOP, 'CONTRASTIVE_LOSS_WEIGHT', 0.5)  # 对比损失权重
        
        if self.use_focal_loss:
            print(f"启用 Focal Loss: alpha={self.focal_alpha}, gamma={self.focal_gamma}")
        print(f"损失权重: 分类损失={self.classification_loss_weight}, 对比损失={self.contrastive_loss_weight}")
        
        # ========== 以下为 CoOp 相关代码（已注释，保留用于回溯）==========
        # self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # self.text_encoder = TextEncoder(clip_model)

    def forward(self, image, label=None):
        """
        原始 CLIP 前向传播（仅使用对比损失）
        
        Args:
            image: 图像tensor [batch_size, 3, H, W]
            label: 标签 [batch_size]（训练时需要，用于计算对比损失）
        
        Returns:
            训练模式: (logits, contrastive_loss)
            评估模式: logits
        """
        device = next(self.image_encoder.parameters()).device
        logit_scale = self.logit_scale.exp()
        
        # 获取图像特征
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if self.training and label is not None:
            # ========== 训练模式：使用分类损失 + 对比损失 ==========
            batch_size = image_features.shape[0]
            
            # 1. 计算分类 logits（使用所有类别的文本特征）
            logits = logit_scale * image_features @ self.class_text_features.t()  # [batch_size, n_cls]
            
            # 2. 分类损失：支持 Focal Loss 或标准交叉熵
            if self.use_focal_loss:
                # 使用 Focal Loss（用于处理类别不平衡）
                probs = F.softmax(logits, dim=1)
                p_t = probs.gather(1, label.unsqueeze(1)).squeeze(1)  # [batch_size]
                focal_weight = (1 - p_t) ** self.focal_gamma
                loss_ce = -self.focal_alpha * focal_weight * torch.log(p_t + 1e-8)
                loss_ce = loss_ce.mean()
                
                # 如果损失为 NaN 或 Inf，回退到标准交叉熵
                if torch.isnan(loss_ce) or torch.isinf(loss_ce):
                    loss_ce = F.cross_entropy(logits, label)
            else:
                # 使用标准交叉熵损失
                loss_ce = F.cross_entropy(logits, label)
            
            # 3. 对比损失（CLIP loss）：图像特征与文本特征的对比学习
            # 为每个样本生成对应的文本提示并编码
            batch_texts = [self.class_prompts[label[i]] for i in range(batch_size)]
            batch_text_features_list = []
            for text in batch_texts:
                tokenized = clip.tokenize(text).to(device)
                # 使用 CLIP 的 encode_text 方法
                text_feat = self.clip_model.encode_text(tokenized)
                batch_text_features_list.append(text_feat)
            batch_text_features = torch.cat(batch_text_features_list, dim=0)  # [batch_size, dim]
            batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度矩阵
            logits_per_image = logit_scale * image_features @ batch_text_features.t()
            logits_per_text = logit_scale * batch_text_features @ image_features.t()
            
            # 创建对比学习的标签（对角线匹配：batch内每个图像对应一个文本）
            contrastive_labels = torch.arange(batch_size, device=image_features.device)
            
            # 双向对比损失（原始 CLIP 损失）
            contrastive_loss = (
                F.cross_entropy(logits_per_image, contrastive_labels) +
                F.cross_entropy(logits_per_text, contrastive_labels)
            ) / 2
            
            # 返回 logits 和损失（兼容原有接口）
            loss_sccm = torch.tensor(0.0, device=device)  # 占位符，不用于训练
            return logits, loss_ce, contrastive_loss, loss_sccm
        else:
            # ========== 评估模式：返回所有类别的 logits ==========
            logits = logit_scale * image_features @ self.class_text_features.t()
            return logits


@TRAINER_REGISTRY.register()
class CoOp_CLIP(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building original CLIP (contrastive loss only, no CoOp)")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # ========== 原始 CLIP：只训练图像编码器 ==========
        print("Turning off gradients in the text encoder")
        print("Keeping gradients for: image_encoder (visual) only")
        names_to_update = []
        
        # 添加图像编码器的所有参数
        for name, param in self.model.named_parameters():
            if name.startswith("image_encoder."):
                names_to_update.append(name)
                param.requires_grad_(True)

        # 冻结其他所有参数（文本编码器、logit_scale等）
        for name, param in self.model.named_parameters():
            if name not in names_to_update:
                param.requires_grad_(False)

        # ========== 以下为 CoOp 相关代码（已注释，保留用于回溯）==========
        # print("Turning off gradients in both the image and the text encoder")
        # for name, param in self.model.named_parameters():
        #     if "prompt_learner" not in name:
        #         param.requires_grad_(False)
        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        # # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # ========== 原始 CLIP：优化器包含所有可训练参数（主要是图像编码器）==========
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # 注册模型（不使用 prompt_learner，直接注册整个模型）
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                logits, loss_ce, contrastive_loss, loss_sccm = model(image, label)
                # ========== 结合分类损失和对比损失 ==========
                classification_weight = getattr(self.cfg.TRAINER.COOP, 'CLASSIFICATION_LOSS_WEIGHT', 0.5)
                contrastive_weight = getattr(self.cfg.TRAINER.COOP, 'CONTRASTIVE_LOSS_WEIGHT', 0.5)
                loss = classification_weight * loss_ce + contrastive_weight * contrastive_loss
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss_ce, contrastive_loss, loss_sccm = model(image, label)
            
            # ========== 结合分类损失和对比损失 ==========
            classification_weight = getattr(self.cfg.TRAINER.COOP, 'CLASSIFICATION_LOSS_WEIGHT', 0.5)
            contrastive_weight = getattr(self.cfg.TRAINER.COOP, 'CONTRASTIVE_LOSS_WEIGHT', 0.5)
            loss = classification_weight * loss_ce + contrastive_weight * contrastive_loss
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),  # 分类损失（Focal Loss 或交叉熵）
            "contrastive_loss": contrastive_loss.item(),
            "loss_sccm": loss_sccm.item(),  # 占位符，实际为0
            "acc": compute_accuracy(logits, label)[0].item(),
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

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
