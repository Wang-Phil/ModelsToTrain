import copy
import os.path as osp
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import os

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

# 设置环境变量，优先使用本地缓存的模型
# 配置 Hugging Face 中国镜像（如果未设置）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# 增加超时时间（从默认的10秒增加到300秒）
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
# 设置重试次数
os.environ.setdefault('HF_HUB_DOWNLOAD_MAX_RETRIES', '10')
# 如果网络有问题，可以设置离线模式（需要确保模型已缓存）
# os.environ.setdefault('HF_HUB_OFFLINE', '1')  # 取消注释以启用离线模式


# ========== 以下为 CoOp 相关类（已注释，保留用于回溯）==========
# class TextEncoder(nn.Module):
#     def __init__(self, biomedclip_model):
#         super().__init__()
#         self.model = biomedclip_model
#         self.dtype = biomedclip_model.text.transformer.dtype
# 
#     def forward(self, prompts,tokenized_prompts):
#         x = self.model.encode_text(prompts,True,tokenized_prompts)
#         return x

# class PromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, biomedclip_model):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = cfg.TRAINER.BIOMEDCOOP.N_CTX
#         ctx_init = cfg.TRAINER.BIOMEDCOOP.CTX_INIT
#         dtype = biomedclip_model.text.transformer.dtype
#         ctx_dim = 768
#         clip_imsize = 224
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
# 
#         # 获取设备信息（从 biomedclip_model 获取）
#         device = next(biomedclip_model.parameters()).device
#         
#         if ctx_init and n_ctx==4:
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             prompt = self.tokenizer(ctx_init)
#             # 确保 prompt 在正确的设备上
#             if isinstance(prompt, torch.Tensor):
#                 prompt = prompt.to(device)
#             elif isinstance(prompt, dict):
#                 prompt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prompt.items()}
#             with torch.no_grad():
#                 embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#         else:
#             # random initialization
#             if cfg.TRAINER.BIOMEDCOOP.CSC:
#                 print("Initializing class-specific contexts")
#                 ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             else:
#                 print("Initializing a generic context")
#                 ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)
#         print(f'Initial text context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
#         self.ctx = nn.Parameter(ctx_vectors)
# 
#         classnames_clean = [name.replace("_", " ") for name in classnames]
#         # Store original classnames for later use
#         self.classnames = classnames_clean
#         name_lens = [len(self.tokenizer(name)) for name in classnames_clean]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames_clean]
# 
#         # Tokenize prompts and ensure they are on the correct device
#         tokenized_prompts_list = []
#         for p in prompts:
#             tokenized = self.tokenizer(p)
#             if isinstance(tokenized, torch.Tensor):
#                 tokenized = tokenized.to(device)
#             elif isinstance(tokenized, dict):
#                 tokenized = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
#             tokenized_prompts_list.append(tokenized)
#         tokenized_prompts = torch.cat(tokenized_prompts_list)  # (n_cls, n_tkn)
#         
#         # Also create frozen CLIP for zero-shot features
#         # device 已经在上面获取了
#         biomedclip_model_temp,_ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#         biomedclip_model_temp = biomedclip_model_temp.float().eval().to(device)
#         with torch.no_grad():
#             embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
#             self.ZS_image_encoder = biomedclip_model_temp.visual
#             
#             # Load class text descriptions from JSON file
#             class_texts_file = getattr(cfg.TRAINER.BIOMEDCOOP, 'CLASS_TEXTS_FILE', None)
#             if class_texts_file is None:
#                 # Try default path
#                 default_path = osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), 'class_texts_hip_prosthesis.json')
#                 if osp.exists(default_path):
#                     class_texts_file = default_path
#                 else:
#                     # Try relative to current directory
#                     default_path = 'class_texts_hip_prosthesis.json'
#                     if osp.exists(default_path):
#                         class_texts_file = default_path
#             
#             if class_texts_file and osp.exists(class_texts_file):
#                 print(f"Loading class text descriptions from: {class_texts_file}")
#                 with open(class_texts_file, 'r', encoding='utf-8') as f:
#                     class_texts_dict = json.load(f)
#                 
#                 # Generate text features from class descriptions
#                 class_text_descriptions = []
#                 for classname in classnames_clean:
#                     # Try to find matching description (handle variations in naming)
#                     description = None
#                     for key, value in class_texts_dict.items():
#                         if key.lower() == classname.lower() or key.replace(" ", "_").lower() == classname.lower():
#                             description = value
#                             break
#                     
#                     if description is None:
#                         # Fallback to class name if description not found
#                         print(f"Warning: No description found for {classname}, using class name")
#                         description = classname
#                     
#                     class_text_descriptions.append(description)
#                 
#                 # Encode class descriptions
#                 # 获取设备信息
#                 device = next(biomedclip_model_temp.parameters()).device
#                 class_text_features = []
#                 for desc in class_text_descriptions:
#                     desc_tokenized = self.tokenizer(desc)
#                     if isinstance(desc_tokenized, torch.Tensor):
#                         desc_tokenized = desc_tokenized.to(device)
#                     else:
#                         desc_tokenized = desc_tokenized['input_ids'].to(device) if 'input_ids' in desc_tokenized else desc_tokenized.to(device)
#                     text_feat = biomedclip_model_temp.encode_text(desc_tokenized, normalize=False)
#                     class_text_features.append(text_feat)
#                 
#                 # Stack and normalize
#                 self.class_text_features = torch.cat(class_text_features, dim=0)  # [n_cls, dim]
#                 self.class_text_features = self.class_text_features / self.class_text_features.norm(dim=-1, keepdim=True)
#                 print(f"Loaded {len(class_text_descriptions)} class text descriptions")
#             else:
#                 print(f"Warning: Class texts file not found at {class_texts_file}, using default templates")
#                 # Fallback to default behavior (but we won't use it for SCCM)
#                 self.class_text_features = None
#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#         
#         # Store biomedclip_model_temp for encoding class texts during training
#         self.biomedclip_model_temp = biomedclip_model_temp
# 
#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens
#         self.class_token_position = cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION
# 
#     def construct_prompts(self, ctx, prefix, suffix, label=None):
#         # dim0 is either batch_size (during training) or n_cls (during testing)
#         # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
#         # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
#         # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
# 
#         prefix = self.token_prefix
#         suffix = self.token_suffix
# 
#         if self.class_token_position == "end":
#             prompts = torch.cat(
#                 [
#                     prefix,  # (n_cls, 1, dim)
#                     ctx,     # (n_cls, n_ctx, dim)
#                     suffix,  # (n_cls, *, dim)
#                 ],
#                 dim=1,
#             )
# 
#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
#                 ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,     # (1, 1, dim)
#                         ctx_i_half1,  # (1, n_ctx//2, dim)
#                         class_i,      # (1, name_len, dim)
#                         ctx_i_half2,  # (1, n_ctx//2, dim)
#                         suffix_i,     # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
# 
#         elif self.class_token_position == "front":
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i : i + 1, :, :]
#                 class_i = suffix[i : i + 1, :name_len, :]
#                 suffix_i = suffix[i : i + 1, name_len:, :]
#                 ctx_i = ctx[i : i + 1, :, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         class_i,   # (1, name_len, dim)
#                         ctx_i,     # (1, n_ctx, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
# 
#         else:
#             raise ValueError
# 
#         return prompts
# 
#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
# 
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#         prompts = self.construct_prompts(ctx, prefix, suffix)
# 
#         return prompts

class CustomCLIP(nn.Module):
    """
    原始 BiomedCLIP 模型（仅使用对比损失）
    不使用 CoOp prompt learning，不使用 SCCM 损失，不使用分类损失
    只使用原始的 CLIP 对比学习损失
    """
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        # ========== 原始 BiomedCLIP 模型（不使用 CoOp）==========
        # 直接使用原始的 BiomedCLIP 模型
        self.image_encoder = biomedclip_model.visual
        self.biomedclip_model = biomedclip_model  # 保存完整模型引用，用于文本编码
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        # 存储类别名称（用于生成文本提示）
        self.classnames = [name.replace("_", " ") for name in classnames]
        
        # 获取 tokenizer
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        # 加载类别文本描述（从 JSON 文件）
        class_texts_file = getattr(cfg.TRAINER.BIOMEDCOOP, 'CLASS_TEXTS_FILE', None)
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
        with torch.no_grad():
            # 安全地获取设备
            try:
                device = next(biomedclip_model.parameters()).device
            except StopIteration:
                device = biomedclip_model.logit_scale.device
            class_text_features_list = []
            for prompt in self.class_prompts:
                tokenized = self.tokenizer(prompt)
                if isinstance(tokenized, torch.Tensor):
                    tokenized = tokenized.to(device)
                elif isinstance(tokenized, dict):
                    tokenized = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
                text_feat = biomedclip_model.encode_text(tokenized, normalize=True)
                class_text_features_list.append(text_feat)
            self.register_buffer('class_text_features', torch.cat(class_text_features_list, dim=0))  # [n_cls, dim]
        
        # ========== 以下为 CoOp 相关代码（已注释，保留用于回溯）==========
        # self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        # self.cfg = cfg
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # Focal Loss 参数（从配置中读取，如果没有则使用默认值）
        # self.use_focal_loss = getattr(cfg.TRAINER.BIOMEDCOOP, 'USE_FOCAL_LOSS', False)
        # self.focal_alpha = getattr(cfg.TRAINER.BIOMEDCOOP, 'FOCAL_ALPHA', 0.25)
        # self.focal_gamma = getattr(cfg.TRAINER.BIOMEDCOOP, 'FOCAL_GAMMA', 2.0)
        # 
        # if self.use_focal_loss:
        #     print(f"启用 Focal Loss: alpha={self.focal_alpha}, gamma={self.focal_gamma}")

    def forward(self, image, label=None):
        """
        原始 BiomedCLIP 前向传播（仅使用对比损失）
        
        Args:
            image: 图像tensor [batch_size, 3, H, W]
            label: 标签 [batch_size]（训练时需要，用于计算对比损失）
        
        Returns:
            训练模式: (logits, contrastive_loss)
            评估模式: logits
        """
        # 安全地获取设备：优先从输入图像获取，如果失败则从模型参数获取
        try:
            device = image.device
        except:
            try:
                device = next(self.image_encoder.parameters()).device
            except StopIteration:
                device = self.logit_scale.device
        
        logit_scale = self.logit_scale.exp()
        
        # 获取图像特征
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if self.training and label is not None:
            # ========== 训练模式：使用对比损失 ==========
            batch_size = image_features.shape[0]
            
            # 为每个样本生成对应的文本提示并编码
            batch_texts = [self.class_prompts[label[i]] for i in range(batch_size)]
            batch_text_features_list = []
            for text in batch_texts:
                tokenized = self.tokenizer(text)
                if isinstance(tokenized, torch.Tensor):
                    tokenized = tokenized.to(device)
                elif isinstance(tokenized, dict):
                    tokenized = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
                # 使用 BiomedCLIP 的 encode_text 方法（完整模型的方法）
                text_feat = self.biomedclip_model.encode_text(tokenized, normalize=True)
                batch_text_features_list.append(text_feat)
            batch_text_features = torch.cat(batch_text_features_list, dim=0)  # [batch_size, dim]
            
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
            
            # 计算分类 logits（用于准确率计算，但不用于损失）
            logits = logit_scale * image_features @ self.class_text_features.t()
            
            # ========== 以下为 CoOp + 分类损失 + SCCM 损失代码（已注释，保留用于回溯）==========
            # # 确保 tokenized_prompts 在正确的设备上
            # tokenized_prompts = self.tokenized_prompts.to(device)
            # 
            # prompts = self.prompt_learner()
            # # 确保 prompts 在正确的设备上
            # if isinstance(prompts, torch.Tensor):
            #     prompts = prompts.to(device)
            # elif isinstance(prompts, (list, tuple)):
            #     prompts = [p.to(device) if isinstance(p, torch.Tensor) else p for p in prompts]
            # 
            # # Compute the prompted image and text features
            # text_features = self.text_encoder(prompts, tokenized_prompts)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # # Compute the prompted logits
            # logits = logit_scale * image_features @ text_features.t()
            # 
            # # 1. 分类损失：支持 Focal Loss 或标准交叉熵
            # # 这是图像特征与类别文本特征之间的分类损失
            # if self.use_focal_loss:
            #     # 使用 Focal Loss（用于处理类别不平衡）
            #     probs = F.softmax(logits, dim=1)
            #     p_t = probs.gather(1, label.unsqueeze(1)).squeeze(1)
            #     focal_weight = (1 - p_t) ** self.focal_gamma
            #     loss_ce = -self.focal_alpha * focal_weight * torch.log(p_t + 1e-8)
            #     loss_ce = loss_ce.mean()
            #     
            #     # 如果损失为 NaN 或 Inf，回退到标准交叉熵
            #     if torch.isnan(loss_ce) or torch.isinf(loss_ce):
            #         loss_ce = F.cross_entropy(logits, label)
            # else:
            #     # 使用标准交叉熵损失
            #     loss_ce = F.cross_entropy(logits, label)
            # 
            # # 2. 对比损失（CLIP loss）：图像特征与文本特征的对比学习
            # # 这是原始的 CLIP 对比损失，用于图像-文本对齐
            # # 注意：text_features 是 [n_cls, embed_dim]，我们需要为每个 batch 样本选择对应的类别文本特征
            # batch_size = image_features.shape[0]
            # 
            # # 为每个 batch 样本选择对应的类别文本特征
            # batch_text_features = text_features[label]  # [batch_size, embed_dim]
            # 
            # image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            # batch_text_features_norm = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)
            # 
            # # 计算相似度矩阵（batch 内的图像-文本配对）
            # logits_per_image = logit_scale * image_features_norm @ batch_text_features_norm.T
            # logits_per_text = logit_scale * batch_text_features_norm @ image_features_norm.T
            # 
            # # 创建对比学习的标签（对角线匹配：batch内每个图像对应一个文本）
            # contrastive_labels = torch.arange(batch_size, device=image_features.device)
            # 
            # # 双向对比损失
            # contrastive_loss = (
            #     F.cross_entropy(logits_per_image, contrastive_labels) +
            #     F.cross_entropy(logits_per_text, contrastive_labels)
            # ) / 2
            # 
            # # SCCM Loss: Compare CoOp learned text features with class text descriptions
            # if hasattr(self.prompt_learner, 'class_text_features') and self.prompt_learner.class_text_features is not None:
            #     # Use pre-computed class text features from JSON file
            #     # 确保 class_text_features 在正确的设备上
            #     class_text_features = self.prompt_learner.class_text_features.to(image_features.device)
            # else:
            #     # Fallback: encode class texts on-the-fly (if not pre-computed)
            #     with torch.no_grad():
            #         class_texts_file = getattr(self.cfg.TRAINER.BIOMEDCOOP, 'CLASS_TEXTS_FILE', None)
            #         if class_texts_file and osp.exists(class_texts_file):
            #             with open(class_texts_file, 'r', encoding='utf-8') as f:
            #                 class_texts_dict = json.load(f)
            #             
            #             classnames_clean = [name.replace("_", " ") for name in self.prompt_learner.classnames]
            #             class_text_descriptions = []
            #             for classname in classnames_clean:
            #                 description = None
            #                 for key, value in class_texts_dict.items():
            #                     if key.lower() == classname.lower() or key.replace(" ", "_").lower() == classname.lower():
            #                         description = value
            #                         break
            #                 if description is None:
            #                     description = classname
            #                 class_text_descriptions.append(description)
            #             
            #             # Encode descriptions
            #             # 获取设备信息
            #             device = next(self.prompt_learner.biomedclip_model_temp.parameters()).device
            #             class_text_features_list = []
            #             for desc in class_text_descriptions:
            #                 desc_tokenized = self.prompt_learner.tokenizer(desc)
            #                 if isinstance(desc_tokenized, torch.Tensor):
            #                     desc_tokenized = desc_tokenized.to(device)
            #                 else:
            #                     desc_tokenized = desc_tokenized['input_ids'].to(device) if 'input_ids' in desc_tokenized else desc_tokenized.to(device)
            #                 text_feat = self.prompt_learner.biomedclip_model_temp.encode_text(desc_tokenized, normalize=False)
            #                 class_text_features_list.append(text_feat)
            #             
            #             class_text_features = torch.cat(class_text_features_list, dim=0)
            #             class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
            #         else:
            #             # No class texts available, skip SCCM loss
            #             class_text_features = None
            # 
            # # 3. SCCM Loss: Compare CoOp learned text features with class text descriptions
            # if class_text_features is not None:
            #     # Compute SCCM loss: MSE between CoOp text features and class text descriptions
            #     # text_features: [n_cls, embed_dim] - CoOp learned features (所有类别的prompt features)
            #     # class_text_features: [n_cls, embed_dim] - Class text description features
            #     # 我们需要比较每个类别的 CoOp learned features 和对应的 class text description features
            #     # 注意：text_features 是 [n_cls, embed_dim]，不是 [batch_size, embed_dim]
            #     
            #     loss_mse = torch.nn.MSELoss()
            #     # 直接比较所有类别的 CoOp features 和 class text features
            #     loss_sccm = loss_mse(text_features, class_text_features) * self.cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA
            # else:
            #     # No class texts available, set SCCM loss to 0
            #     loss_sccm = torch.tensor(0.0, device=text_features.device)
            # 
            # return logits, loss_ce, contrastive_loss, loss_sccm
            
            # 返回 logits 和对比损失（兼容原有接口）
            loss_ce = torch.tensor(0.0, device=device)  # 占位符，不用于训练
            loss_sccm = torch.tensor(0.0, device=device)  # 占位符，不用于训练
            return logits, loss_ce, contrastive_loss, loss_sccm
        else:
            # ========== 评估模式：返回所有类别的 logits ==========
            logits = logit_scale * image_features @ self.class_text_features.t()
            return logits


@TRAINER_REGISTRY.register()
class BiomedCoOp_BiomedCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        print(f"注意：如果网络连接有问题，将自动使用本地缓存的模型")
        try:
            biomedclip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            print("✓ 成功加载 BiomedCLIP 模型")
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            print("提示：如果网络连接有问题，请确保模型已下载到本地缓存")
            print("缓存路径: ~/.cache/huggingface/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            raise
        if cfg.TRAINER.BIOMEDCOOP.PREC == "fp32" or cfg.TRAINER.BIOMEDCOOP.PREC == "amp":
            biomedclip_model.float()

        print("Building original BiomedCLIP (contrastive loss only, no CoOp)")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())

        # ========== 原始 BiomedCLIP：只训练图像编码器 ==========
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

        # Double check - 统计实际参数数量（元素个数）
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
        
        # ========== 以下为 CoOp 相关代码（已注释，保留用于回溯）==========
        # print("Keeping gradients for: prompt_learner.ctx (CoOp) and image_encoder (visual)")
        # names_to_update = ["prompt_learner.ctx"]
        # coop_params = 0
        # if 'prompt_learner.ctx' in name:
        #     coop_params += param_count
        # print(f"  - CoOp (prompt_learner.ctx): {coop_params:,} parameters")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # ========== 原始 BiomedCLIP：优化器包含所有可训练参数（主要是图像编码器）==========
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # 注册模型（不使用 prompt_learner，直接注册整个模型）
        self.register_model("model", self.model, self.optim, self.sched)
        
        # ========== 以下为 CoOp 相关代码（已注释，保留用于回溯）==========
        # # NOTE: only give prompt_learner to the optimizer
        # self.register_model("prompt_learner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDCOOP.PREC == "amp" else None
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

        prec = self.cfg.TRAINER.BIOMEDCOOP.PREC
        if prec == "amp":
            with autocast():
                logits, loss_ce, contrastive_loss, loss_sccm = model(image, label)
                # ========== 原始 BiomedCLIP：只使用对比损失 ==========
                loss = contrastive_loss  # 只使用对比损失
                # ========== 以下为 CoOp 相关代码（已注释，保留用于回溯）==========
                # # 总损失 = 分类损失 + 对比损失 + SCCM损失
                # loss = loss_ce + contrastive_loss + loss_sccm
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss_ce, contrastive_loss, loss_sccm = model(image, label)
            
            # ========== 原始 BiomedCLIP：只使用对比损失 ==========
            loss = contrastive_loss  # 只使用对比损失
            # ========== 以下为 CoOp 相关代码（已注释，保留用于回溯）==========
            # # 总损失 = 分类损失 + 对比损失 + SCCM损失
            # loss = loss_ce + contrastive_loss + loss_sccm
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),  # 占位符，实际为0
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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)