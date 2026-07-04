import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import math
import os
import json
import requests
from tqdm import tqdm
from collections import OrderedDict

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy

import torch
import torch.nn.functional as F
from torch import nn

# 确保从本地 clip 目录导入 pmcclip
import sys
import os
# 添加本地 clip 目录到路径（如果还没有）
local_clip_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'clip')
if local_clip_path not in sys.path:
    sys.path.insert(0, local_clip_path)

try:
    from clip.pmcclip import ModifiedResNet
except ImportError:
    # 如果从本地 clip 导入失败，尝试直接导入
    try:
        from pmcclip import ModifiedResNet
    except ImportError:
        # 最后尝试从完整路径导入
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

# Directory where the files should be located
directory = "clip/checkpoints"

# File URLs
files = {
    "text_encoder.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth",
    "image_encoder(resnet50).pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth",
}

# Function to download a file
def download_file(url, filepath):
    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, "wb") as file:
            # Use tqdm to show the progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))  # Update progress bar by the chunk size
        print(f"{filepath} downloaded successfully.")
    else:
        print(f"Failed to download {filepath}. HTTP Status Code: {response.status_code}")

class TextEncoder(nn.Module):
    def __init__(self, pmcclip_model):
        super().__init__()
        self.model = pmcclip_model
        self.dtype = torch.float32
        self.text_encoder = pmcclip_model.text_encoder
        self.text_projection_layer = pmcclip_model.text_projection_layer

    def forward(self, prompts,tokenized_prompts):

        output = self.text_encoder(inputs_embeds=prompts.cuda(), attention_mask=tokenized_prompts['attention_mask'].cuda())
        pooler_output = output.pooler_output
        text_feature = pooler_output @ self.text_projection_layer

        return text_feature

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = torch.float32
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        vis_dim = 768
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = self.tokenizer(ctx_init, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']
            with torch.no_grad():
                embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(prompt.cuda()).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        
        if cfg.TRAINER.COOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']) \
                     for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = self.tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')

        with torch.no_grad():
            embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(tokenized_prompts['input_ids'].cuda()).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

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

class PMCCLIP(nn.Module):
    def __init__(self,image_encoder, text_encoder, projection_layer):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_projection_layer = projection_layer
        self.logit_scale = 4.4292
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
    def forward(self,image,text):
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        text_feature = self.text_encoder(input_ids)
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        image_feature = self.image_encoder(image)
        if isinstance(image_feature, dict):
            image_feature = image_feature['image_features']

        return image_feature, text_feature

class CustomCLIP(nn.Module):
    """
    原始 PMC-CLIP 模型（仅使用对比损失）
    不使用 CoOp prompt learning，只使用原始的 CLIP 对比学习损失
    """
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        # ========== 原始 PMC-CLIP 模型（不使用 CoOp）==========
        self.image_encoder = pmcclip_model.image_encoder
        self.pmcclip_model = pmcclip_model  # 保存完整模型引用，用于文本编码
        self.logit_scale = pmcclip_model.logit_scale
        self.dtype = torch.float32
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        # 存储类别名称（用于生成文本提示）
        self.classnames = [name.replace("_", " ") for name in classnames]
        
        # 获取 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        
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
            device = next(pmcclip_model.image_encoder.parameters()).device
            class_text_features_list = []
            for i, prompt in enumerate(self.class_prompts):
                if i % 3 == 0:  # 每3个类别打印一次进度
                    print(f"  编码进度: {i+1}/{len(self.class_prompts)}")
                encoded_input = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
                input_ids = encoded_input['input_ids'].to(device)
                text_feat = pmcclip_model.text_encoder(input_ids)
                pooler_output = text_feat.pooler_output
                text_feat = pooler_output @ pmcclip_model.text_projection_layer
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
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
        # self.prompt_learner = PromptLearner(cfg, classnames, pmcclip_model)
        # self.cfg = cfg
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # self.text_encoder = TextEncoder(pmcclip_model)

    def forward(self, image, label=None):
        """
        原始 PMC-CLIP 前向传播（仅使用对比损失）
        
        Args:
            image: 图像tensor [batch_size, 3, H, W]
            label: 标签 [batch_size]（训练时需要，用于计算对比损失）
        
        Returns:
            训练模式: (logits, contrastive_loss)
            评估模式: logits
        """
        device = next(self.image_encoder.parameters()).device
        logit_scale = math.exp(self.logit_scale)
        
        # 获取图像特征
        image_features = self.image_encoder(image.type(self.dtype))
        if isinstance(image_features, dict):
            image_features = image_features['image_features']
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
                encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
                input_ids = encoded_input['input_ids'].to(device)
                # 使用 PMC-CLIP 的文本编码器
                text_feat = self.pmcclip_model.text_encoder(input_ids)
                pooler_output = text_feat.pooler_output
                text_feat = pooler_output @ self.pmcclip_model.text_projection_layer
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
class CoOp_PMCCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Check for files in the directory and download if necessary
        for filename, url in files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")

        print(f"Loading PMC-CLIP (backbone: RN50)")
        image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'), weights_only=True))

        # Load Text Encoder
        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load(os.path.join(directory,'text_encoder.pth'), weights_only=True))

        # Load Text Proj Layer

        text_projection_layer = torch.load(os.path.join(directory,'text_projection_layer.pth'), weights_only=True)
        text_projection_layer = nn.Parameter(text_projection_layer)

        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_encoder = image_encoder.to(device).eval()
        text_encoder = text_encoder.to(device).eval()
        text_projection_layer = text_projection_layer.to(device)

        pmcclip_model = PMCCLIP(image_encoder, text_encoder, text_projection_layer).to(device).eval()

        print("Building original PMC-CLIP (contrastive loss only, no CoOp)")
        self.model = CustomCLIP(cfg, classnames, pmcclip_model)

        # ========== 原始 PMC-CLIP：只训练图像编码器 ==========
        print("Turning off gradients in the text encoder")
        print("Keeping gradients for: image_encoder only")
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

        # Double check - 统计实际参数数量
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
        # print("Turning off gradients in both the image and the text encoder")
        # name_to_update = "prompt_learner"
        # for name, param in self.model.named_parameters():
        #     if name_to_update not in name:
        #         param.requires_grad_(False)
        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        # # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # ========== 原始 PMC-CLIP：优化器包含所有可训练参数（主要是图像编码器）==========
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
            optim.zero_grad()
            loss.backward()
            optim.step()

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