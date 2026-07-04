"""
Biomed-Super-CoLoRA 模型实现
结合了：
1. 文本端：BiomedCoOp 的 Context Optimization (CoOp)
2. 视觉端：LoRA (Low-Rank Adaptation) 到图像编码器
3. 损失函数：SuperCLIP + Visual Consistency Loss
"""

import copy
import os.path as osp
import json
import math
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

# 设置环境变量
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
os.environ.setdefault('HF_HUB_DOWNLOAD_MAX_RETRIES', '10')


# ========== LoRA 实现 ==========
class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层
    用于在保持原始线性层的同时，添加低秩适应参数
    """
    def __init__(self, linear_layer, r=4, alpha=1):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.lora_alpha = alpha
        
        # LoRA weights: A 和 B 矩阵
        self.lora_A = nn.Parameter(
            self.linear.weight.new_zeros((r, linear_layer.in_features))
        )
        self.lora_B = nn.Parameter(
            self.linear.weight.new_zeros((linear_layer.out_features, r))
        )
        self.scaling = self.lora_alpha / self.r
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # 原始前向传播
        result = self.linear(x)
        # LoRA 前向传播: x @ A^T @ B^T * scaling
        result += (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return result


def inject_lora(model, r=4, target_modules=None):
    """
    递归地为模型注入 LoRA 层
    
    Args:
        model: 要注入 LoRA 的模型
        r: LoRA 的秩（rank）
        target_modules: 目标模块名称列表（如果为 None，则自动检测 Linear 层）
    """
    if target_modules is None:
        # 默认目标：ViT 的 Attention 层中的 Linear 层
        target_modules = ['qkv', 'proj', 'fc1', 'fc2', 'to_qkv', 'to_out']
    
    for name, module in list(model.named_children()):
        # 检查是否是目标 Linear 层
        if isinstance(module, nn.Linear):
            # 检查名称是否包含目标关键字
            should_inject = any(target in name.lower() for target in target_modules)
            if should_inject:
                # 替换为 LoRA 层
                lora_layer = LoRALayer(module, r=r)
                setattr(model, name, lora_layer)
        else:
            # 递归处理子模块
            inject_lora(module, r=r, target_modules=target_modules)


# ========== TextEncoder ==========
class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts, tokenized_prompts):
        x = self.model.encode_text(prompts, True, tokenized_prompts)
        return x


# ========== PromptLearner (CoOp) ==========
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDCOOP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDCOOP.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 获取设备信息
        device = next(biomedclip_model.parameters()).device
        
        if ctx_init and n_ctx == 4:
            # 使用给定的词初始化上下文向量
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            if isinstance(prompt, torch.Tensor):
                prompt = prompt.to(device)
            elif isinstance(prompt, dict):
                prompt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prompt.items()}
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # 随机初始化
            if cfg.TRAINER.BIOMEDCOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames_clean = [name.replace("_", " ") for name in classnames]
        self.classnames = classnames_clean
        name_lens = [len(self.tokenizer(name)) for name in classnames_clean]
        prompts = [prompt_prefix + " " + name + "." for name in classnames_clean]

        # Tokenize prompts
        tokenized_prompts_list = []
        for p in prompts:
            tokenized = self.tokenizer(p)
            if isinstance(tokenized, torch.Tensor):
                tokenized = tokenized.to(device)
            elif isinstance(tokenized, dict):
                tokenized = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
            tokenized_prompts_list.append(tokenized)
        tokenized_prompts = torch.cat(tokenized_prompts_list)  # (n_cls, n_tkn)
        
        # 创建冻结的 BiomedCLIP 用于零样本特征和视觉一致性损失
        biomedclip_model_temp, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        biomedclip_model_temp = biomedclip_model_temp.float().eval().to(device)
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            # 保存冻结的图像编码器（用于视觉一致性损失）
            self.ZS_image_encoder = biomedclip_model_temp.visual
            
            # 加载类别文本描述（如果可用）
            class_texts_file = getattr(cfg.TRAINER.BIOMEDCOOP, 'CLASS_TEXTS_FILE', None)
            if class_texts_file is None:
                default_path = osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), 'class_texts_hip_prosthesis.json')
                if osp.exists(default_path):
                    class_texts_file = default_path
                else:
                    default_path = 'class_texts_hip_prosthesis.json'
                    if osp.exists(default_path):
                        class_texts_file = default_path
            
            if class_texts_file and osp.exists(class_texts_file):
                print(f"Loading class text descriptions from: {class_texts_file}")
                with open(class_texts_file, 'r', encoding='utf-8') as f:
                    class_texts_dict = json.load(f)
                
                class_text_descriptions = []
                for classname in classnames_clean:
                    description = None
                    for key, value in class_texts_dict.items():
                        if key.lower() == classname.lower() or key.replace(" ", "_").lower() == classname.lower():
                            description = value
                            break
                    if description is None:
                        description = classname
                    class_text_descriptions.append(description)
                
                # 编码类别文本描述
                device = next(biomedclip_model_temp.parameters()).device
                class_text_features = []
                for desc in class_text_descriptions:
                    desc_tokenized = self.tokenizer(desc)
                    if isinstance(desc_tokenized, torch.Tensor):
                        desc_tokenized = desc_tokenized.to(device)
                    else:
                        desc_tokenized = desc_tokenized['input_ids'].to(device) if 'input_ids' in desc_tokenized else desc_tokenized.to(device)
                    text_feat = biomedclip_model_temp.encode_text(desc_tokenized, normalize=False)
                    class_text_features.append(text_feat)
                
                self.class_text_features = torch.cat(class_text_features, dim=0)
                self.class_text_features = self.class_text_features / self.class_text_features.norm(dim=-1, keepdim=True)
                print(f"Loaded {len(class_text_descriptions)} class text descriptions")
            else:
                print(f"Warning: Class texts file not found at {class_texts_file}, using default templates")
                self.class_text_features = None
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
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
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
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
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError(f"Unknown class_token_position: {self.class_token_position}")

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


# ========== BiomedSuperCoLoRA 模型 ==========
class BiomedSuperCoLoRA(nn.Module):
    """
    Biomed-Super-CoLoRA 模型
    结合了 CoOp (文本端) + LoRA (视觉端) + SuperCLIP 损失 + Visual Consistency Loss
    """
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # 图像编码器（已注入 LoRA）
        self.image_encoder = biomedclip_model.visual
        
        # 文本编码器
        self.text_encoder = TextEncoder(biomedclip_model)
        
        # 其他参数
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        
        # 损失权重（从配置读取，或使用默认值）
        self.sccm_lambda = getattr(cfg.TRAINER.BIOMEDCOOP, 'SCCM_LAMBDA', 1.0)
        self.visual_reg_lambda = getattr(cfg.TRAINER.BIOMEDCOOP, 'VISUAL_REG_LAMBDA', 10.0)
        self.kdsp_lambda = getattr(cfg.TRAINER.BIOMEDCOOP, 'KDSP_LAMBDA', 0.0)  # 默认不使用 KDSP
        
        # Focal Loss 参数（可选）
        self.use_focal_loss = getattr(cfg.TRAINER.BIOMEDCOOP, 'USE_FOCAL_LOSS', False)
        self.focal_alpha = getattr(cfg.TRAINER.BIOMEDCOOP, 'FOCAL_ALPHA', 0.25)
        self.focal_gamma = getattr(cfg.TRAINER.BIOMEDCOOP, 'FOCAL_GAMMA', 2.0)

    def forward(self, image, label=None):
        device = image.device
        logit_scale = self.logit_scale.exp()
        
        # 确保 tokenized_prompts 在正确的设备上
        tokenized_prompts = self.tokenized_prompts.to(device)
        
        prompts = self.prompt_learner()
        if isinstance(prompts, torch.Tensor):
            prompts = prompts.to(device)
        
        # 计算图像和文本特征
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 图像特征（使用 LoRA 微调后的编码器）
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 计算分类 logits
        logits = logit_scale * image_features @ text_features.t()
        
        if self.training and label is not None:
            # ========== 训练模式：计算所有损失 ==========
            
            # 1. SuperCLIP 分类损失（CE Loss）
            if self.use_focal_loss:
                # Focal Loss
                probs = F.softmax(logits, dim=1)
                p_t = probs.gather(1, label.unsqueeze(1)).squeeze(1)
                focal_weight = (1 - p_t) ** self.focal_gamma
                loss_ce = -self.focal_alpha * focal_weight * torch.log(p_t + 1e-8)
                loss_ce = loss_ce.mean()
                if torch.isnan(loss_ce) or torch.isinf(loss_ce):
                    loss_ce = F.cross_entropy(logits, label)
            else:
                # 标准交叉熵损失
                loss_ce = F.cross_entropy(logits, label)
            
            # 2. 对比损失（CLIP loss）
            batch_size = image_features.shape[0]
            batch_text_features = text_features[label]  # [batch_size, embed_dim]
            batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)
            
            logits_per_image = logit_scale * image_features @ batch_text_features.t()
            logits_per_text = logit_scale * batch_text_features @ image_features.t()
            
            contrastive_labels = torch.arange(batch_size, device=device)
            contrastive_loss = (
                F.cross_entropy(logits_per_image, contrastive_labels) +
                F.cross_entropy(logits_per_text, contrastive_labels)
            ) / 2
            
            # 3. SCCM Loss（如果可用）
            if hasattr(self.prompt_learner, 'class_text_features') and self.prompt_learner.class_text_features is not None:
                class_text_features = self.prompt_learner.class_text_features.to(device)
                loss_sccm = F.mse_loss(text_features, class_text_features) * self.sccm_lambda
            else:
                loss_sccm = torch.tensor(0.0, device=device)
            
            # 4. Visual Consistency Loss（视觉一致性损失）
            with torch.no_grad():
                # 使用冻结的原始 BiomedCLIP 图像编码器
                frozen_image_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                frozen_image_features = frozen_image_features / frozen_image_features.norm(dim=-1, keepdim=True)
            
            # 计算当前图像特征与冻结特征之间的 MSE 损失
            loss_visual_reg = F.mse_loss(image_features, frozen_image_features) * self.visual_reg_lambda
            
            # 5. KDSP Loss（可选，如果启用）
            if self.kdsp_lambda > 0:
                # 这里可以实现 KDSP 损失（如果需要）
                # 暂时设为 0
                loss_kdsp = torch.tensor(0.0, device=device)
            else:
                loss_kdsp = torch.tensor(0.0, device=device)
            
            return logits, loss_ce, contrastive_loss, loss_sccm, loss_visual_reg, loss_kdsp
        else:
            # ========== 评估模式：只返回 logits ==========
            return logits


# ========== Trainer ==========
@TRAINER_REGISTRY.register()
class BiomedSuperCoLoRA_Trainer(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        try:
            biomedclip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            print("✓ 成功加载 BiomedCLIP 模型")
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            raise
        
        if cfg.TRAINER.BIOMEDCOOP.PREC == "fp32" or cfg.TRAINER.BIOMEDCOOP.PREC == "amp":
            biomedclip_model.float()

        # 注入 LoRA 到图像编码器
        lora_r = getattr(cfg.TRAINER.BIOMEDCOOP, 'LORA_R', 4)
        print(f"Injecting LoRA to Image Encoder (r={lora_r})...")
        inject_lora(biomedclip_model.visual, r=lora_r)
        print("✓ LoRA 注入完成")

        print("Building BiomedSuperCoLoRA model...")
        self.model = BiomedSuperCoLoRA(cfg, classnames, biomedclip_model)

        # 设置梯度：只训练 CoOp 和 LoRA 参数
        print("Setting up gradients...")
        print("Keeping gradients for: prompt_learner.ctx (CoOp) and LoRA parameters")
        
        # 1. 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 2. 解冻 CoOp 参数
        for name, param in self.model.named_parameters():
            if "prompt_learner.ctx" in name:
                param.requires_grad = True
        
        # 3. 解冻 LoRA 参数
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        # 统计参数
        enabled = set()
        total_params = 0
        coop_params = 0
        lora_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                param_count = param.numel()
                total_params += param_count
                if 'prompt_learner.ctx' in name:
                    coop_params += param_count
                elif 'lora_' in name:
                    lora_params += param_count
        
        print(f"Parameters to be updated: {total_params:,} parameters ({len(enabled)} parameter groups)")
        print(f"  - CoOp (prompt_learner.ctx): {coop_params:,} parameters")
        print(f"  - LoRA: {lora_params:,} parameters")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDCOOP.PREC == "amp" else None
        
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
                logits, loss_ce, contrastive_loss, loss_sccm, loss_visual_reg, loss_kdsp = model(image, label)
                # 总损失 = 分类损失 + 对比损失 + SCCM损失 + 视觉一致性损失 + KDSP损失
                loss = loss_ce + contrastive_loss + loss_sccm + loss_visual_reg + loss_kdsp
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss_ce, contrastive_loss, loss_sccm, loss_visual_reg, loss_kdsp = model(image, label)
            loss = loss_ce + contrastive_loss + loss_sccm + loss_visual_reg + loss_kdsp
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "contrastive_loss": contrastive_loss.item(),
            "loss_sccm": loss_sccm.item(),
            "loss_visual_reg": loss_visual_reg.item(),
            "loss_kdsp": loss_kdsp.item(),
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

            # 忽略固定的 token 向量
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

