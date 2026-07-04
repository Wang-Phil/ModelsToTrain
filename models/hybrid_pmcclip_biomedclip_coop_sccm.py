"""
混合模型 + CoOp + SCCM：PMC-CLIP 图像编码器 + BiomedCLIP 文本编码器 + CoOp Prompt Learning + SCCM 损失
"""

import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import os
import json
import requests
from tqdm import tqdm
import math

# dassl 相关导入（可选）
try:
    from dassl.engine import TRAINER_REGISTRY, TrainerX
    from dassl.utils import load_pretrained_weights, load_checkpoint
    from dassl.optim import build_optimizer, build_lr_scheduler
    from dassl.metrics import compute_accuracy
    DASSL_AVAILABLE = True
except ImportError:
    DASSL_AVAILABLE = False
    TRAINER_REGISTRY = None
    TrainerX = None
    load_pretrained_weights = None
    load_checkpoint = None
    build_optimizer = None
    build_lr_scheduler = None
    compute_accuracy = None

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

# 导入 BiomedCLIP 相关组件
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

# 设置环境变量
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
os.environ.setdefault('HF_HUB_DOWNLOAD_MAX_RETRIES', '10')

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


class TextEncoder(nn.Module):
    """文本编码器：使用 BiomedCLIP 编码文本（通过嵌入向量）"""
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype
        # 获取文本编码器的嵌入层
        self.word_embeddings = biomedclip_model.text.transformer.embeddings.word_embeddings

    def forward(self, prompts, tokenized_prompts):
        """
        使用 prompts（嵌入向量）和 tokenized_prompts 编码文本
        
        Args:
            prompts: [n_cls, seq_len, embed_dim] 嵌入向量
            tokenized_prompts: list of tokenized prompts (dict or tensor)
        
        Returns:
            text_features: [n_cls, 512] 文本特征（经过 text_projection）
        """
        device = next(self.model.parameters()).device
        n_cls = prompts.shape[0]
        
        # 获取 attention_mask
        if isinstance(tokenized_prompts, list) and len(tokenized_prompts) > 0:
            first_tokenized = tokenized_prompts[0]
            if isinstance(first_tokenized, dict):
                attention_mask = first_tokenized.get('attention_mask', None)
                if attention_mask is not None:
                    seq_len = attention_mask.shape[-1] if attention_mask.dim() > 0 else len(attention_mask)
                    attention_mask = attention_mask.unsqueeze(0).expand(n_cls, -1).to(device)
                else:
                    seq_len = prompts.shape[1]
                    attention_mask = torch.ones((n_cls, seq_len), device=device, dtype=torch.long)
            else:
                seq_len = prompts.shape[1]
                attention_mask = torch.ones((n_cls, seq_len), device=device, dtype=torch.long)
        else:
            seq_len = prompts.shape[1]
            attention_mask = torch.ones((n_cls, seq_len), device=device, dtype=torch.long)
        
        # 使用文本编码器的 transformer 部分处理嵌入向量
        output = self.model.text.transformer(
            inputs_embeds=prompts.type(self.dtype), 
            attention_mask=attention_mask
        )
        
        # 通过 pooler 和 proj 得到最终的文本特征 [n_cls, 512]
        if hasattr(self.model.text, 'pooler'):
            # pooler 需要 output (BaseModelOutput) 和 attention_mask 参数
            pooled_output = self.model.text.pooler(output, attention_mask=attention_mask)  # [n_cls, 768]
        else:
            # 如果没有 pooler，直接使用 CLS token
            pooled_output = output.last_hidden_state[:, 0, :]  # [n_cls, 768] - 使用 CLS token
        
        # 应用 proj 层（BiomedCLIP 使用 proj 而不是 text_projection）
        text_features = self.model.text.proj(pooled_output)  # [n_cls, 512]
        
        return text_features


class PromptLearner(nn.Module):
    """CoOp Prompt Learner（使用 BiomedCLIP 文本编码器）"""
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDCOOP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDCOOP.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768  # BiomedCLIP 文本编码器的 embedding 维度
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        # 获取设备
        device = next(biomedclip_model.parameters()).device
        
        if ctx_init:
            # 使用给定词初始化上下文向量
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
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        # Tokenize 所有 prompts（批量处理，每个类别一个 prompt）
        # 收集所有 input_ids
        all_input_ids = []
        all_attention_masks = []
        for p in prompts:
            tokenized = self.tokenizer(p)
            if isinstance(tokenized, torch.Tensor):
                all_input_ids.append(tokenized.to(device))
            elif isinstance(tokenized, dict):
                input_ids = tokenized.get('input_ids', tokenized)
                if isinstance(input_ids, torch.Tensor):
                    all_input_ids.append(input_ids.to(device))
                else:
                    all_input_ids.append(torch.tensor(input_ids, device=device))
                if 'attention_mask' in tokenized:
                    attn_mask = tokenized['attention_mask']
                    if isinstance(attn_mask, torch.Tensor):
                        all_attention_masks.append(attn_mask.to(device))
                    else:
                        all_attention_masks.append(torch.tensor(attn_mask, device=device))
            else:
                all_input_ids.append(torch.tensor(tokenized, device=device))
        
        # 堆叠为 [n_cls, seq_len]
        stacked_input_ids = torch.cat(all_input_ids, dim=0)  # [n_cls, seq_len]
        if all_attention_masks:
            stacked_attention_masks = torch.cat(all_attention_masks, dim=0)  # [n_cls, seq_len]
        else:
            stacked_attention_masks = torch.ones_like(stacked_input_ids)
        
        with torch.no_grad():
            # 获取所有类别的 embedding：[n_cls, seq_len, embed_dim]
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(stacked_input_ids).type(dtype)
        
        # token_prefix: [n_cls, 1, embed_dim] - 每个类别的 SOS token
        # token_suffix: [n_cls, *, embed_dim] - 每个类别的后续 tokens（类名 + EOS 等）
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # 跳过 ctx 位置后的所有 tokens
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        # 保存 tokenized prompts 用于 attention mask
        self.register_buffer("tokenized_input_ids", stacked_input_ids)
        self.register_buffer("tokenized_attention_masks", stacked_attention_masks)
        self.tokenized_prompts = {'input_ids': stacked_input_ids, 'attention_mask': stacked_attention_masks}
        self.name_lens = name_lens
        
        # 加载类别文本描述（用于 SCCM 损失）
        class_texts_file = getattr(cfg.TRAINER.BIOMEDCOOP, 'CLASS_TEXTS_FILE', None)
        if class_texts_file is None:
            default_path = osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), 'class_texts_hip_prosthesis.json')
            if osp.exists(default_path):
                class_texts_file = default_path
            else:
                default_path = 'class_texts_hip_prosthesis.json'
                if osp.exists(default_path):
                    class_texts_file = default_path
        
        # 预编码类别文本描述特征（用于 SCCM 损失）
        if class_texts_file and osp.exists(class_texts_file):
            print(f"加载类别文本描述用于 SCCM 损失: {class_texts_file}")
            with open(class_texts_file, 'r', encoding='utf-8') as f:
                class_texts_dict = json.load(f)
            
            class_text_descriptions = []
            for classname in classnames:
                description = None
                for key, value in class_texts_dict.items():
                    if (key.lower() == classname.lower() or
                        key.replace(" ", "_").lower() == classname.replace(" ", "_").lower() or
                        key.replace("_", " ").lower() == classname.lower()):
                        description = value
                        break
                if description is None:
                    description = f"a photo of {classname}."
                else:
                    description = f"{classname}: {description}"
                class_text_descriptions.append(description)
            
            # 编码类别文本描述
            with torch.no_grad():
                class_text_features_list = []
                for desc in class_text_descriptions:
                    desc_tokenized = self.tokenizer(desc)
                    if isinstance(desc_tokenized, torch.Tensor):
                        desc_tokenized = desc_tokenized.to(device)
                    elif isinstance(desc_tokenized, dict):
                        desc_tokenized = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in desc_tokenized.items()}
                    text_feat = biomedclip_model.encode_text(desc_tokenized, normalize=True)
                    class_text_features_list.append(text_feat)
                self.class_text_features = torch.cat(class_text_features_list, dim=0)  # [n_cls, 512]
                print(f"✓ 完成类别文本描述特征编码（用于 SCCM 损失）")
        else:
            print(f"警告: 未找到类别文本描述文件，SCCM 损失将使用默认 prompt")
            self.class_text_features = None
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """构建 prompts"""
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts
    
    def forward(self):
        """生成 prompts"""
        ctx = self.ctx  # [n_ctx, embed_dim]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, n_ctx, embed_dim]
        
        # token_prefix 和 token_suffix 已经是 [n_cls, ...] 形状
        prefix = self.token_prefix  # [n_cls, 1, embed_dim]
        suffix = self.token_suffix  # [n_cls, *, embed_dim]
        
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


class HybridCLIPWithCoOpSCCM(nn.Module):
    """
    混合模型 + CoOp + SCCM：
    - PMC-CLIP ResNet50 图像编码器（768维）
    - BiomedCLIP 文本编码器（512维）+ CoOp Prompt Learning
    - SCCM 损失：CoOp 特征 vs 类别文本描述特征
    - 损失函数：SCCM + CE + 对比损失 + 蒸馏损失
    """
    def __init__(self, cfg, classnames, pmcclip_image_encoder, biomedclip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.cfg = cfg
        self.dtype = torch.float32
        
        # ========== 图像编码器：PMC-CLIP ResNet50 ==========
        self.image_encoder = pmcclip_image_encoder
        self.image_embed_dim = 768
        
        # ========== 文本编码器：BiomedCLIP ==========
        self.biomedclip_model = biomedclip_model
        self.logit_scale = biomedclip_model.logit_scale
        self.text_embed_dim = 512
        
        # ========== CoOp Prompt Learner ==========
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.text_encoder = TextEncoder(biomedclip_model)
        
        # ========== 图像投影层：768 -> 512 ==========
        self.image_projection = nn.Linear(self.image_embed_dim, self.text_embed_dim)
        # 使用更小的初始化，避免初始阶段特征差异过大
        nn.init.normal_(self.image_projection.weight, std=0.02)
        nn.init.zeros_(self.image_projection.bias)
        
        # ========== Teacher（用于蒸馏）==========
        self.teacher_image_encoder = biomedclip_model.visual
        for param in self.teacher_image_encoder.parameters():
            param.requires_grad = False
        self.teacher_image_encoder.eval()
        
        # 损失权重
        self.sccm_lambda = getattr(cfg.TRAINER.BIOMEDCOOP, 'SCCM_LAMBDA', 1.0)
        self.classification_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'CLASSIFICATION_LOSS_WEIGHT', 0.5)
        self.contrastive_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'CONTRASTIVE_LOSS_WEIGHT', 0.5)
        self.distillation_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'DISTILLATION_LOSS_WEIGHT', 0.0)
        
        print(f"损失权重: SCCM={self.sccm_lambda}, CE={self.classification_loss_weight}, "
              f"Contrastive={self.contrastive_loss_weight}, Distill={self.distillation_loss_weight}")
    
    def forward(self, image, label=None):
        """前向传播"""
        device = image.device
        eps = 1e-8
        
        logit_scale = self.logit_scale.exp()
        
        # 获取图像特征
        image_features_raw = self.image_encoder(image.type(self.dtype))
        if isinstance(image_features_raw, dict):
            image_features_raw = image_features_raw['image_features']
        image_features_raw = image_features_raw / (image_features_raw.norm(dim=-1, keepdim=True) + eps)
        
        # 投影到 512 维
        image_features = self.image_projection(image_features_raw)
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + eps)
        
        # 生成 CoOp prompts
        prompts = self.prompt_learner()  # [n_cls, seq_len, embed_dim]
        
        # 编码 CoOp prompts
        # 需要处理 tokenized_prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        # 将 prompts 和 tokenized_prompts 传递给文本编码器
        text_features = self.text_encoder(prompts, tokenized_prompts)  # [n_cls, 512]
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
                    loss_ce = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss_ce = torch.tensor(0.0, device=device)
            
            # 2. 对比损失
            if self.contrastive_loss_weight > 0:
                # 获取每个样本对应的类别文本特征
                batch_text_features = text_features[label]  # [batch_size, 512]
                batch_text_features = batch_text_features / (batch_text_features.norm(dim=-1, keepdim=True) + eps)
                
                # 直接使用 logit_scale（不再除以温度，因为 logit_scale 已经是合适的缩放因子）
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
                    contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                contrastive_loss = torch.tensor(0.0, device=device)
            
            # 3. SCCM 损失：CoOp 特征 vs 类别文本描述特征
            if self.sccm_lambda > 0 and hasattr(self.prompt_learner, 'class_text_features') and self.prompt_learner.class_text_features is not None:
                class_text_features = self.prompt_learner.class_text_features.to(device)  # [n_cls, 512]
                class_text_features = class_text_features / (class_text_features.norm(dim=-1, keepdim=True) + eps)
                
                # CoOp 特征：text_features [n_cls, 512]
                # 类别文本描述特征：class_text_features [n_cls, 512]
                loss_mse = nn.MSELoss()
                loss_sccm = loss_mse(text_features, class_text_features) * self.sccm_lambda
                
                if torch.isnan(loss_sccm):
                    loss_sccm = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss_sccm = torch.tensor(0.0, device=device)
            
            # 4. 蒸馏损失
            if self.distillation_loss_weight > 0:
                with torch.no_grad():
                    teacher_dtype = next(self.teacher_image_encoder.parameters()).dtype
                    teacher_features = self.teacher_image_encoder(image.type(teacher_dtype))
                    if isinstance(teacher_features, dict):
                        teacher_features = teacher_features.get('image_features', teacher_features)
                    teacher_features = teacher_features / (teacher_features.norm(dim=-1, keepdim=True) + eps)
                
                loss_distill = F.mse_loss(image_features, teacher_features) * self.distillation_loss_weight
                
                if torch.isnan(loss_distill):
                    loss_distill = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss_distill = torch.tensor(0.0, device=device)
            
            return logits, loss_ce, contrastive_loss, loss_sccm, loss_distill
        else:
            return logits


# TrainerX 类（如果 dassl 可用）
if DASSL_AVAILABLE:
    @TRAINER_REGISTRY.register()
    class Hybrid_PMCResNet50_BiomedCLIP_CoOp_SCCM(TrainerX):
        def check_cfg(self, cfg):
            assert cfg.TRAINER.BIOMEDCOOP.PREC in ["fp16", "fp32", "amp"]
        
        def build_model(self):
            cfg = self.cfg
            classnames = self.dm.dataset.classnames
            
            print("=" * 80)
            print("构建混合模型 + CoOp + SCCM：PMC-CLIP ResNet50 + BiomedCLIP Text + CoOp + SCCM")
            print("=" * 80)
            
            # 检查 PMC-CLIP 文件
            print("检查 PMC-CLIP 模型文件...")
            for filename, url in files.items():
                filepath = os.path.join(directory, filename)
                if not os.path.exists(filepath):
                    print(f"{filename} 未找到，正在下载...")
                    download_file(url, filepath)
                else:
                    print(f"✓ {filename} 已存在")
            
            # 加载 PMC-CLIP ResNet50
            print("\n加载 PMC-CLIP ResNet50 图像编码器...")
            pmc_image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
            pmc_image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'), weights_only=True))
            
            # 加载 BiomedCLIP
            print("\n加载 BiomedCLIP 文本编码器...")
            try:
                biomedclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                print("✓ BiomedCLIP 模型加载完成")
            except Exception as e:
                print(f"✗ 加载失败: {e}")
                raise
            
            if cfg.TRAINER.BIOMEDCOOP.PREC == "fp32" or cfg.TRAINER.BIOMEDCOOP.PREC == "amp":
                biomedclip_model.float()
            
            pmc_image_encoder = pmc_image_encoder.to(self.device).eval()
            biomedclip_model = biomedclip_model.to(self.device).eval()
            
            # 构建模型
            print("\n构建混合模型 + CoOp + SCCM...")
            self.model = HybridCLIPWithCoOpSCCM(cfg, classnames, pmc_image_encoder, biomedclip_model)
            
            # 设置训练参数
            print("\n设置训练参数：")
            print("✓ 冻结文本编码器（BiomedCLIP）")
            print("✓ 训练图像编码器（PMC-CLIP ResNet50）")
            print("✓ 训练 CoOp prompts")
            print("✓ 训练投影层（image_projection: 768->512）")
            
            # 冻结文本编码器
            for param in self.model.biomedclip_model.parameters():
                param.requires_grad = False
            
            # 训练图像编码器和 CoOp
            for name, param in self.model.named_parameters():
                if "image_encoder" in name or "prompt_learner.ctx" in name or "image_projection" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            
            # 统计参数
            enabled = set()
            total_params = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
                    total_params += param.numel()
            
            print(f"参数统计: {total_params:,} 可训练参数")
            
            if cfg.MODEL.INIT_WEIGHTS:
                load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            
            self.model.to(self.device)
            self.optim = build_optimizer(self.model, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("model", self.model, self.optim, self.sched)
            self.scaler = GradScaler() if cfg.TRAINER.BIOMEDCOOP.PREC == "amp" else None
            
            device_count = torch.cuda.device_count()
            if device_count > 1:
                print(f"检测到 {device_count} 个 GPU，使用所有 GPU 进行训练!")
                self.model = nn.DataParallel(self.model)
        
        def forward_backward(self, batch):
            image, label = self.parse_batch_train(batch)
            model = self.model
            optim = self.optim
            scaler = self.scaler
            
            model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            
            prec = self.cfg.TRAINER.BIOMEDCOOP.PREC
            if prec == "amp":
                with autocast():
                    logits, loss_ce, contrastive_loss, loss_sccm, loss_distill = model(image, label)
                    loss = (model_ref.classification_loss_weight * loss_ce +
                           model_ref.contrastive_loss_weight * contrastive_loss +
                           loss_sccm +
                           loss_distill)
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                logits, loss_ce, contrastive_loss, loss_sccm, loss_distill = model(image, label)
                loss = (model_ref.classification_loss_weight * loss_ce +
                       model_ref.contrastive_loss_weight * contrastive_loss +
                       loss_sccm +
                       loss_distill)
                self.model_backward_and_update(loss)
            
            loss_summary = {
                "loss": loss.item(),
                "loss_ce": loss_ce.item(),
                "contrastive_loss": contrastive_loss.item(),
                "loss_sccm": loss_sccm.item(),
                "loss_distill": loss_distill.item(),
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
                return
            
            names = self.get_model_names()
            model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
            
            for name in names:
                model_path = osp.join(directory, name, model_file)
                if not osp.exists(model_path):
                    raise FileNotFoundError(f'Model not found at "{model_path}"')
                
                checkpoint = load_checkpoint(model_path)
                state_dict = checkpoint["state_dict"]
                epoch = checkpoint["epoch"]
                
                if "prompt_learner.token_prefix" in state_dict:
                    del state_dict["prompt_learner.token_prefix"]
                if "prompt_learner.token_suffix" in state_dict:
                    del state_dict["prompt_learner.token_suffix"]
                
                print(f"Loading weights to {name} from {model_path} (epoch = {epoch})")
                self._models[name].load_state_dict(state_dict, strict=False)

