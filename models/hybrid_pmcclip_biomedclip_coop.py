"""
混合模型 + CoOp：PMC-CLIP ResNet50 图像编码器 + BiomedCLIP 文本编码器 + CoOp Prompt Learning
参考 biomedcoop_biomedclip.py 中的 CoOp 实现
"""

import copy
import os.path as osp
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import os

# 尝试导入 dassl（可选）
try:
    from dassl.engine import TRAINER_REGISTRY, TrainerX
    from dassl.utils import load_pretrained_weights, load_checkpoint
    from dassl.optim import build_optimizer, build_lr_scheduler
    from dassl.metrics import compute_accuracy
    DASSL_AVAILABLE = True
except ImportError:
    DASSL_AVAILABLE = False
    print("警告: dassl 不可用，TrainerX 相关功能将被禁用")

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

# 设置环境变量，优先使用本地缓存的模型
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
os.environ.setdefault('HF_HUB_DOWNLOAD_MAX_RETRIES', '10')


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
            tokenized_prompts: dict with 'input_ids' and 'attention_mask' or tensor
        
        Returns:
            text_features: [n_cls, 512] 文本特征（经过 text_projection）
        """
        # prompts 是嵌入向量 [n_cls, seq_len, embed_dim]
        # 需要获取 attention_mask
        if isinstance(tokenized_prompts, dict):
            attention_mask = tokenized_prompts.get('attention_mask', None)
        else:
            # 假设所有位置都是有效的
            attention_mask = torch.ones(prompts.shape[:2], device=prompts.device, dtype=torch.long)
        
        # 使用文本编码器的 transformer 部分处理嵌入向量
        output = self.model.text.transformer(
            inputs_embeds=prompts.type(self.dtype), 
            attention_mask=attention_mask.to(prompts.device)
        )
        
        # 通过 pooler 和 proj 得到最终的文本特征 [n_cls, 512]
        # BiomedCLIP 的文本编码器使用 pooler 和 proj（不是 text_projection）
        # pooler 接受 BaseModelOutput 对象和 attention_mask
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
    """CoOp Prompt Learner：学习可训练的上下文 tokens"""
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDCOOP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDCOOP.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768  # BiomedCLIP 的文本嵌入维度
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 获取设备信息
        device = next(biomedclip_model.parameters()).device
        
        if ctx_init and n_ctx == 4:
            # 使用给定词汇初始化上下文向量
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
        
        # 创建冻结的 BiomedCLIP 用于零样本特征
        biomedclip_model_temp, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        biomedclip_model_temp = biomedclip_model_temp.float().eval().to(device)
        # 确保 biomedclip_model_temp 的所有参数都被冻结
        for param in biomedclip_model_temp.parameters():
            param.requires_grad = False
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = biomedclip_model_temp.visual
            
            # 加载类别文本描述（可选）
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
                        print(f"Warning: No description found for {classname}, using class name")
                        description = classname
                    class_text_descriptions.append(description)
                
                # 编码类别描述
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
                
                self.class_text_features = torch.cat(class_text_features, dim=0)  # [n_cls, dim]
                self.class_text_features = self.class_text_features / self.class_text_features.norm(dim=-1, keepdim=True)
                print(f"Loaded {len(class_text_descriptions)} class text descriptions")
            else:
                print(f"Warning: Class texts file not found at {class_texts_file}, using default templates")
                self.class_text_features = None
        
        # 保存 token 向量
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.biomedclip_model_temp = biomedclip_model_temp
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """构建 prompts"""
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
        """前向传播：生成 prompts"""
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class HybridCLIPWithCoOp(nn.Module):
    """
    混合模型 + CoOp：PMC-CLIP ResNet50 图像编码器 + BiomedCLIP 文本编码器 + CoOp Prompt Learning
    使用 CoOp 学习可训练的 prompt，支持分类损失、对比损失和蒸馏损失
    """
    def __init__(self, cfg, classnames, pmcclip_image_encoder, biomedclip_model):
        super().__init__()
        self.n_cls = len(classnames)

        # ========== 图像编码器：使用 PMC-CLIP 的 ResNet50 ==========
        self.image_encoder = pmcclip_image_encoder
        self.dtype = torch.float32
        self.image_embed_dim = 768  # PMC-CLIP ResNet50 输出维度

        # ========== 文本编码器：使用 BiomedCLIP ==========
        self.biomedclip_model = biomedclip_model  # BiomedCLIP 完整模型，用于文本编码
        self.logit_scale = biomedclip_model.logit_scale
        self.text_embed_dim = 512  # BiomedCLIP 文本特征维度

        # ========== CoOp Prompt Learner ==========
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.text_encoder = TextEncoder(biomedclip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # ========== 蒸馏：BiomedCLIP 图像编码器作为 teacher ==========
        self.teacher_image_encoder = biomedclip_model.visual
        for param in self.teacher_image_encoder.parameters():
            param.requires_grad = False
        self.teacher_image_encoder.eval()
        self.teacher_image_embed_dim = 512

        # ========== 投影层：将文本特征从 512 维投影到 768 维 ==========
        # CoOp 学习的文本特征需要投影到图像特征维度
        self.text_projection = nn.Linear(self.text_embed_dim, self.image_embed_dim)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)

        # ========== 蒸馏投影层：将 student 特征投影到 teacher 维度 ==========
        self.distill_projection = nn.Linear(self.image_embed_dim, self.teacher_image_embed_dim)
        nn.init.xavier_uniform_(self.distill_projection.weight)
        nn.init.zeros_(self.distill_projection.bias)

        # ========== 损失函数权重配置 ==========
        self.classification_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'CLASSIFICATION_LOSS_WEIGHT', 0.5)
        self.contrastive_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'CONTRASTIVE_LOSS_WEIGHT', 0.5)
        self.distillation_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'DISTILLATION_LOSS_WEIGHT', 0.0)

        print(f"损失权重: 分类损失={self.classification_loss_weight}, 对比损失={self.contrastive_loss_weight}, 蒸馏损失={self.distillation_loss_weight}")

    def forward(self, image, label=None):
        """
        前向传播：使用 CoOp prompts + 分类损失 + 对比损失 + 蒸馏损失

        Args:
            image: 图像tensor [batch_size, 3, H, W]
            label: 标签 [batch_size]（训练时需要）

        Returns:
            训练模式: (logits, loss_ce, contrastive_loss, loss_distill)
            评估模式: logits
        """
        device = image.device
        logit_scale = self.logit_scale.exp()

        # 获取图像特征（PMC-CLIP ResNet50）
        image_features = self.image_encoder(image.type(self.dtype))
        if isinstance(image_features, dict):
            image_features = image_features['image_features']
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.training and label is not None:
            # ========== 训练模式 ==========
            batch_size = image_features.shape[0]

            # 1. 使用 CoOp 生成 prompts 并编码文本特征
            prompts = self.prompt_learner()  # [n_cls, seq_len, embed_dim]
            tokenized_prompts = self.tokenized_prompts.to(device)
            
            # 编码所有类别的文本特征
            text_features = self.text_encoder(prompts, tokenized_prompts)  # [n_cls, 512]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 将文本特征投影到图像特征维度（512 -> 768）
            text_features_proj = self.text_projection(text_features)  # [n_cls, 768]
            text_features_proj = text_features_proj / text_features_proj.norm(dim=-1, keepdim=True)

            # 2. 计算分类 logits
            logits = logit_scale * image_features @ text_features_proj.t()  # [batch_size, n_cls]

            # 3. 分类损失
            if self.classification_loss_weight > 0:
                loss_ce = F.cross_entropy(logits, label)
            else:
                loss_ce = torch.tensor(0.0, device=device)

            # 4. 对比损失：图像特征与对应类别文本特征的对比学习
            if self.contrastive_loss_weight > 0:
                # 为每个样本选择对应的类别文本特征
                batch_text_features = text_features_proj[label]  # [batch_size, 768]
                batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)

                # 计算相似度矩阵
                logits_per_image = logit_scale * image_features @ batch_text_features.t()
                logits_per_text = logit_scale * batch_text_features @ image_features.t()

                # 创建对比学习的标签
                contrastive_labels = torch.arange(batch_size, device=device)

                # 双向对比损失
                contrastive_loss = (
                    F.cross_entropy(logits_per_image, contrastive_labels) +
                    F.cross_entropy(logits_per_text, contrastive_labels)
                ) / 2
            else:
                contrastive_loss = torch.tensor(0.0, device=device)

            # 5. 蒸馏损失：将 BiomedCLIP 图像编码器的知识蒸馏到 PMC-CLIP ResNet50
            if self.distillation_loss_weight > 0:
                student_features = image_features  # [batch_size, 768]
                
                with torch.no_grad():
                    teacher_dtype = next(self.teacher_image_encoder.parameters()).dtype
                    teacher_features = self.teacher_image_encoder(image.type(teacher_dtype))
                    if isinstance(teacher_features, dict):
                        teacher_features = teacher_features.get('image_features', teacher_features)
                    teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)  # [batch_size, 512]
                
                # 将 student 特征投影到 teacher 特征维度
                student_features_proj = self.distill_projection(student_features)  # [batch_size, 512]
                student_features_proj = student_features_proj / student_features_proj.norm(dim=-1, keepdim=True)
                
                # 计算蒸馏损失（MSE）
                loss_distill = F.mse_loss(student_features_proj, teacher_features)
            else:
                loss_distill = torch.tensor(0.0, device=device)

            return logits, loss_ce, contrastive_loss, loss_distill
        else:
            # ========== 评估模式 ==========
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts.to(device)
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 投影到图像特征维度
            text_features_proj = self.text_projection(text_features)  # [n_cls, 768]
            text_features_proj = text_features_proj / text_features_proj.norm(dim=-1, keepdim=True)
            
            logits = logit_scale * image_features @ text_features_proj.t()
            return logits


# TrainerX 类（如果 dassl 可用）
if DASSL_AVAILABLE:
    @TRAINER_REGISTRY.register()
    class Hybrid_PMCResNet50_BiomedCLIP_CoOp(TrainerX):
        def check_cfg(self, cfg):
            assert cfg.TRAINER.BIOMEDCOOP.PREC in ["fp16", "fp32", "amp"]

        def build_model(self):
            cfg = self.cfg
            classnames = self.dm.dataset.classnames

            print("=" * 80)
            print("构建混合模型 + CoOp：PMC-CLIP ResNet50 + BiomedCLIP 文本编码器 + CoOp")
            print("=" * 80)

            # 加载 PMC-CLIP 图像编码器
            import os
            directory = "clip/checkpoints"
            from models.coop_pmcclip import ModifiedResNet
            pmc_image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
            pmc_image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'), weights_only=True))

            # 加载 BiomedCLIP
            biomedclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            if cfg.TRAINER.BIOMEDCOOP.PREC == "fp32" or cfg.TRAINER.BIOMEDCOOP.PREC == "amp":
                biomedclip_model.float()

            pmc_image_encoder = pmc_image_encoder.to(self.device).eval()
            biomedclip_model = biomedclip_model.to(self.device).eval()

            self.model = HybridCLIPWithCoOp(cfg, classnames, pmc_image_encoder, biomedclip_model)

            # 冻结文本编码器，只训练图像编码器、CoOp prompts 和投影层
            print("Turning off gradients in the text encoder")
            print("Keeping gradients for: prompt_learner.ctx (CoOp), image_encoder, text_projection, distill_projection")
            
            names_to_update = ["prompt_learner.ctx"]
            for name, param in self.model.named_parameters():
                if name.startswith("image_encoder.") or name.startswith("text_projection.") or name.startswith("distill_projection."):
                    names_to_update.append(name)
                    param.requires_grad_(True)

            for name, param in self.model.named_parameters():
                if name not in names_to_update:
                    param.requires_grad_(False)

            if cfg.MODEL.INIT_WEIGHTS:
                load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

            self.model.to(self.device)
            self.optim = build_optimizer(self.model, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("prompt_learner", self.model, self.optim, self.sched)
            
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
                    logits, loss_ce, contrastive_loss, loss_distill = model(image, label)
                    loss = (self.model.classification_loss_weight * loss_ce + 
                           self.model.contrastive_loss_weight * contrastive_loss +
                           self.model.distillation_loss_weight * loss_distill)
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                logits, loss_ce, contrastive_loss, loss_distill = model(image, label)
                loss = (self.model.classification_loss_weight * loss_ce + 
                       self.model.contrastive_loss_weight * contrastive_loss +
                       self.model.distillation_loss_weight * loss_distill)
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "loss_ce": loss_ce.item(),
                "contrastive_loss": contrastive_loss.item(),
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
                print("Note that load_model() is skipped as no pretrained model is given")
                return

            names = self.get_model_names()
            model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"

            for name in names:
                model_path = osp.join(directory, name, model_file)
                if not osp.exists(model_path):
                    raise FileNotFoundError('Model not found at "{}"'.format(model_path))

                checkpoint = load_checkpoint(model_path)
                state_dict = checkpoint["state_dict"]
                epoch = checkpoint["epoch"]

                if "prompt_learner.token_prefix" in state_dict:
                    del state_dict["prompt_learner.token_prefix"]
                if "prompt_learner.token_suffix" in state_dict:
                    del state_dict["prompt_learner.token_suffix"]

                print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
                self._models[name].load_state_dict(state_dict, strict=False)

