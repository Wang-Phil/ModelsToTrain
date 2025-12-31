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
from collections import OrderedDict
import math

# dassl 相关导入（可选，仅用于 TrainerX 类）
try:
    from dassl.engine import TRAINER_REGISTRY, TrainerX
    from dassl.utils import load_pretrained_weights, load_checkpoint
    from dassl.optim import build_optimizer, build_lr_scheduler
    from dassl.metrics import compute_accuracy
    DASSL_AVAILABLE = True
except ImportError:
    DASSL_AVAILABLE = False
    # 定义占位符以避免在 TrainerX 类中使用时出错
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

# 设置环境变量，优先使用本地缓存的模型
# 配置 Hugging Face 中国镜像（如果未设置）
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

# Function to download a file
def download_file(url, filepath):
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


class HybridCLIP(nn.Module):
    """
    混合模型：PMC-CLIP 的 ResNet50 图像编码器 + BiomedCLIP 的文本编码器
    
    维度对齐策略：
    - 图像编码器输出：768维 -> 通过 image_projection 投影到 512维
    - 文本编码器输出：512维（无需投影）
    - Teacher 图像编码器输出：512维
    
    这样图像、文本、teacher都在512维，统一对齐，只需训练一个投影层。
    训练图像编码器和投影层，使用对比损失 + 分类损失 + 蒸馏损失
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
        self.logit_scale = biomedclip_model.logit_scale  # 使用 BiomedCLIP 的 logit_scale
        self.text_embed_dim = 512  # BiomedCLIP 文本特征维度
        
        # ========== 蒸馏：BiomedCLIP 图像编码器作为 teacher ==========
        # 保存 BiomedCLIP 的图像编码器（teacher），用于知识蒸馏
        self.teacher_image_encoder = biomedclip_model.visual  # BiomedCLIP 的图像编码器（teacher）
        # 冻结 teacher 模型
        for param in self.teacher_image_encoder.parameters():
            param.requires_grad = False
        self.teacher_image_encoder.eval()  # 设置为评估模式
        self.teacher_image_embed_dim = 512  # BiomedCLIP 图像特征维度
        
        # ========== 图像投影层：将图像特征从 768 维投影到 512 维 ==========
        # 统一维度策略：将图像特征投影到文本/teacher维度（512维）
        # 这样图像、文本、teacher都在512维，无需多个投影层
        # Image (768->512) vs Text (512) [维度匹配，无需 Text 升维]
        # Image (768->512) vs Teacher (512) [维度匹配，直接蒸馏]
        self.image_projection = nn.Linear(self.image_embed_dim, self.text_embed_dim)
        # 使用 Xavier 初始化
        nn.init.xavier_uniform_(self.image_projection.weight)
        nn.init.zeros_(self.image_projection.bias)

        # 获取 BiomedCLIP 的 tokenizer
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        # 存储类别名称（用于生成文本提示）
        self.classnames = [name.replace("_", " ") for name in classnames]

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
        print(f"预编码 {len(self.class_prompts)} 个类别的文本特征（使用 BiomedCLIP）...")
        with torch.no_grad():
            # 安全地获取设备
            try:
                device = next(biomedclip_model.parameters()).device
            except StopIteration:
                device = biomedclip_model.logit_scale.device

            class_text_features_list = []
            for i, prompt in enumerate(self.class_prompts):
                if i % 3 == 0:  # 每3个类别打印一次进度
                    print(f"  编码进度: {i+1}/{len(self.class_prompts)}")
                tokenized = self.tokenizer(prompt)
                if isinstance(tokenized, torch.Tensor):
                    tokenized = tokenized.to(device)
                elif isinstance(tokenized, dict):
                    tokenized = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
                # 使用 BiomedCLIP 的 encode_text 方法
                text_feat = biomedclip_model.encode_text(tokenized, normalize=True)  # [1, 512]
                # 文本特征已经是512维，无需投影，直接保存
                class_text_features_list.append(text_feat)
            print(f"✓ 完成文本特征预编码")
            # 保存文本特征（512维），与图像投影后的维度匹配
            self.register_buffer('class_text_features_raw', torch.cat(class_text_features_list, dim=0))  # [n_cls, 512]

        # 损失函数权重配置
        self.classification_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'CLASSIFICATION_LOSS_WEIGHT', 0.5)
        self.contrastive_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'CONTRASTIVE_LOSS_WEIGHT', 0.5)
        self.distillation_loss_weight = getattr(cfg.TRAINER.BIOMEDCOOP, 'DISTILLATION_LOSS_WEIGHT', 0.0)

        print(f"损失权重: 分类损失={self.classification_loss_weight}, 对比损失={self.contrastive_loss_weight}, 蒸馏损失={self.distillation_loss_weight}")

    def forward(self, image, label=None):
        """
        前向传播：使用对比损失 + 分类损失 + 蒸馏损失

        Args:
            image: 图像tensor [batch_size, 3, H, W]
            label: 标签 [batch_size]（训练时需要，用于计算损失）

        Returns:
            训练模式: (logits, loss_ce, contrastive_loss, loss_distill)
            评估模式: logits
        """
        # 安全地获取设备
        try:
            device = image.device
        except:
            try:
                device = next(self.image_encoder.parameters()).device
            except StopIteration:
                device = self.logit_scale.device

        logit_scale = self.logit_scale.exp()

        # 获取图像特征（使用 PMC-CLIP 的 ResNet50）
        image_features_raw = self.image_encoder(image.type(self.dtype))
        if isinstance(image_features_raw, dict):
            image_features_raw = image_features_raw['image_features']
        image_features_raw = image_features_raw / image_features_raw.norm(dim=-1, keepdim=True)  # [batch_size, 768]
        
        # 将图像特征投影到512维（与文本和teacher维度匹配）
        image_features = self.image_projection(image_features_raw)  # [batch_size, 512]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.training and label is not None:
            # ========== 训练模式：使用分类损失 + 对比损失 ==========
            batch_size = image_features.shape[0]

            # 文本特征已经是512维，无需投影
            class_text_features = self.class_text_features_raw  # [n_cls, 512]
            class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)

            # 1. 计算分类 logits（使用所有类别的预编码文本特征）
            logits = logit_scale * image_features @ class_text_features.t()  # [batch_size, n_cls]

            # 2. 分类损失：根据权重决定是否计算
            if self.classification_loss_weight > 0:
                loss_ce = F.cross_entropy(logits, label)
            else:
                # 如果权重为0，不计算损失（使用0作为占位符）
                loss_ce = torch.tensor(0.0, device=device)

            # 3. 对比损失：根据权重决定是否计算
            if self.contrastive_loss_weight > 0:
                # 为每个样本选择对应的类别文本特征
                batch_text_features = class_text_features[label]  # [batch_size, 512]
                batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)

                # 计算相似度矩阵（batch 内的图像-文本配对）
                logits_per_image = logit_scale * image_features @ batch_text_features.t()
                logits_per_text = logit_scale * batch_text_features @ image_features.t()

                # 创建对比学习的标签（对角线匹配：batch内每个图像对应一个文本）
                contrastive_labels = torch.arange(batch_size, device=image_features.device)

                # 双向对比损失
                contrastive_loss = (
                    F.cross_entropy(logits_per_image, contrastive_labels) +
                    F.cross_entropy(logits_per_text, contrastive_labels)
                ) / 2
            else:
                # 如果权重为0，不计算损失（使用0作为占位符）
                contrastive_loss = torch.tensor(0.0, device=device)

            # 4. 蒸馏损失：将 BiomedCLIP 图像编码器的知识蒸馏到 PMC-CLIP ResNet50
            if self.distillation_loss_weight > 0:
                # Student 特征：图像特征已经投影到512维
                student_features = image_features  # [batch_size, 512]
                
                # 获取 teacher 特征（BiomedCLIP 图像编码器的输出，512维）
                with torch.no_grad():
                    # 获取 teacher 模型的 dtype
                    teacher_dtype = next(self.teacher_image_encoder.parameters()).dtype
                    teacher_features = self.teacher_image_encoder(image.type(teacher_dtype))
                    if isinstance(teacher_features, dict):
                        teacher_features = teacher_features.get('image_features', teacher_features)
                    teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)  # [batch_size, 512]
                
                # Student 和 Teacher 都是512维，直接比较
                # 计算蒸馏损失（MSE 损失）
                loss_distill = F.mse_loss(student_features, teacher_features)
            else:
                # 如果权重为0，不计算损失（使用0作为占位符）
                loss_distill = torch.tensor(0.0, device=device)

            # 返回 logits 和损失（使用 loss_distill 替代 loss_sccm）
            return logits, loss_ce, contrastive_loss, loss_distill
        else:
            # ========== 评估模式：返回所有类别的 logits ==========
            # 文本特征已经是512维，无需投影
            class_text_features = self.class_text_features_raw  # [n_cls, 512]
            class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ class_text_features.t()
            return logits


# TrainerX 类只在 dassl 可用时定义（用于 dassl 框架）
if DASSL_AVAILABLE:
    @TRAINER_REGISTRY.register()
    class Hybrid_PMCResNet50_BiomedCLIP(TrainerX):
        def check_cfg(self, cfg):
            assert cfg.TRAINER.BIOMEDCOOP.PREC in ["fp16", "fp32", "amp"]

        def build_model(self):
            cfg = self.cfg
            classnames = self.dm.dataset.classnames

            print("=" * 80)
            print("构建混合模型：PMC-CLIP ResNet50 + BiomedCLIP 文本编码器")
            print("=" * 80)

            # 1. 检查并下载 PMC-CLIP 模型文件
            print("检查 PMC-CLIP 模型文件...")
            for filename, url in files.items():
                filepath = os.path.join(directory, filename)
                if not os.path.exists(filepath):
                    print(f"{filename} 未找到，正在下载...")
                    download_file(url, filepath)
                else:
                    print(f"✓ {filename} 已存在")

            # 2. 加载 PMC-CLIP 的 ResNet50 图像编码器
            print("\n加载 PMC-CLIP ResNet50 图像编码器...")
            pmc_image_encoder = ModifiedResNet(layers=[3,4,6,3], output_dim=768, heads=8, image_size=224, width=64)
            pmc_image_encoder.load_state_dict(torch.load(os.path.join(directory,'image_encoder(resnet50).pth'), weights_only=True))
            print("✓ PMC-CLIP ResNet50 图像编码器加载完成")

            # 3. 加载 BiomedCLIP 模型
            print("\n加载 BiomedCLIP 文本编码器...")
            print("注意：如果网络连接有问题，将自动使用本地缓存的模型")
            try:
                biomedclip_model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                print("✓ BiomedCLIP 模型加载完成")
            except Exception as e:
                print(f"✗ 加载 BiomedCLIP 失败: {e}")
                print("提示：如果网络连接有问题，请确保模型已下载到本地缓存")
                print("缓存路径: ~/.cache/huggingface/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
                raise

            if cfg.TRAINER.BIOMEDCOOP.PREC == "fp32" or cfg.TRAINER.BIOMEDCOOP.PREC == "amp":
                biomedclip_model.float()

            # 4. 构建混合模型
            print("\n构建混合 CLIP 模型...")
            self.model = HybridCLIP(cfg, classnames, pmc_image_encoder, biomedclip_model.eval())

            # 5. 设置训练参数：训练图像编码器和投影层
            print("\n设置训练参数：")
            print("✓ 冻结文本编码器（BiomedCLIP）")
            print("✓ 训练图像编码器（PMC-CLIP ResNet50）")
            print("✓ 训练投影层（image_projection: 768->512）")

            # 设置参数的可训练性
            for name, param in self.model.named_parameters():
                # 只要不是 teacher 或原始 biomedclip，都应该训练
                if "image_encoder" in name or "projection" in name:
                    param.requires_grad_(True)
                elif "teacher" in name or "biomedclip" in name:
                    param.requires_grad_(False)
                else:
                    # 处理 logit_scale 等其他参数，通常建议微调 logit_scale
                    # 但这里 logit_scale 来自 biomedclip，为了保持一致性，先冻结
                    # 如果需要微调，可以改为 True
                    param.requires_grad_(False)

            # 统计参数
            enabled = set()
            total_params = 0
            image_encoder_params = 0
            projection_params = 0

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
                    param_count = param.numel()
                    total_params += param_count
                    if 'image_encoder' in name:
                        image_encoder_params += param_count
                    elif 'projection' in name:
                        projection_params += param_count

            print("参数统计:")
            print(f"  总参数: {total_params:,} 参数 ({len(enabled)} 个参数组)")
            print(f"  图像编码器: {image_encoder_params:,} 参数")
            print(f"  投影层: {projection_params:,} 参数")
            print(f"  文本编码器: 冻结（BiomedCLIP）")

            # 加载预训练权重（如果指定）
            if cfg.MODEL.INIT_WEIGHTS:
                load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

            self.model.to(self.device)

            # 优化器：只优化可训练的参数
            self.optim = build_optimizer(self.model, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("model", self.model, self.optim, self.sched)

            self.scaler = GradScaler() if cfg.TRAINER.BIOMEDCOOP.PREC == "amp" else None

            # 多GPU支持
            device_count = torch.cuda.device_count()
            if device_count > 1:
                print(f"检测到 {device_count} 个 GPU，使用所有 GPU 进行训练!")
                self.model = nn.DataParallel(self.model)

        def forward_backward(self, batch):
            image, label = self.parse_batch_train(batch)

            model = self.model
            optim = self.optim
            scaler = self.scaler

            # 如果使用了 DataParallel，需要通过 module 访问自定义属性
            model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

            prec = self.cfg.TRAINER.BIOMEDCOOP.PREC
            if prec == "amp":
                with autocast():
                    logits, loss_ce, contrastive_loss, loss_distill = model(image, label)
                    # 总损失 = 分类损失 + 对比损失 + 蒸馏损失
                    loss = (model_ref.classification_loss_weight * loss_ce + 
                           model_ref.contrastive_loss_weight * contrastive_loss +
                           model_ref.distillation_loss_weight * loss_distill)
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                logits, loss_ce, contrastive_loss, loss_distill = model(image, label)
                # 总损失 = 分类损失 + 对比损失 + 蒸馏损失
                loss = (model_ref.classification_loss_weight * loss_ce + 
                       model_ref.contrastive_loss_weight * contrastive_loss +
                       model_ref.distillation_loss_weight * loss_distill)
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "loss_ce": loss_ce.item(),  # 分类损失
                "contrastive_loss": contrastive_loss.item(),  # 对比损失
                "loss_distill": loss_distill.item(),  # 蒸馏损失
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

                print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
                # set strict=False
                self._models[name].load_state_dict(state_dict, strict=False)
