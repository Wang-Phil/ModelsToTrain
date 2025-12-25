# MetaCLIP 项目介绍与训练指南

## 项目概述

**MetaCLIP** 是 Meta (Facebook) AI Research 开发的一个大规模 CLIP (Contrastive Language-Image Pre-training) 模型训练项目。该项目专注于解决 CLIP 训练中的关键问题，特别是数据构建和训练方法。

### 核心特点

1. **数据驱动的方法**：MetaCLIP 强调高质量数据构建的重要性，提供了完整的数据筛选和构建流程
2. **多语言支持**：MetaCLIP 2 扩展到支持 329 种语言的全球数据
3. **可复现性**：提供完整的数据卡片（Data Card）和训练配置，便于研究和复现
4. **模型规模**：从 ViT-S 到 ViT-bigG 的多种模型规模

## 项目版本

### MetaCLIP 1
- **论文**：[Demystifying CLIP Data](https://arxiv.org/abs/2309.16671) (ICLR 2024 Spotlight)
- **重点**：数据构建方法论，使用 Common Crawl 数据
- **数据规模**：400M 和 2.5B 图像-文本对
- **模型**：ViT-B/32, ViT-B/16, ViT-L/14, ViT-H/14, ViT-bigG/14

### MetaCLIP 2 (Worldwide)
- **论文**：[Meta CLIP 2: A Worldwide Scaling Recipe](https://arxiv.org/abs/2507.22062) (NeurIPS 2025 Spotlight)
- **重点**：扩展到全球多语言数据，解决多语言 CLIP 训练中的挑战
- **数据规模**：29B 图像-文本对，覆盖 329 种语言
- **模型**：ViT-H/14, ViT-bigG/14 以及多个蒸馏模型

## 项目结构

```
MetaCLIP/
├── src/
│   ├── mini_clip/          # 轻量级 CLIP 实现
│   └── training/           # 训练代码
├── configs/                # 训练配置文件
├── metaclip/               # 数据构建工具
├── apps/                   # 应用代码（MoDE, Altogether等）
├── docs/                   # 文档
└── submit.py              # 分布式训练提交脚本
```

## 预训练模型

### MetaCLIP 2 模型

| 模型 | 预训练权重 | Tokenizer | 分辨率 | CVQA-LOCAL ZS Acc. |
|------|-----------|-----------|--------|-------------------|
| ViT-H-14-quickgelu-worldwide | metaclip2_worldwide | facebook/xlm-v-base | 224 | 57.4 |
| ViT-H-14-378-worldwide | metaclip2_worldwide | facebook/xlm-v-base | 378 | 58.2 |
| ViT-bigG-14-worldwide | metaclip2_worldwide | facebook/xlm-v-base | 224 | 60.7 |
| ViT-bigG-14-378-worldwide | metaclip2_worldwide | facebook/xlm-v-base | 378 | 62.0 |

### MetaCLIP 1 模型

| 模型 | 预训练权重 | 数据规模 | 分辨率 | ImageNet ZS Acc. |
|------|-----------|---------|--------|-----------------|
| ViT-B-32-quickgelu | metaclip_400m | 400M | 224 | 65.5 |
| ViT-B-16-quickgelu | metaclip_400m | 400M | 224 | 70.8 |
| ViT-L-14-quickgelu | metaclip_400m | 400M | 224 | 76.2 |
| ViT-H-14-quickgelu | metaclip_2_5b | 2.5B | 224 | 80.5 |
| ViT-bigG-14-quickgelu | metaclip_2_5b | 2.5B | 224 | 82.1 |

## 快速开始

### 1. 环境安装

```bash
# 创建 conda 环境
conda create -n metaclip python=3.10 pytorch torchvision pytorch-cuda=11.7 \
    tqdm ftfy braceexpand regex pandas submitit=1.2.1 \
    -c pytorch-nightly -c nvidia -c conda-forge -c anaconda

conda activate metaclip

# 安装项目
cd MetaCLIP
pip install -e .
```

### 2. 使用预训练模型

#### 方法一：使用 mini_clip（本仓库）

```python
import torch
from PIL import Image
from src.mini_clip.factory import create_model_and_transforms, get_tokenizer

# 加载模型
model, _, preprocess = create_model_and_transforms(
    'ViT-H-14-quickgelu-worldwide@WorldWideCLIP', 
    pretrained='metaclip2_worldwide'
)
tokenize = get_tokenizer("facebook/xlm-v-base")

# 处理图像和文本
image = preprocess(Image.open("image.jpg")).unsqueeze(0)
text = tokenize(["a diagram", "a dog", "a cat"])

# 编码
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 计算相似度
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```

#### 方法二：使用 Hugging Face

```python
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch

# Meta CLIP 2
processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
model = AutoModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

image = Image.open("image.jpg")
inputs = processor(
    text=["a diagram", "a dog", "a cat"], 
    images=image, 
    return_tensors="pt", 
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    text_probs = logits_per_image.softmax(dim=-1)

print("Label probs:", text_probs)
```

## 训练指南

### 训练流程概述

MetaCLIP 的训练分为两个主要阶段：

1. **数据构建阶段**：构建高质量的图像-文本对数据集
2. **模型训练阶段**：使用构建的数据训练 CLIP 模型

### 数据构建（MetaCLIP 2 Worldwide）

MetaCLIP 2 的数据构建过程非常复杂，涉及多语言数据处理：

#### 步骤 1：下载 Wikipedia 语料库

```bash
# 下载所有 329 种语言的 Wikipedia 数据
for lang_code in en zh ja ko ...; do
    bash metaclip/metadata/download_wikipedia.sh $lang_code data/metadata_source/wiki_text
done
```

#### 步骤 2：构建多语言 WordNet

```bash
python metaclip/metadata/build_multilingual_wordnet.py
```

#### 步骤 3：构建 Wikipedia N-grams

```bash
# 并行处理（推荐）
python metaclip/metadata/build_ngram.py submitit

# 顺序处理（测试用）
python metaclip/metadata/build_ngram.py
```

#### 步骤 4：构建 Wikipedia 文章标题

```bash
python metaclip/metadata/build_title.py submitit
python metaclip/metadata/build_title.py  # 合并多个日期范围的数据
```

#### 步骤 5：合并元数据源

```bash
python metaclip/metadata/build_metadata.py
```

#### 步骤 6：数据筛选和构建

```bash
# 使用元数据筛选图像-文本对
python metaclip/metadata/filter_data.py
```

详细的数据构建流程请参考：
- [MetaCLIP 2 数据构建文档](docs/metaclip2.md)
- [MetaCLIP 1 数据构建文档](docs/metaclip1.md)

### 模型训练

#### 方法一：使用配置文件（推荐）

1. **准备配置文件**

查看 `configs/` 目录下的配置文件：
- `metaclip_v1.py` - MetaCLIP 1 配置
- `metaclip_v2.py` - MetaCLIP 2 配置
- `metaclip_v1_2.py` - MetaCLIP 1.2 (Altogether) 配置

2. **使用 submitit 提交训练任务（集群环境）**

```bash
python submit.py config_name \
    --ngpus 64 \
    --nodes 1 \
    --timeout 4320 \
    --partition learnlab
```

参数说明：
- `config_name`: 配置文件名（不含 .py 后缀）
- `--ngpus`: 每个节点的 GPU 数量
- `--nodes`: 节点数量
- `--timeout`: 任务超时时间（分钟）
- `--partition`: SLURM 分区名称

3. **直接运行训练（单机/多机）**

```bash
# 单机多卡
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    src/training/main.py \
    --config configs/metaclip_v2.py

# 多机多卡
python -m torch.distributed.launch \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    src/training/main.py \
    --config configs/metaclip_v2.py
```

#### 方法二：自定义训练脚本

参考 `src/training/train.py` 和 `src/training/main.py` 编写自定义训练脚本。

### 训练配置示例

典型的训练配置包括：

```python
# configs/custom_config.py
from configs.metaclip_v2 import *

# 模型配置
model = 'ViT-B-32-quickgelu-worldwide@WorldWideCLIP'
pretrained = None  # 从头训练

# 数据配置
train_data = 'path/to/train/data'
val_data = 'path/to/val/data'

# 训练超参数
batch_size = 8192  # 全局批次大小
lr = 1e-3
warmup = 10000
epochs = 32

# 优化器配置
optimizer = 'adamw'
weight_decay = 0.2

# 输出配置
output_dir = 'outputs/custom_training'
```

### 训练注意事项

1. **大规模训练**：
   - MetaCLIP 2 使用 29B 图像-文本对，需要大量计算资源
   - 建议使用多节点多 GPU 训练（如 256 x A100）

2. **数据加载**：
   - 使用高效的数据加载器（如 WebDataset）
   - 确保数据 I/O 不会成为瓶颈

3. **混合精度训练**：
   - 默认使用 FP16/BF16 混合精度训练
   - 可以显著减少内存占用和加速训练

4. **检查点保存**：
   - 定期保存检查点（每个 epoch）
   - 支持从检查点恢复训练

5. **验证和评估**：
   - 定期进行零样本评估
   - 使用标准基准数据集（ImageNet, COCO 等）

## 与你的 CLIP 训练代码的对比

### 相似之处

1. **架构**：都使用 CLIP 的对比学习架构
2. **训练方式**：都使用图像-文本对比损失
3. **模型结构**：都支持 ViT 和 ResNet 等视觉编码器

### 主要区别

| 特性 | MetaCLIP | 你的代码 |
|------|----------|---------|
| **数据规模** | 29B 图像-文本对 | 自定义数据集 |
| **多语言支持** | 329 种语言 | 主要支持中英文 |
| **数据构建** | 完整的数据筛选流程 | 使用现有数据集 |
| **训练规模** | 大规模分布式训练 | 单机/小规模训练 |
| **模型规模** | ViT-S 到 ViT-bigG | ResNet, StarNet 等 |
| **交叉验证** | 无（大规模数据） | 支持 5 折交叉验证 |

### 适用场景

**MetaCLIP 适合**：
- 大规模预训练
- 多语言 CLIP 模型开发
- 研究数据构建方法
- 需要最强性能的场景

**你的代码适合**：
- 特定领域微调（如医学图像）
- 小到中等规模数据集
- 需要交叉验证的严谨评估
- 快速原型开发

## 参考资源

1. **论文**：
   - [Meta CLIP 2: A Worldwide Scaling Recipe](https://arxiv.org/abs/2507.22062)
   - [Demystifying CLIP Data](https://arxiv.org/abs/2309.16671)

2. **代码仓库**：
   - GitHub: https://github.com/facebookresearch/MetaCLIP
   - Hugging Face: https://huggingface.co/collections/facebook/meta-clip-687e97787e9155bc480ef446

3. **文档**：
   - [MetaCLIP 2 数据构建指南](docs/metaclip2.md)
   - [MetaCLIP 1 数据构建指南](docs/metaclip1.md)

4. **演示**：
   - [Hugging Face Spaces](https://huggingface.co/spaces/activebus/MetaCLIP)
   - [Colab Notebook](https://colab.research.google.com/drive/1V0Rv1QQJkcolTjiwJuRsqWycROvYjOwg?usp=sharing)

## 总结

MetaCLIP 是一个专注于大规模 CLIP 训练的研究项目，特别强调数据构建的重要性。如果你的目标是：

- **使用预训练模型**：直接使用 Hugging Face 或本仓库提供的预训练模型
- **研究数据构建**：参考 MetaCLIP 的数据构建流程
- **大规模训练**：使用 MetaCLIP 的训练框架
- **特定领域应用**：可以结合 MetaCLIP 的预训练模型和你的微调代码

对于你的医学图像分类任务，建议：
1. 使用 MetaCLIP 的预训练模型作为初始化
2. 在你的数据集上进行微调
3. 使用你的交叉验证框架进行严谨评估



