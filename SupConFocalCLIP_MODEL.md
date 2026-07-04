# SupConFocalCLIP 模型介绍

## 模型名称

**SupConFocalCLIP**

- **SupCon**: Supervised Contrastive Learning（有监督对比学习）
- **Focal**: Focal Loss（焦点损失，用于处理类别不平衡）
- **CLIP**: Contrastive Language-Image Pre-training（对比语言-图像预训练）

## 模型概述

SupConFocalCLIP 是一个面向医学图像分类的多任务学习模型，结合了：
1. **有监督对比学习**（SupCon Loss）：增强类内特征紧密度
2. **跨模态对齐**（CLIP Loss）：实现图像-文本语义对齐
3. **类别不平衡处理**（Focal Loss）：提升少数类别的分类性能

该模型在保持CLIP架构的基础上，通过多损失函数组合，实现了在医学图像分类任务上的优异性能。

## 模型架构

### 整体架构

SupConFocalCLIP 采用**双编码器架构**，包含图像编码器、文本编码器和统一嵌入空间：

```
                    ┌─────────────────┐
                    │   输入图像       │
                    │ [B, 3, H, W]    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  图像编码器      │
                    │ (ImageEncoder)  │
                    │  - Backbone     │
                    │  - Projection   │
                    │  - L2 Norm      │
                    └────────┬────────┘
                             │
                             │ 图像特征
                             │ [B, embed_dim]
                             │
                    ┌────────▼──────────────────┐
                    │                           │
                    │    统一嵌入空间            │
                    │   (Unified Embedding)     │
                    │      embed_dim = 512      │
                    │                           │
                    ┌────────┬──────────────────┘
                             │
                             │ 文本特征
                             │ [B, embed_dim] 或
                             │ [num_classes, embed_dim]
                    ┌────────▼────────┐
                    │  文本编码器      │
                    │ (TextEncoder)   │
                    │  - Tokenizer    │
                    │  - Backbone     │
                    │  - Projection   │
                    │  - L2 Norm      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   输入文本       │
                    │  或类别描述      │
                    └─────────────────┘
```

### 1. 图像编码器 (ImageEncoder)

**功能**：将输入图像编码为统一嵌入空间的特征向量

**架构组成**：
- **Backbone**：预训练的视觉模型
  - ResNet系列：ResNet18/34/50/101/152
    - ImageNet预训练权重
    - CLIP预训练权重（ResNet50/101）
    - PMC-CLIP预训练权重（ResNet50）
  - Vision Transformer (ViT)
  - BiomedCLIP图像编码器
- **投影层**：`nn.Linear(feature_dim, embed_dim)`
  - 将backbone输出投影到统一维度（默认512维）
- **归一化层**：L2归一化
  - `F.normalize(x, p=2, dim=1)`

**前向传播流程**：
```python
# 1. 通过backbone提取特征
x = backbone(images)  # [batch_size, feature_dim]

# 2. 投影到统一嵌入空间
x = projection(x)  # [batch_size, embed_dim]

# 3. L2归一化
x = F.normalize(x, p=2, dim=1)  # [batch_size, embed_dim]
```

**输出**：`[batch_size, embed_dim]` 的归一化特征向量

### 2. 文本编码器 (TextEncoder)

**功能**：将输入文本（类别描述或配对文本）编码为统一嵌入空间的特征向量

**架构组成**：
- **Tokenizer**：文本分词器
  - CLIP Tokenizer
  - BERT/PubMedBERT Tokenizer
  - BiomedCLIP Tokenizer
- **Backbone**：预训练的语言模型
  - **CLIP文本编码器**：ViT-B/32, RN50等
  - **PubMedBERT**：医学领域BERT模型
  - **BiomedCLIP文本编码器**：基于PubMedBERT的医学CLIP
  - **PMC-CLIP文本编码器**：BiomedBERT + text_projection_layer
- **投影层**：`nn.Linear(hidden_dim, embed_dim)`
  - 将语言模型输出投影到统一维度
- **归一化层**：L2归一化

**前向传播流程**：
```python
# 1. Tokenization
token_ids = tokenizer(texts)  # [batch_size, seq_len]

# 2. 通过语言模型编码
outputs = backbone(token_ids)  # [batch_size, seq_len, hidden_dim]

# 3. 提取CLS token或pooler输出
x = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_dim]
# 或
x = outputs.pooler_output  # [batch_size, hidden_dim]

# 4. 投影到统一嵌入空间
x = projection(x)  # [batch_size, embed_dim]

# 5. L2归一化
x = F.normalize(x, p=2, dim=1)  # [batch_size, embed_dim]
```

**输出**：
- Batch内配对：`[batch_size, embed_dim]`
- 类别文本：`[num_classes, embed_dim]`

### 3. CLIPModel 主模型

**核心组件**：
- `image_encoder`：图像编码器实例
- `text_encoder`：文本编码器实例
- `temperature`：可学习的温度参数（`nn.Parameter`）
  - 默认值：0.07
  - 用于相似度缩放：`similarity = similarity / temperature`

**关键方法**：

1. **`forward(images, texts)`**：前向传播
   ```python
   image_features = image_encoder(images)  # [B, embed_dim]
   text_features = text_encoder(texts)     # [B, embed_dim] 或 [C, embed_dim]
   return image_features, text_features
   ```

2. **`compute_similarity(image_features, text_features)`**：计算相似度
   ```python
   similarity = image_features @ text_features.T  # [B, B] 或 [B, C]
   similarity = similarity / self.temperature     # 应用温度参数
   return similarity
   ```

3. **`predict(images, class_texts)`**：零样本分类预测
   ```python
   image_features = image_encoder(images)
   class_text_features = text_encoder(class_texts)  # [C, embed_dim]
   similarity = compute_similarity(image_features, class_text_features)
   probabilities = F.softmax(similarity, dim=1)
   predictions = torch.argmax(similarity, dim=1)
   return predictions, probabilities
   ```

## 损失函数设计

SupConFocalCLIP 采用**多任务学习**策略，通过组合三种损失函数实现优化：

### 总损失函数

```
L_total = λ₁ × L_supcon + λ₂ × L_clip + λ₃ × L_focal
```

其中：
- `L_supcon`：有监督对比学习损失（SupCon Loss）
- `L_clip`：跨模态对齐损失（CLIP Loss）
- `L_focal`：分类损失（Focal Loss）
- `λ₁, λ₂, λ₃`：损失权重（默认均为1.0）

### 1. SupCon Loss（有监督对比学习损失）

**目的**：增强类内特征紧密度，使同类样本在特征空间中聚集

**数学公式**：

对于每个anchor样本 `i`，其损失为：

\[
L_{supcon}^{(i)} = -\frac{\tau}{\tau_{base}} \cdot \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}
\]

其中：
- `P(i)`: 样本i的正样本集合（batch中所有同类别样本）
- `A(i)`: 样本i的所有对比样本集合（排除自身）
- `z_i`: 样本i的特征向量（L2归一化后）
- `τ`: 温度参数（默认0.07）
- `τ_base`: 基础温度参数（默认0.07）

**特点**：
- 利用标签信息，使同类样本聚集
- 比自监督对比学习（SimCLR）更有效
- 适合类别不平衡场景

### 2. CLIP Loss（跨模态对齐损失）

**目的**：实现图像-文本语义对齐，使匹配的图像-文本对在特征空间中接近

**数学公式**：

双向对比损失：

\[
L_{clip} = \frac{1}{2} \left[ L_{i2t} + L_{t2i} \right]
\]

其中：

\[
L_{i2t} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}
\]

\[
L_{t2i} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(T_i, I_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(T_i, I_j) / \tau)}
\]

其中：
- `N`: batch size
- `I_i`: 第i个图像特征
- `T_i`: 第i个文本特征（batch内配对）
- `sim(·,·)`: 余弦相似度（点积，因为特征已归一化）
- `τ`: 温度参数（可学习）

**特点**：
- 双向对比学习，提高对齐效果
- 支持零样本分类
- 增强跨模态理解能力

### 3. Focal Loss（分类损失）

**目的**：处理类别不平衡问题，关注难分类样本

**数学公式**：

\[
L_{focal} = -\alpha (1 - p_t)^{\gamma} \log(p_t)
\]

其中：
- `p_t`: 模型对真实类别的预测概率
- `α`: 平衡因子（默认0.25），用于平衡正负样本
- `γ`: 聚焦参数（默认2.0），用于降低易分类样本的权重

**计算流程**：
```python
# 1. 计算分类logits
class_logits = image_features @ class_text_features.T / temperature
# [batch_size, num_classes]

# 2. 计算Focal Loss
probs = F.softmax(class_logits, dim=1)
p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [batch_size]
focal_weight = (1 - p_t) ** gamma
loss = -alpha * focal_weight * torch.log(p_t + 1e-8)
loss = loss.mean()
```

**特点**：
- 自动关注难分类样本
- 有效处理类别不平衡
- 降低易分类样本的权重

## 训练策略

### 训练流程

```python
# 1. 前向传播
image_features, text_features = model(images, texts=texts)

# 2. 计算各损失
# SupCon Loss
image_features_for_supcon = image_features.unsqueeze(1)  # [B, 1, embed_dim]
supcon_loss = supcon_criterion(image_features_for_supcon, labels)

# CLIP Loss
clip_loss = clip_criterion(image_features, text_features)

# Focal Loss（分类损失）
class_text_features = model.text_encoder(texts=class_texts)  # [C, embed_dim]
class_logits = model.compute_similarity(image_features, class_text_features)
focal_loss = focal_loss(class_logits, labels, alpha=0.25, gamma=2.0)

# 3. 组合损失
total_loss = (supcon_loss_weight * supcon_loss + 
              clip_loss_weight * clip_loss + 
              class_loss_weight * focal_loss)

# 4. 反向传播
total_loss.backward()
optimizer.step()
```

### 关键配置

- **损失权重**：
  - `supcon_loss_weight = 1.0`
  - `clip_loss_weight = 1.0`
  - `class_loss_weight = 1.0`
- **Focal Loss参数**：
  - `focal_alpha = 0.25`
  - `focal_gamma = 2.0`
- **SupCon参数**：
  - `supcon_temperature = 0.07`
  - `supcon_base_temperature = 0.07`
- **文本编码器**：通常冻结（`freeze_text_encoder = True`）

## 模型特点

### 1. 多任务学习

通过组合三种损失函数，同时优化：
- 类内紧密度（SupCon Loss）
- 跨模态对齐（CLIP Loss）
- 分类性能（Focal Loss）

### 2. 类别不平衡处理

- 使用Focal Loss自动关注难分类样本
- 支持加权采样（处理数据不平衡）
- 在少数类别上表现更好

### 3. 跨模态理解

- 支持图像-文本对齐
- 支持零样本分类
- 可扩展性强（易于添加新的类别描述）

### 4. 灵活性

- 支持多种图像编码器（ResNet, ViT, BiomedCLIP等）
- 支持多种文本编码器（CLIP, PubMedBERT, BiomedCLIP等）
- 可配置的损失权重
- 可选择冻结编码器

## 实验性能

根据消融实验结果，SupConFocalCLIP（SupCon + CLIP + Focal Loss）的表现：

- **mAP**: 81.38% ± 2.19%
- **准确率**: 84.11% ± 0.94%
- **Precision**: 81.38% ± 2.19%
- **Recall**: 64.83% ± 4.47%
- **F1 Score**: 68.31% ± 4.95%

**对比基线**：
- 仅使用SupCon Loss：mAP 10.64%（几乎无法学习）
- 仅使用CLIP Loss（LSAL）：mAP 80.01%
- SupCon + CLIP（无Focal Loss）：mAP 82.10%（最高mAP）
- **SupCon + CLIP + Focal Loss**：mAP 81.38%（准确率最高）

## 应用场景

1. **医学图像分类**
   - 需要文本描述的医学图像分类任务
   - 类别不平衡的医学数据集

2. **少样本学习**
   - 利用文本描述进行少样本学习
   - 零样本分类

3. **跨模态检索**
   - 图像-文本检索
   - 语义搜索

## 参考文献

- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- SupCon: [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
- Focal Loss: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

## 代码位置

- **模型定义**：`models/clip.py`
- **损失函数**：`train_clip.py` (SupConLoss, CLIPLoss, focal_loss函数)
- **训练脚本**：`train_clip.py`
- **配置示例**：`train_clip_config.json`

## 总结

SupConFocalCLIP 是一个面向医学图像分类的创新模型，通过结合有监督对比学习、跨模态对齐和焦点损失，在保持CLIP架构优势的基础上，实现了对类别不平衡，数据量较少 数据的有效处理，为医学图像分类任务提供了一个强大的解决方案。

