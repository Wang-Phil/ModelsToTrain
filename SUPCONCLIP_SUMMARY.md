# SupConCLIP 模型总结

## 一、模型架构

SupConCLIP 是一个结合了有监督对比学习（Supervised Contrastive Learning）和 CLIP（Contrastive Language-Image Pre-training）架构的模型，用于医学图像分类任务。

### 1.1 整体架构

模型采用双编码器架构，包含以下核心组件：

```
输入图像 → 图像编码器 → 图像特征 [batch_size, embed_dim]
                                    ↓
                              统一嵌入空间
                                    ↑
输入文本 → 文本编码器 → 文本特征 [batch_size, embed_dim] 或 [num_classes, embed_dim]
```

### 1.2 图像编码器 (ImageEncoder)

**支持的模型类型：**
- **ResNet系列**：ResNet18/34/50/101/152
  - 支持 ImageNet 预训练权重
  - 支持 CLIP 预训练权重（ResNet50/101）
  - 支持 PMC-CLIP 预训练权重（ResNet50）
- **ViT (Vision Transformer)**：使用 `google/vit-base-patch16-224`
- **BiomedCLIP**：使用 `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`

**架构流程：**
1. 使用预训练backbone提取图像特征
2. 通过投影层将特征投影到统一嵌入空间（`embed_dim`）
3. L2归一化：`F.normalize(x, p=2, dim=1)`

**输出：** `[batch_size, embed_dim]` 的归一化特征向量

### 1.3 文本编码器 (TextEncoder)

**支持的模型类型：**
- **PubMedBERT**：`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- **BiomedCLIP文本编码器**：使用 open_clip 加载
- **PMC-CLIP文本编码器**：BiomedBERT + text_projection_layer
- **CLIP文本编码器**：支持 ViT-B/32, RN50 等

**架构流程：**
1. Tokenization：将文本转换为token IDs
2. 通过预训练语言模型编码
3. 提取CLS token或pooler输出
4. 通过投影层投影到统一嵌入空间
5. L2归一化

**输出：** `[batch_size, embed_dim]` 或 `[num_classes, embed_dim]` 的归一化特征向量

### 1.4 CLIPModel 主模型

**核心组件：**
- `image_encoder`: 图像编码器实例
- `text_encoder`: 文本编码器实例
- `temperature`: 可学习的温度参数（用于相似度缩放）

**关键方法：**
- `forward(images, texts)`: 编码图像和文本，返回特征
- `compute_similarity(image_features, text_features)`: 计算余弦相似度并应用温度参数
- `predict(images, class_texts)`: 零样本分类预测

## 二、损失函数设计

SupConCLIP 采用**多任务学习**策略，结合三种损失函数：

```
总损失 = λ₁ × SupCon Loss + λ₂ × CLIP Loss + λ₃ × Classification Loss
```

其中：
- `λ₁ = supcon_loss_weight`（默认1.0）
- `λ₂ = clip_loss_weight`（默认1.0）
- `λ₃ = class_loss_weight`（默认1.0）

### 2.1 损失函数组合策略

三种损失函数的作用：

1. **SupCon Loss**：拉近同类图像的特征表示
   - 目标：使同一类别的图像在特征空间中聚集
   - 作用：增强类内紧密度

2. **CLIP Loss**：对齐图像和文本特征
   - 目标：使匹配的图像-文本对在特征空间中接近
   - 作用：实现跨模态对齐

3. **Classification Loss**：直接优化分类性能
   - 目标：最小化分类错误
   - 作用：提供明确的分类监督信号

## 三、各损失函数的详细计算方式

### 3.1 SupCon Loss（有监督对比学习损失）

**参考论文：** [Supervised Contrastive Learning (2020)](https://arxiv.org/pdf/2004.11362.pdf)

**核心思想：**
对于一个batch中的某张图片，它的正样本不仅包括它自己的增强版本，还包括batch中所有属于同一类别的其他图片。

**输入：**
- `features`: `[batch_size, n_views, embed_dim]` 或 `[batch_size, embed_dim]`
- `labels`: `[batch_size]` 类别标签

**计算步骤：**

1. **特征预处理**
   ```python
   # 如果只有单视图，扩展维度
   if len(features.shape) < 3:
       features = features.unsqueeze(1)  # [batch_size, 1, embed_dim]
   ```

2. **生成正样本掩码（Mask）**
   ```python
   labels = labels.contiguous().view(-1, 1)  # [batch_size, 1]
   mask = torch.eq(labels, labels.T).float()  # [batch_size, batch_size]
   # mask[i, j] = 1 表示样本i和j属于同一类别
   ```

3. **处理多视图特征**
   ```python
   contrast_count = features.shape[1]  # 视图数量
   contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
   # [batch_size * n_views, embed_dim]
   
   # 根据contrast_mode选择anchor
   if contrast_mode == 'all':
       anchor_feature = contrast_feature  # 所有视图作为anchor
   else:
       anchor_feature = features[:, 0]  # 只使用第一个视图
   ```

4. **计算相似度矩阵**
   ```python
   # 计算点积相似度
   anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) / temperature
   
   # 数值稳定性：减去每行最大值
   logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
   logits = anchor_dot_contrast - logits_max.detach()
   ```

5. **构建掩码**
   ```python
   # 扩展mask以匹配多视图
   mask = mask.repeat(anchor_count, contrast_count)
   
   # 移除自身对比（self-contrast）
   logits_mask = torch.scatter(
       torch.ones_like(mask),
       1,
       torch.arange(batch_size * anchor_count).view(-1, 1),
       0
   )
   mask = mask * logits_mask
   ```

6. **计算对数概率**
   ```python
   exp_logits = torch.exp(logits) * logits_mask
   log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
   ```

7. **计算正样本对的平均对数似然**
   ```python
   mask_pos_pairs = mask.sum(1)  # 每个anchor的正样本数量
   mask_pos_pairs = torch.where(mask_pos_pairs > 0, mask_pos_pairs, 
                                 torch.ones_like(mask_pos_pairs))
   
   mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
   ```

8. **最终损失**
   ```python
   loss = - (temperature / base_temperature) * mean_log_prob_pos
   loss = loss.view(anchor_count, batch_size).mean()
   ```

**数学公式：**

对于每个anchor样本 `i`，其损失为：

\[
L_{supcon}^{(i)} = -\frac{\tau}{\tau_{base}} \cdot \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}
\]

其中：
- `P(i)`: 样本i的正样本集合（同类别样本）
- `A(i)`: 样本i的所有对比样本集合（排除自身）
- `z_i`: 样本i的特征向量（已归一化）
- `τ`: 温度参数（temperature）
- `τ_base`: 基础温度参数（base_temperature）

### 3.2 CLIP Loss（图像-文本对比损失）

**输入：**
- `image_features`: `[batch_size, embed_dim]`
- `text_features`: `[batch_size, embed_dim]`（batch内配对）或 `[num_classes, embed_dim]`（类别文本）

**计算步骤：**

1. **特征归一化**
   ```python
   image_features = F.normalize(image_features, dim=1)
   text_features = F.normalize(text_features, dim=1)
   ```

2. **计算相似度矩阵**
   ```python
   logits = image_features @ text_features.T / temperature
   # [batch_size, batch_size] 或 [batch_size, num_classes]
   ```

3. **双向对比损失（batch内配对）**
   ```python
   if text_features.shape[0] == batch_size:
       labels = torch.arange(batch_size, device=device)
       # 图像到文本
       loss_i2t = F.cross_entropy(logits, labels)
       # 文本到图像
       loss_t2i = F.cross_entropy(logits.T, labels)
       loss = (loss_i2t + loss_t2i) / 2
   ```

**数学公式：**

对于batch内的图像-文本配对：

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
- `T_i`: 第i个文本特征
- `sim(·,·)`: 余弦相似度（点积，因为特征已归一化）
- `τ`: 温度参数

### 3.3 Classification Loss（分类损失）

**注意：** 当前实现使用的是**标准交叉熵损失（Cross Entropy Loss）**，而不是 Focal Loss。虽然代码库中在其他损失函数（如 `DistillationLoss`）中实现了 Focal Loss，但在 SupConCLIP 的训练流程中，分类损失使用的是标准的 `F.cross_entropy`。

**输入：**
- `image_features`: `[batch_size, embed_dim]`
- `class_text_features`: `[num_classes, embed_dim]`（所有类别的文本特征）
- `labels`: `[batch_size]` 真实类别标签

**计算步骤：**

1. **计算分类logits**
   ```python
   image_features_norm = F.normalize(image_features, dim=1)
   class_text_features_norm = F.normalize(class_text_features, dim=1)
   class_logits = model.compute_similarity(image_features_norm, class_text_features_norm)
   # [batch_size, num_classes]
   ```

2. **标准交叉熵损失**
   ```python
   class_loss = F.cross_entropy(class_logits, labels)
   ```

**数学公式：**

\[
L_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_{y_i}) / \tau)}{\sum_{c=1}^{C} \exp(\text{sim}(I_i, T_c) / \tau)}
\]

其中：
- `N`: batch size
- `C`: 类别数量
- `y_i`: 样本i的真实类别
- `T_c`: 类别c的文本特征
- `τ`: 温度参数

**关于 Focal Loss：**

虽然当前实现未使用 Focal Loss，但代码库中已有 Focal Loss 的实现（在 `DistillationLoss` 类中）。如果需要使用 Focal Loss 来处理类别不平衡问题，可以修改训练代码，将分类损失替换为：

```python
# Focal Loss 实现（参考 DistillationLoss.classification_loss）
probs = F.softmax(class_logits, dim=1)
p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
focal_weight = (1 - p_t) ** gamma
class_loss = -alpha * focal_weight * torch.log(p_t + 1e-8)
class_loss = class_loss.mean()
```

其中：
- `alpha`: 平衡因子（默认0.25）
- `gamma`: 聚焦参数（默认2.0）

## 四、训练流程

### 4.1 前向传播

```python
# 1. 编码图像和文本
image_features, text_features = model(images, texts=texts)
# image_features: [batch_size, embed_dim]
# text_features: [batch_size, embed_dim]

# 2. 计算各损失
# SupCon Loss
image_features_for_supcon = image_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
supcon_loss = supcon_criterion(image_features_for_supcon, labels=labels)

# CLIP Loss
clip_loss = clip_criterion(image_features, text_features)

# Classification Loss
class_text_features = model.text_encoder(texts=class_texts)  # [num_classes, embed_dim]
class_logits = model.compute_similarity(image_features, class_text_features)
class_loss = F.cross_entropy(class_logits, labels)

# 3. 组合损失
total_loss = (supcon_loss_weight * supcon_loss + 
              clip_loss_weight * clip_loss + 
              class_loss_weight * class_loss)
```

### 4.2 反向传播

```python
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

## 五、关键参数

### 5.1 模型参数
- `embed_dim`: 统一嵌入空间维度（默认512）
- `temperature`: 温度参数（默认0.07，可学习）

### 5.2 损失函数参数
- `supcon_temperature`: SupCon损失的温度参数（默认0.07）
- `supcon_base_temperature`: SupCon损失的基础温度（默认0.07）
- `supcon_contrast_mode`: 对比模式（'all' 或 'one'）
- `clip_temperature`: CLIP损失的温度参数（默认0.07）

### 5.3 损失权重
- `supcon_loss_weight`: SupCon损失权重（默认1.0）
- `clip_loss_weight`: CLIP损失权重（默认1.0）
- `class_loss_weight`: 分类损失权重（默认1.0）

## 六、优势与特点

1. **多任务学习**：同时优化类内紧密度、跨模态对齐和分类性能
2. **有监督对比学习**：利用标签信息，使同类样本聚集
3. **跨模态对齐**：通过CLIP损失实现图像-文本语义对齐
4. **灵活架构**：支持多种预训练编码器组合
5. **零样本能力**：通过文本描述实现零样本分类

## 七、应用场景

- 医学图像分类（需要文本描述）
- 少样本学习
- 零样本分类
- 跨模态检索

## 八、代码位置

- **模型定义**：`models/clip.py`
- **损失函数**：`train_clip.py` (SupConLoss, CLIPLoss)
- **训练脚本**：`train_clip.py` (train_epoch函数)

