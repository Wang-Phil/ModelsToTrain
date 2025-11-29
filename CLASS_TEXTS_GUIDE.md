# 类别文本描述配置指南

## 概述

在CLIP模型中，类别文本描述对于图像-文本对齐非常重要。更好的文本描述可以提高模型的分类性能。

## 使用方法

### 方法1: 使用文本模板

使用统一的文本模板为所有类别生成描述：

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/my_model \
    --text-template "这是一张{class_name}的医学图像" \
    ...
```

支持的模板变量：
- `{class_name}`: 类别名称

示例：
- 模板：`"这是一张{class_name}的医学图像"`
- 类别：`"正常"` → 文本：`"这是一张正常的医学图像"`
- 类别：`"异常"` → 文本：`"这是一张异常的医学图像"`

### 方法2: 使用JSON配置文件（推荐）

为每个类别自定义详细的文本描述，创建JSON文件：

**示例文件：`class_texts.json`**
```json
{
  "正常": "这是一张正常的医学X光图像，显示正常的骨骼结构和组织",
  "异常": "这是一张异常的医学X光图像，显示可能存在病变或损伤",
  "骨折": "这是一张显示骨折的医学X光图像",
  "肿瘤": "这是一张显示肿瘤的医学图像"
}
```

**使用配置文件：**
```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/my_model \
    --class-texts-file class_texts.json \
    ...
```

### 方法3: 在Python代码中直接指定

```python
class_texts_dict = {
    "正常": "这是一张正常的医学X光图像",
    "异常": "这是一张异常的医学X光图像",
}

dataset = CLIPDataset(
    data_dir="single_label_data",
    class_texts_dict=class_texts_dict
)
```

## 优先级

如果同时提供了多种方式，优先级如下（高到低）：
1. `class_texts_file` - JSON文件
2. `class_texts_dict` - Python字典
3. `text_template` - 文本模板
4. 默认 - 直接使用类别名称

## 配置文件支持

在 `train_clip_config.json` 中可以添加：

```json
{
  "image_encoder_name": "starnet_dual_pyramid_rcf",
  "text_encoder_name": "bert-base-chinese",
  "text_template": "这是一张{class_name}的医学图像",
  "class_texts_file": "class_texts.json",
  ...
}
```

## 最佳实践

### 1. 使用描述性文本

❌ **不好的描述：**
```json
{
  "类别1": "类别1",
  "类别2": "类别2"
}
```

✅ **好的描述：**
```json
{
  "类别1": "这是一张显示正常解剖结构的医学X光图像",
  "类别2": "这是一张显示异常病变的医学X光图像"
}
```

### 2. 包含领域信息

对于医学图像分类，可以包含：
- 图像类型（X光、CT、MRI等）
- 解剖部位
- 正常/异常状态
- 病变类型

示例：
```json
{
  "正常肺部": "这是一张显示正常肺部结构的胸部X光图像，肺野清晰，没有异常阴影",
  "肺炎": "这是一张显示肺炎病变的胸部X光图像，可见肺部炎症阴影",
  "肺结核": "这是一张显示肺结核病变的胸部X光图像，可见结核结节和阴影"
}
```

### 3. 保持一致的语言风格

保持所有描述使用相同的语言风格和格式，这样模型更容易学习到一致的模式。

### 4. 使用中文BERT模型

如果使用中文类别描述，建议使用 `bert-base-chinese` 作为文本编码器：

```bash
python train_clip.py \
    --text-encoder bert-base-chinese \
    --class-texts-file class_texts.json \
    ...
```

## 自动生成类别文本描述文件

你可以使用以下脚本自动生成类别文本描述文件：

```python
import json
from pathlib import Path

data_dir = Path("single_label_data")
classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

# 方法1: 使用模板生成
class_texts = {}
template = "这是一张{class_name}的医学图像"
for cls in classes:
    class_texts[cls] = template.format(class_name=cls)

# 方法2: 手动定制每个类别
# class_texts = {
#     "正常": "这是一张正常的医学X光图像",
#     "异常": "这是一张异常的医学X光图像",
#     ...
# }

with open("class_texts.json", "w", encoding="utf-8") as f:
    json.dump(class_texts, f, ensure_ascii=False, indent=2)

print("类别文本描述文件已生成: class_texts.json")
```

## 验证文本描述

训练时会自动打印每个类别的文本描述，你可以检查是否正确：

```
类别文本描述:
  正常: 这是一张正常的医学X光图像
  异常: 这是一张异常的医学X光图像
  肺炎: 这是一张显示肺炎病变的医学X光图像
```

## 示例

完整的训练命令示例：

```bash
# 使用JSON文件
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/my_model \
    --image-encoder starnet_dual_pyramid_rcf \
    --text-encoder bert-base-chinese \
    --class-texts-file class_texts.json \
    --batch-size 32 \
    --epochs 100 \
    --gpu-id 0

# 使用文本模板
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/my_model \
    --text-template "这是一张{class_name}的医学X光图像" \
    ...
```

## 注意事项

1. **编码问题**：确保JSON文件使用UTF-8编码
2. **类别名称匹配**：JSON文件中的类别名称必须与数据文件夹中的类别名称完全匹配
3. **缺失类别**：如果JSON文件中没有某个类别，将使用文本模板或类别名称本身
4. **文本长度**：建议文本描述在10-100个字符之间，过长可能影响BERT模型的性能

