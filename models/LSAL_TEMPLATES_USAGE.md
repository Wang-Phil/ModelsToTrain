# LSAL 使用类别模板描述指南

## 概述

`build_llm_semantics.py` 现在支持使用详细的类别描述模板（如 `hip_prosthesis_prompt_templates.py`）来生成更准确的LLM语义矩阵。每个类别可以使用多个详细的医学描述，而不是简单的模板。

## 使用方法

### 方法1：使用模板文件（推荐）

如果你有 `hip_prosthesis_prompt_templates.py` 这样的模板文件，可以直接指定：

```bash
# 从JSON文件加载类别名称，并使用模板文件
python models/build_llm_semantics.py \
    --classnames_file class_texts_hip_prosthesis.json \
    --templates_file hip_prosthesis_prompt_templates.py \
    --output_dir ./semantics \
    --tau 0.1 \
    --device cuda

# 或者从命令行指定类别名称
python models/build_llm_semantics.py \
    --classnames "Acetabular Loosening" "Dislocation" "Fracture" "Good Place" \
    --templates_file hip_prosthesis_prompt_templates.py \
    --output_dir ./semantics \
    --tau 0.1
```

### 方法2：自动检测模板文件

如果模板文件在项目根目录下且名为 `hip_prosthesis_prompt_templates.py`，脚本会自动检测并使用：

```bash
# 脚本会自动查找 ./hip_prosthesis_prompt_templates.py
python models/build_llm_semantics.py \
    --classnames_file class_texts_hip_prosthesis.json \
    --output_dir ./semantics \
    --tau 0.1
```

### 方法3：不使用模板（使用默认模板）

如果不指定模板文件，将使用默认的简单模板：

```bash
python models/build_llm_semantics.py \
    --classnames "Pneumonia" "Fracture" "Edema" \
    --output_dir ./semantics \
    --tau 0.1
```

## 模板文件格式

模板文件应该是一个Python文件，包含一个名为 `HIP_PROSTHESIS_TEMPLATES` 的字典（或其他名称，但需要修改代码）：

```python
HIP_PROSTHESIS_TEMPLATES = {
    "Acetabular Loosening": [
        "X-ray showing radiolucent lines around the acetabular cup indicating loosening.",
        "The radiograph reveals gaps between the acetabular component and the surrounding bone.",
        # ... 更多描述
    ],
    "Dislocation": [
        "X-ray showing the femoral head displaced from the acetabular socket.",
        # ... 更多描述
    ],
    # ... 更多类别
}
```

## 类别名称匹配

脚本会自动匹配类别名称，支持：
- 大小写不敏感匹配
- 空格和下划线互换（"Acetabular Loosening" 匹配 "Acetabular_Loosening"）

如果某个类别在模板文件中找不到匹配，将使用默认模板。

## 输出示例

使用模板文件时，输出会显示每个类别使用的模板数量：

```
Loading class templates from: hip_prosthesis_prompt_templates.py
✓ Loaded templates for 9 classes
  Total templates: 522
    - Acetabular Loosening: 58 templates
    - Dislocation: 51 templates
    - Fracture: 51 templates
    - Good Place: 51 templates
    - Infection: 51 templates
    - Native Hip: 51 templates
    - Spacer: 51 templates
    - Stem Loosening: 51 templates
    - Wear: 51 templates

Computing LLM Semantic Centers...
  Number of classes: 9
  Class 1/9: Acetabular Loosening - Using 58 custom templates
  Class 2/9: Dislocation - Using 51 custom templates
  ...
```

## 优势

使用详细的类别描述模板相比默认模板的优势：

1. **更丰富的语义信息**：每个类别有50+个详细描述，覆盖不同的表达方式
2. **更准确的语义中心**：多个描述的平均值更能代表该类别的真实语义
3. **更好的类别区分度**：详细的医学描述能更好地捕捉类别间的细微差别
4. **医学专业性**：使用专业的医学术语和描述，更适合医学图像分类

## 注意事项

1. **类别名称一致性**：确保模板文件中的类别名称与数据集中的类别名称一致（或至少可以匹配）
2. **模板数量**：每个类别的模板数量可以不同，但建议至少10个以上以获得稳定的语义中心
3. **模板质量**：模板应该是高质量的医学描述，而不是简单的重复

## 完整示例

```bash
# 1. 准备类别名称文件（class_texts_hip_prosthesis.json）
{
  "Acetabular Loosening": "...",
  "Dislocation": "...",
  "Fracture": "...",
  "Good Place": "...",
  "Infection": "...",
  "Native Hip": "...",
  "Spacer": "...",
  "Stem Loosening": "...",
  "Wear": "..."
}

# 2. 运行脚本生成语义矩阵
python models/build_llm_semantics.py \
    --classnames_file class_texts_hip_prosthesis.json \
    --templates_file hip_prosthesis_prompt_templates.py \
    --output_dir ./semantics_hip_prosthesis \
    --tau 0.1 \
    --device cuda

# 3. 输出文件
# ./semantics_hip_prosthesis/
#   ├── class_centers.pt
#   ├── soft_labels_matrix.pt
#   ├── classnames.json
#   └── config.json
```

## 故障排除

1. **找不到模板文件**：
   - 检查文件路径是否正确
   - 确保文件包含 `HIP_PROSTHESIS_TEMPLATES` 字典

2. **类别名称不匹配**：
   - 检查模板文件中的类别名称
   - 脚本会输出警告信息，提示哪些类别使用了默认模板

3. **模板加载失败**：
   - 确保Python文件语法正确
   - 检查文件编码（应该是UTF-8）

