# CLIP 短词优化 Agent（髋关节置换术后 X 光分类）

用 LLM（DeepSeek）+ CLIP 离线打分 + 迭代控制器，自动优化每类「核心视觉短语」，提升 Zero-Shot 分类准确率。

## 结构

- **Generator** (`generator.py`)：DeepSeek API，根据反馈生成/变异候选短词。
- **Evaluator** (`evaluator.py`)：用预提取的图像特征 I 与文本特征 T 算相似度，得到准确率与混淆矩阵。
- **Controller** (`controller.py`)：调度「评估 → 反馈 → 生成」循环，直至收敛或达到最大轮数。

## 环境

```bash
cd /home/ln/wangweicheng/ModelsTotrain/clip_prompt_agent
pip install -r requirements.txt
# CLIP 二选一：
pip install git+https://github.com/openai/CLIP.git
# 或
pip install open-clip-torch
```

## 配置

- **前置知识（骨科专业类别描述）**：Generator 会加载「每类别的专业描述」并注入到每次 prompt 中，使生成的短词贴合医学含义。默认读取项目根目录下的 `class_texts_hip_prosthesis.json`（键为类别名，值为该类的英文长描述）。可通过环境变量 `HIP_CLASS_DESCRIPTIONS` 指定其它 JSON 路径；若文件不存在则不加前置知识。
- **DeepSeek**：设置 `DEEPSEEK_API_KEY`，可选 `DEEPSEEK_API_BASE`、`DEEPSEEK_MODEL`。
- **验证集**：按类别分子目录放置图片，如：
  ```
  validation/
    Good Place/
      img1.jpg
      img2.png
    Stem Loosening/
      ...
  ```
  通过 `HIP_VALID_DIR` 或 `--image_root` 指定该目录。
- **特征缓存**：默认 `cache/validation_features.pt`，可用 `--features` 覆盖。

## 使用

1. **仅提取验证集图像特征（只做一次）**
   ```bash
   export DEEPSEEK_API_KEY=your_key
   python main.py extract --image_root /path/to/validation --output cache/validation_features.pt
   ```

2. **仅运行 Agent（需已存在特征文件）**
   ```bash
   python main.py agent --features cache/validation_features.pt --output results/run1.json
   ```

3. **先提取特征再跑 Agent**
   ```bash
   python main.py full --image_root /path/to/validation --output results/run1.json
   ```

输出：终端打印每类 **Top-K 短词**（用于 Ensemble）；若指定 `--output`，会保存 JSON（含 best_phrases、best_accuracy、history）。

## 参数摘要

| 参数 | 说明 |
|------|------|
| `--max_iter` | 最大迭代轮数（默认 20） |
| `--stagnation` | 连续几轮准确率不提升则停止（默认 3） |
| `--top_k` | 每类保留短语数，用于 Ensemble（默认 5） |
| `--device` | cuda / cpu |

## 注意

- 验证集不要与最终测试集重合，避免提示词级过拟合。
- 初始生成用较高 Temperature，反馈微调用较低 Temperature（在 `config.py` 中可调）。
