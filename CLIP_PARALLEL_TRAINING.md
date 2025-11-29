# CLIP 并行训练说明

## 更新内容

`run_train_clip.sh` 已更新为支持**并行训练多个配置**，参考了 `run_multi_model_cv.sh` 的实现方式。

## 主要改进

### 1. 多 GPU 支持

- **之前**: 只支持单个 GPU (`GPU_ID=5`)
- **现在**: 支持多个 GPU (`GPUS=(5)` 或 `GPUS=(0 1 2 3)`)

### 2. 并行执行

- **之前**: 所有配置在同一个进程中**顺序执行**
- **现在**: 每个配置在**独立的后台进程**中**并行执行**

### 3. GPU 自动分配

- 多个配置会自动循环分配到不同的 GPU
- 例如：3个配置，2个GPU → GPU0: 配置1, GPU1: 配置2, GPU0: 配置3

### 4. 独立日志文件

- **之前**: 所有配置的日志输出到同一个文件
- **现在**: 每个配置有独立的日志文件

## 使用方法

### 基本配置

```bash
# GPU列表（支持多个GPU）
GPUS=(5)              # 单个GPU
# GPUS=(0 1 2 3)     # 多个GPU

# 图像编码器列表
IMAGE_ENCODERS=(
    "starnet_s1"
    "starnet_s2"
    "starnet_s3"
)

# 文本编码器列表
TEXT_ENCODERS=(
    "clip:ViT-B/32"
    "bert-base-chinese"
)

# 启用多配置并行训练
USE_MULTI_CONFIG=true
```

### 运行脚本

```bash
bash run_train_clip.sh
```

## 执行流程

1. **检查GPU可用性**: 验证所有指定的GPU是否可用
2. **生成配置组合**: 自动生成所有图像编码器 × 文本编码器的组合
3. **分配GPU**: 循环分配GPU给每个配置
4. **并行启动**: 每个配置在后台独立进程启动
5. **等待完成**: 等待所有训练任务完成
6. **生成总结**: 显示成功和失败的配置

## 输出结构

### 日志文件

```
logs/clip_training/
├── starnet_s1_clip_ViT-B_32_gpu5_20251128_120000.log
├── starnet_s1_bert-base-chinese_gpu5_20251128_120001.log
├── starnet_s2_clip_ViT-B_32_gpu5_20251128_120002.log
└── ...
```

### 输出目录

```
checkpoints/clip_models/
├── starnet_s1_clip_ViT-B_32/
├── starnet_s1_bert-base-chinese/
├── starnet_s2_clip_ViT-B_32/
└── ...
```

## 特性说明

### 1. GPU 等待机制

- 如果GPU使用率 > 70%，会等待直到可用
- 最多等待5分钟，超时后继续执行

### 2. 进程管理

- 使用 `declare -A pids` 跟踪所有后台进程
- 使用 `wait` 等待所有进程完成
- 记录每个配置的成功/失败状态

### 3. 错误处理

- 单个配置失败不会影响其他配置
- 所有配置完成后显示总结报告
- 失败的配置会显示日志文件路径

## 配置示例

### 示例 1: 单 GPU 并行训练

```bash
GPUS=(5)
IMAGE_ENCODERS=(
    "starnet_s1"
    "starnet_s2"
)
TEXT_ENCODERS=(
    "clip:ViT-B/32"
)
USE_MULTI_CONFIG=true
```

**结果**: 2个配置在GPU 5上**顺序执行**（因为只有一个GPU）

### 示例 2: 多 GPU 并行训练

```bash
GPUS=(0 1 2 3)
IMAGE_ENCODERS=(
    "starnet_s1"
    "starnet_s2"
    "starnet_s3"
    "starnet_s4"
)
TEXT_ENCODERS=(
    "clip:ViT-B/32"
)
USE_MULTI_CONFIG=true
```

**结果**: 4个配置在4个GPU上**并行执行**

### 示例 3: 使用配置文件

```bash
GPUS=(0 1)
CONFIG_FILE="train_clip_config.json"
USE_MULTI_CONFIG=true
```

**结果**: 从配置文件读取配置，分配到2个GPU并行执行

## 性能对比

### 顺序执行（旧方式）

- 3个配置，每个需要2小时
- **总时间**: 6小时

### 并行执行（新方式）

- 3个配置，2个GPU
- **总时间**: 约4小时（第一个GPU运行2个配置，第二个GPU运行1个配置）

## 注意事项

1. **显存管理**: 确保每个GPU有足够的显存
2. **日志查看**: 每个配置有独立日志，便于调试
3. **GPU负载**: 脚本会自动检查GPU使用率
4. **进程数**: 同时运行的进程数 = min(配置数, GPU数)

## 故障排除

### 问题 1: GPU 不可用

```
[ERROR] GPU 5 不可用，跳过
```

**解决**: 检查GPU ID是否正确，或使用 `nvidia-smi` 查看可用GPU

### 问题 2: 显存不足

**解决**: 
- 减少同时运行的配置数
- 增加GPU数量
- 减小批次大小

### 问题 3: 配置读取失败

**解决**: 检查配置文件格式是否正确（JSON格式）

## 总结

新的并行训练方式可以：
- ✅ 充分利用多GPU资源
- ✅ 显著缩短总训练时间
- ✅ 独立日志便于调试
- ✅ 自动GPU分配和等待
- ✅ 完善的错误处理

现在可以高效地并行训练多个CLIP配置了！

