# -*- coding: utf-8 -*-
"""Agent 配置：路径、API、CLIP 模型与迭代参数。"""

import os
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
# Agent 使用的图像目录：默认用训练数据 single_label_data，每类最多取 50 张
VALID_IMAGE_DIR = os.environ.get(
    "HIP_VALID_DIR",
    str(PROJECT_ROOT / "single_label_data"),
)
FEATURES_PATH = ROOT / "cache" / "validation_features.pt"
# 提取特征时每类最多使用的图片数
MAX_IMAGES_PER_CLASS = int(os.environ.get("HIP_MAX_IMAGES_PER_CLASS", "50"))
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 骨科专业类别描述（前置知识） ----------
# 每类别的长描述，用于注入 LLM prompt，便于生成贴合专业的短词
CLASS_DESCRIPTIONS_PATH = os.environ.get(
    "HIP_CLASS_DESCRIPTIONS",
    str(PROJECT_ROOT / "class_texts_hip_prosthesis.json"),
)

# ---------- 类别（与 class_texts_hip_prosthesis_short.json 一致） ----------
CLASS_NAMES = [
    "Acetabular Loosening",
    "Dislocation",
    "Fracture",
    "Good Place",
    "Native Hip",
    "Spacer",
    "Stem Loosening",
    "Infection",
    "Wear",
]

# ---------- DeepSeek API（与项目 auto_prompt_optimizer 一致，硅基流动） ----------
# 生产环境建议用环境变量 DEEPSEEK_API_KEY 覆盖，避免 key 进版本库
DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.siliconflow.cn/v1")
DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "sk-mqubfpfslyohpdbryxjsnrntckfdizhhwrgsviwdisyabccq",
)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3.2")
# 单次请求超时（秒），避免无响应一直挂起
DEEPSEEK_REQUEST_TIMEOUT = float(os.environ.get("DEEPSEEK_REQUEST_TIMEOUT", "120"))

# 初始生成：高 Temperature 增加多样性
TEMPERATURE_INIT = 0.8
# 根据反馈微调：低 Temperature 稳定收敛
TEMPERATURE_REFINE = 0.3

# ---------- CLIP ----------
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B/32")
DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES", "cuda")  # 无 GPU 时改为 "cpu"

# ---------- Agent 迭代 ----------
MAX_ITERATIONS = 20
STAGNATION_ROUNDS = 3  # 连续 N 轮准确率不提升则停止
TOP_K_PHRASES = 5      # 每类保留得分最高的 K 个短词，用于 Ensemble
INIT_PHRASES_PER_CLASS = 10  # 初始每类生成短词数量
