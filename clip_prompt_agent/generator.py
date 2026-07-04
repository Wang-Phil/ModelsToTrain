# -*- coding: utf-8 -*-
"""
Generator (LLM)：根据先验与历史反馈，生成/变异候选短词组合。
使用 DeepSeek 接口（OpenAI 兼容）；前置知识使用骨科医生专业类别描述。
"""

import json
from pathlib import Path
from openai import OpenAI

from config import (
    CLASS_NAMES,
    CLASS_DESCRIPTIONS_PATH,
    DEEPSEEK_API_BASE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_REQUEST_TIMEOUT,
    TEMPERATURE_INIT,
    TEMPERATURE_REFINE,
    INIT_PHRASES_PER_CLASS,
)


def _load_class_descriptions():
    """加载骨科专业类别描述，用于注入 prompt。"""
    path = Path(CLASS_DESCRIPTIONS_PATH)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k in CLASS_NAMES}


def _format_descriptions_block(descriptions: dict) -> str:
    """Format orthopedic class descriptions as prior-knowledge block."""
    if not descriptions:
        return ""
    lines = ["[Orthopedic class descriptions – use these to derive English short phrases only]"]
    for name in CLASS_NAMES:
        desc = descriptions.get(name, "")
        if desc:
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


# 模块加载时读取一次
_CLASS_DESCRIPTIONS = _load_class_descriptions()

SYSTEM_PROMPT = """You are a senior orthopedic surgeon and AI prompt expert. Your task is to provide short, core visual pathology phrases for zero-shot CLIP classification of hip arthroplasty X-rays.
Requirements:
- Every phrase must be in English only, a few words (e.g. 2–6 words), suitable for CLIP text encoder.
- Class-exclusive: Phrases for a class must describe ONLY that class. Do NOT use phrases that could apply to or describe another class (e.g. "Stem Loosening" phrases must mention stem/femoral stem, not acetabular/cup; "Acetabular Loosening" must mention acetabulum/cup, not stem).
- Visually discriminative: Use imaging-specific, non-generic wording so each class is easy to tell apart. Avoid vague terms that fit multiple categories.
- Base phrases on the provided orthopedic class descriptions; keep medical meaning accurate.
- Output must be valid JSON only, no other text."""


def _build_init_user_prompt(n_per_class: int = 10):
    classes_str = ", ".join(CLASS_NAMES)
    prior = _format_descriptions_block(_CLASS_DESCRIPTIONS)
    prior_block = f"\n\n{prior}\n" if prior else ""
    return f"""I need to run CLIP zero-shot classification on hip arthroplasty X-rays. All classes (each phrase must describe ONLY its own class, never another): {classes_str}.
{prior_block}
For each class, provide exactly {n_per_class} short visual pathology phrases in English only (a few words each).
Critical: (1) Phrases for a class must describe ONLY that class—do not use wording that could describe any other class. (2) Use class-specific terms (e.g. "acetabular/cup" only for Acetabular Loosening; "femoral stem/stem" only for Stem Loosening; "dislocation/head out of cup" only for Dislocation). (3) Be visually discriminative and avoid generic descriptions that could fit multiple classes.
Output a single JSON object: keys = class names (exactly as above), values = list of English phrases. Example:
{{"Good Place": ["well positioned prosthesis", "no loosening", ...], "Stem Loosening": ["femoral stem subsidence", "stem radiolucent lines", ...], "Acetabular Loosening": ["acetabular cup migration", "cup bone lucency", ...], ...}}"""


def _build_refine_user_prompt(feedback_text: str, n_per_class: int = 10):
    prior = _format_descriptions_block(_CLASS_DESCRIPTIONS)
    prior_block = f"\nReference class descriptions (keep medical accuracy):\n{prior}\n\n" if prior else ""
    classes_list = json.dumps(CLASS_NAMES)
    return f"""Previous round CLIP validation results:

{feedback_text}
{prior_block}
Using the feedback above:
1. You are given per-phrase accuracy: each phrase was evaluated alone in CLIP. Keep and extend the style of high-accuracy phrases, drop or replace low-accuracy ones.
2. Combine (a) the orthopedic doctor descriptions above and (b) the high-accuracy phrases from the list to generate new phrases—preserve wording that worked well and refine the rest.
3. For confused class pairs, use more discriminative, class-exclusive phrases (e.g. stem vs acetabular wording so they do not describe each other).
4. Each class's phrases must describe ONLY that class—do not use phrases that could apply to another class in this list: {classes_list}.
5. Generate {n_per_class} new candidate phrases per class in English only. Be visually specific and avoid generic wording.

Output a single JSON object: keys = class names (exactly as below), values = list of English phrases. No text outside JSON.
Class names: {classes_list}"""


def _parse_phrases_from_response(text: str):
    """从 LLM 回复中解析 JSON，得到 {class_name: [phrase1, phrase2, ...]}。"""
    text = text.strip()
    # 去除可能的 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 尝试截取首段 { ... }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
        else:
            raise ValueError(f"无法从回复中解析 JSON: {text[:500]}")
    # 统一键名为 config 中的类别名
    result = {}
    for name in CLASS_NAMES:
        # 允许 LLM 用略有差异的键名
        val = data.get(name)
        if val is None:
            for k, v in data.items():
                if k.strip() == name or k.replace(" ", "") == name.replace(" ", ""):
                    val = v
                    break
        if val is None:
            val = []
        result[name] = val if isinstance(val, list) else [str(val)]
    return result


def call_llm(
    user_content: str,
    temperature: float = TEMPERATURE_INIT,
    timeout: float = None,
):
    """调用 DeepSeek（OpenAI 兼容），返回回复内容。带超时，避免无响应挂起。"""
    if not DEEPSEEK_API_KEY:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    t = timeout if timeout is not None else DEEPSEEK_REQUEST_TIMEOUT
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
        timeout=t,
    )
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def generate_initial_phrases(n_per_class: int = None):
    """初始种子生成：每类 n 个短词。"""
    n_per_class = n_per_class or INIT_PHRASES_PER_CLASS
    user = _build_init_user_prompt(n_per_class)
    raw = call_llm(user, temperature=TEMPERATURE_INIT)
    return _parse_phrases_from_response(raw)


def generate_refined_phrases(feedback_text: str, n_per_class: int = None):
    """根据反馈变异/生成新短词。"""
    n_per_class = n_per_class or INIT_PHRASES_PER_CLASS
    user = _build_refine_user_prompt(feedback_text, n_per_class)
    raw = call_llm(user, temperature=TEMPERATURE_REFINE)
    return _parse_phrases_from_response(raw)
