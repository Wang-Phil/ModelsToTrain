# -*- coding: utf-8 -*-
"""
Evaluator (CLIP 离线打分器)：
加载预提取的图像特征 I，对 LLM 生成的短词做 Text 编码得到 T，
计算相似度、准确率与每类混淆情况。
"""

import torch
import numpy as np
from pathlib import Path

from config import CLASS_NAMES, FEATURES_PATH, TOP_K_PHRASES

try:
    import clip
    USE_OPEN_CLIP = False
except ImportError:
    import open_clip
    USE_OPEN_CLIP = True


def load_features(path=None):
    path = path or FEATURES_PATH
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    I = data["I"]
    labels = data["labels"]
    meta = data.get("meta", {})
    return I, labels, meta


def load_clip_text_encoder(device="cuda"):
    if USE_OPEN_CLIP:
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return model, device, tokenizer
    else:
        model, _ = clip.load("ViT-B/32", device=device)
        return model, device, None


def encode_texts(model, texts: list[str], device, use_open_clip: bool, tokenizer=None):
    """对文本列表编码，返回 (N, dim) 的 tensor。"""
    if use_open_clip and tokenizer is not None:
        text_tokens = tokenizer(texts)
        if not isinstance(text_tokens, torch.Tensor):
            text_tokens = torch.tensor(text_tokens)
        text_tokens = text_tokens.to(device)
    else:
        import clip as clip_mod
        text_tokens = clip_mod.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        feats = model.encode_text(text_tokens)
    return feats.cpu().float()


def phrases_to_texts(phrases_by_class: dict) -> tuple[list[str], list[int]]:
    """
    phrases_by_class: {class_name: [phrase1, phrase2, ...]}
    返回 (all_texts, class_indices)，每个 text 对应一个 (class, phrase) 的编码。
    """
    all_texts = []
    class_indices = []
    for cls_name in CLASS_NAMES:
        phrases = phrases_by_class.get(cls_name, [])
        for p in phrases:
            if p and str(p).strip():
                all_texts.append(str(p).strip())
                class_indices.append(CLASS_NAMES.index(cls_name))
    return all_texts, class_indices


def compute_similarity(I: torch.Tensor, T: torch.Tensor, normalize=True):
    """I: (N_img, dim), T: (N_txt, dim). 返回 (N_img, N_txt)。"""
    if normalize:
        I = I / I.norm(dim=-1, keepdim=True)
        T = T / T.norm(dim=-1, keepdim=True)
    return I @ T.T


def eval_phrase_set(
    phrases_by_class: dict,
    I: torch.Tensor,
    labels: torch.Tensor,
    device="cuda",
    features_path=None,
    top_k_ensemble=TOP_K_PHRASES,
):
    """
    评估当前这批短词在验证集上的表现。
    - 若某类有多个短词，取该类下 top_k_ensemble 个短词的特征平均后作为该类文本特征（Ensemble）。
    - 然后做 I @ T.T，每张图预测为相似度最大的类。
    返回：
      accuracy, per_class_acc, confusion_matrix (np), detail_dict
    """
    I, labels, _ = load_features(features_path)
    I = I.float()
    labels = labels.numpy()
    n_class = len(CLASS_NAMES)
    model, dev, tokenizer = load_clip_text_encoder(device)

    # 为每个类别构建 ensemble 文本特征：该类下短语编码后取平均
    class_text_feats = []
    for cls_name in CLASS_NAMES:
        phrases = phrases_by_class.get(cls_name, [])
        phrases = [p for p in phrases if p and str(p).strip()][:top_k_ensemble * 2]
        if not phrases:
            class_text_feats.append(torch.zeros(I.shape[1]))
            continue
        texts = [str(p).strip() for p in phrases[:top_k_ensemble]]
        if not texts:
            class_text_feats.append(torch.zeros(I.shape[1]))
            continue
        T = encode_texts(model, texts, dev, USE_OPEN_CLIP, tokenizer)
        T = T.mean(dim=0, keepdim=True)
        class_text_feats.append(T.squeeze(0))
    T = torch.stack(class_text_feats, dim=0)
    T = T / (T.norm(dim=-1, keepdim=True) + 1e-8)
    I_n = I / (I.norm(dim=-1, keepdim=True) + 1e-8)
    logits = (I_n @ T.T).numpy()
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()

    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for p, g in zip(preds, labels):
        confusion[g, p] += 1
    per_class_acc = np.diag(confusion) / (confusion.sum(axis=1) + 1e-8)

    detail = {
        "accuracy": float(accuracy),
        "per_class_acc": {CLASS_NAMES[i]: float(per_class_acc[i]) for i in range(n_class)},
        "confusion_matrix": confusion,
    }
    return accuracy, per_class_acc, confusion, detail


def eval_phrases_one_by_one(
    phrases_by_class: dict,
    device="cuda",
    features_path=None,
):
    """
    逐条短词评估：每个短词单独作为该类的文本送入 CLIP，其余类用类别名，算一次准确率。
    返回 per_phrase_results: { class_name: [ (phrase, accuracy), ... ] }，以及 round_accuracy（每类取最高分短词组成 9 条再算的整体准确率）。
    """
    I, labels, _ = load_features(features_path)
    I = I.float()
    labels = labels.numpy()
    n_class = len(CLASS_NAMES)
    model, dev, tokenizer = load_clip_text_encoder(device)
    I_n = I / (I.norm(dim=-1, keepdim=True) + 1e-8)

    # 每个类别名编码一次，用于“其他类”的占位
    base_texts = list(CLASS_NAMES)
    base_T = encode_texts(model, base_texts, dev, USE_OPEN_CLIP, tokenizer)
    base_T = base_T / (base_T.norm(dim=-1, keepdim=True) + 1e-8)

    per_phrase_results = {}
    best_phrase_per_class = []  # 每类选准确率最高的一条，用于算 round_accuracy

    for c, cls_name in enumerate(CLASS_NAMES):
        phrases = [p for p in phrases_by_class.get(cls_name, []) if p and str(p).strip()]
        if not phrases:
            per_phrase_results[cls_name] = []
            best_phrase_per_class.append(base_texts[c])
            continue
        accs = []
        for p in phrases:
            texts = list(base_texts)
            texts[c] = str(p).strip()
            T = encode_texts(model, texts, dev, USE_OPEN_CLIP, tokenizer)
            T = T / (T.norm(dim=-1, keepdim=True) + 1e-8)
            logits = (I_n @ T.T).numpy()
            preds = np.argmax(logits, axis=1)
            acc = (preds == labels).mean()
            accs.append((p, float(acc)))
        accs.sort(key=lambda x: -x[1])
        per_phrase_results[cls_name] = accs
        best_phrase_per_class.append(accs[0][0])

    # round_accuracy: 每类用准确率最高的那条短词，组成 9 条再算整体准确率
    T_best = encode_texts(model, best_phrase_per_class, dev, USE_OPEN_CLIP, tokenizer)
    T_best = T_best / (T_best.norm(dim=-1, keepdim=True) + 1e-8)
    logits = (I_n @ T_best.T).numpy()
    preds = np.argmax(logits, axis=1)
    round_accuracy = (preds == labels).mean()

    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for p, g in zip(preds, labels):
        confusion[g, p] += 1
    per_class_acc = np.diag(confusion) / (confusion.sum(axis=1) + 1e-8)
    detail = {
        "accuracy": float(round_accuracy),
        "per_class_acc": {CLASS_NAMES[i]: float(per_class_acc[i]) for i in range(n_class)},
        "confusion_matrix": confusion,
        "per_phrase_results": per_phrase_results,
    }
    return round_accuracy, detail, per_phrase_results


def build_feedback_text(
    phrases_by_class: dict,
    detail: dict,
    top_k_show=3,
):
    """
    Build feedback text for the LLM. If detail contains per_phrase_results (from eval_phrases_one_by_one),
    list each phrase with its accuracy so the LLM can keep high-accuracy and drop low-accuracy phrases.
    """
    lines = []
    acc = detail["accuracy"]
    n_class = len(CLASS_NAMES)
    per_phrase_results = detail.get("per_phrase_results")

    if per_phrase_results is not None:
        lines.append(f"Overall accuracy (using best single phrase per class): {acc:.2%}.")
        lines.append("")
        lines.append("Per-phrase accuracy (each phrase was evaluated alone as the text for that class). Use doctor descriptions and the high-accuracy phrases to generate new phrases.")
        lines.append("")
        for cls_name in CLASS_NAMES:
            pairs = per_phrase_results.get(cls_name, [])
            if not pairs:
                lines.append(f"[{cls_name}] no phrases.")
                continue
            parts = [f"'{phrase}' {a:.1%}" for phrase, a in pairs]
            lines.append(f"[{cls_name}] " + "; ".join(parts) + ".")
        lines.append("")
        cm = detail.get("confusion_matrix")
        if cm is not None:
            confusions = []
            for i in range(n_class):
                for j in range(n_class):
                    if i != j and cm[i, j] > 0:
                        confusions.append((cm[i, j], i, j))
            confusions.sort(reverse=True, key=lambda x: x[0])
            for _, i, j in confusions[:5]:
                lines.append(f"Often confused: [{CLASS_NAMES[i]}] predicted as [{CLASS_NAMES[j]}] ({cm[i, j]} cases).")
        return "\n".join(lines)

    # 兼容旧版：无 per_phrase_results 时按原逻辑
    per_class = detail.get("per_class_acc", {})
    cm = detail.get("confusion_matrix", np.zeros((n_class, n_class)))
    lines.append(f"Overall accuracy: {acc:.2%}.")
    lines.append("")
    for i in range(n_class):
        name = CLASS_NAMES[i]
        phrases = phrases_by_class.get(name, [])
        p_acc = per_class.get(name, 0.0)
        phrase_sample = phrases[:top_k_show] if phrases else ["(none)"]
        lines.append(f"[{name}] accuracy {p_acc:.2%}; phrase examples: {phrase_sample}.")
    lines.append("")
    confusions = []
    for i in range(n_class):
        for j in range(n_class):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], i, j))
    confusions.sort(reverse=True, key=lambda x: x[0])
    for _, i, j in confusions[:5]:
        lines.append(f"Often confused: [{CLASS_NAMES[i]}] predicted as [{CLASS_NAMES[j]}] ({cm[i, j]} cases).")
    return "\n".join(lines)
