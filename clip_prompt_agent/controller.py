# -*- coding: utf-8 -*-
"""
Controller (迭代控制器)：调度 Agent 循环，
将评估结果整理成 Prompt 喂给 LLM，并判断终止条件。
"""

from config import (
    CLASS_NAMES,
    MAX_ITERATIONS,
    STAGNATION_ROUNDS,
    TOP_K_PHRASES,
    FEATURES_PATH,
)
from generator import generate_initial_phrases, generate_refined_phrases
from evaluator import eval_phrases_one_by_one, build_feedback_text


def select_top_k_phrases(per_phrase_results: dict, k: int = TOP_K_PHRASES):
    """
    根据逐条评估的准确率，每类保留准确率最高的 k 个短词。
    per_phrase_results: { class_name: [ (phrase, accuracy), ... ] }，已按 accuracy 降序。
    """
    result = {}
    for name in CLASS_NAMES:
        pairs = per_phrase_results.get(name, [])
        result[name] = [p for p, _ in pairs[:k]]
    return result


def run_agent_loop(
    features_path=None,
    device="cuda",
    max_iter=MAX_ITERATIONS,
    stagnation_rounds=STAGNATION_ROUNDS,
    top_k_ensemble=TOP_K_PHRASES,
):
    """
    执行完整 Agent 迭代循环：逐条短词进 CLIP 算准确率 → 按准确率反馈给 LLM → LLM 结合医生描述与高准确率短词生成新短词。
    返回历史记录与最终每类 Top-K 短词（按准确率排序）。
    """
    features_path = features_path or FEATURES_PATH
    history = []
    best_accuracy = -1.0
    best_phrases = None
    rounds_without_improvement = 0

    # Step 2: 初始种子
    print("[Agent] Round 0: generating initial phrases...")
    phrases_by_class = generate_initial_phrases()
    history.append({"round": 0, "phase": "init", "phrases": {k: list(v) for k, v in phrases_by_class.items()}})
    print("[Agent] Round 0: done.")

    for round_idx in range(1, max_iter + 1):
        # 环节 A：逐条评估（每个短词单独进 CLIP 算准确率）
        print(f"[Agent] Round {round_idx}: evaluating each phrase one by one...", end=" ", flush=True)
        acc, detail, per_phrase_results = eval_phrases_one_by_one(
            phrases_by_class,
            device=device,
            features_path=features_path,
        )
        detail["phrases_by_class"] = {k: list(v) for k, v in phrases_by_class.items()}
        history.append({
            "round": round_idx,
            "accuracy": acc,
            "per_class_acc": detail["per_class_acc"],
            "confusion_matrix": detail["confusion_matrix"].tolist(),
            "phrases": detail["phrases_by_class"],
            "per_phrase_results": {k: [(p, a) for p, a in v] for k, v in per_phrase_results.items()},
        })

        if acc > best_accuracy:
            best_accuracy = acc
            best_phrases = select_top_k_phrases(per_phrase_results, k=top_k_ensemble)
            rounds_without_improvement = 0
            print(f"acc={acc:.2%} (best), refining...")
        else:
            rounds_without_improvement += 1
            print(f"acc={acc:.2%} (no improvement {rounds_without_improvement}/{stagnation_rounds})")

        # 终止条件
        if rounds_without_improvement >= stagnation_rounds:
            print(f"[Agent] Stopping: no improvement for {stagnation_rounds} rounds.")
            break
        if round_idx >= max_iter:
            print(f"[Agent] Stopping: reached max_iter={max_iter}.")
            break

        # 环节 B：反馈（每条短词的准确率给 LLM，便于保留高准确率、抛弃低准确率）
        feedback_text = build_feedback_text(phrases_by_class, detail, top_k_show=3)

        # 环节 C：LLM 根据医生描述 + 高准确率短词生成新短词
        print(f"[Agent] Round {round_idx}: generating refined phrases (from doctor descriptions + high-accuracy phrases)...", end=" ", flush=True)
        phrases_by_class = generate_refined_phrases(feedback_text)
        print("done.")

    return {
        "best_accuracy": best_accuracy,
        "best_phrases": best_phrases,
        "history": history,
        "final_round": len(history) - 1,
    }
