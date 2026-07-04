# -*- coding: utf-8 -*-
"""
入口：先执行离线特征提取（若尚未生成），再运行 Agent 迭代循环，
输出每类 Top-K 提示词及最终准确率。
"""

import argparse
import json
from pathlib import Path

from config import (
    CLASS_NAMES,
    VALID_IMAGE_DIR,
    FEATURES_PATH,
    MAX_IMAGES_PER_CLASS,
    MAX_ITERATIONS,
    STAGNATION_ROUNDS,
    TOP_K_PHRASES,
)
from offline_features import extract_and_save
from controller import run_agent_loop
from generator import generate_initial_phrases


def main():
    parser = argparse.ArgumentParser(description="CLIP 短词优化 Agent（髋关节分类）")
    parser.add_argument(
        "mode",
        nargs="?",
        default="agent",
        choices=["extract", "agent", "full", "test"],
        help="extract=仅提取验证集特征; agent=仅运行 Agent(需已有特征); full=先 extract 再 agent; test=仅测 Generator(无需特征)",
    )
    parser.add_argument("--image_root", default=VALID_IMAGE_DIR, help="验证集图片根目录")
    parser.add_argument("--features", default=str(FEATURES_PATH), help="特征缓存 .pt 路径")
    parser.add_argument("--max_iter", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--stagnation", type=int, default=STAGNATION_ROUNDS)
    parser.add_argument("--top_k", type=int, default=TOP_K_PHRASES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="最终 Top-K 短词与历史保存路径（JSON）")
    parser.add_argument("--max_per_class", type=int, default=MAX_IMAGES_PER_CLASS, help="提取特征时每类最多图片数（默认 50）")
    args = parser.parse_args()

    if args.mode in ("extract", "full"):
        print("Step 1: 离线特征固化...")
        extract_and_save(
            image_root=args.image_root,
            output_path=args.features,
            device=args.device,
            max_per_class=args.max_per_class,
        )
        if args.mode == "extract":
            return

    if args.mode == "test":
        print("Test 模式：仅运行 Generator（初始短词生成），无需特征文件...")
        phrases = generate_initial_phrases()
        print("每类生成的短词：")
        for name in CLASS_NAMES:
            print(f"  {name}: {phrases.get(name, [])}")
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(phrases, f, ensure_ascii=False, indent=2)
            print(f"已保存: {args.output}")
        return

    if args.mode in ("agent", "full"):
        if not Path(args.features).exists():
            print("未找到特征文件，请先运行 mode=extract 或 full。")
            return
        print("Step 2 & 3: 初始生成 + Agent 迭代...")
        result = run_agent_loop(
            features_path=args.features,
            device=args.device,
            max_iter=args.max_iter,
            stagnation_rounds=args.stagnation,
            top_k_ensemble=args.top_k,
        )
        print(f"最佳验证集准确率: {result['best_accuracy']:.2%} (round {result['final_round']})")
        print("每类 Top-{} 短词（Ensemble 用）:".format(args.top_k))
        for name in CLASS_NAMES:
            phrases = result["best_phrases"].get(name, [])
            print(f"  {name}: {phrases}")

        out_path = args.output
        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # 历史中 confusion 等可转为 list 便于 JSON
            dump = {
                "best_accuracy": result["best_accuracy"],
                "best_phrases": result["best_phrases"],
                "final_round": result["final_round"],
                "history": result["history"],
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(dump, f, ensure_ascii=False, indent=2)
            print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
