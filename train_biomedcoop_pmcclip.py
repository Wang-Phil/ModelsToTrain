#!/usr/bin/env python
"""
BiomedCoOp PMC-CLIP 模型训练脚本
使用 Dassl 框架训练 BiomedCoOp_PMCCLIP 模型
"""

import argparse
import os
import sys
from pathlib import Path

# 添加 Dassl 路径
dassl_path = Path(__file__).parent / "Dassl.pytorch"
if str(dassl_path) not in sys.path:
    sys.path.insert(0, str(dassl_path))

import torch
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


def print_args(args, cfg):
    print("=" * 80)
    print("Arguments")
    print("=" * 80)
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("=" * 80)
    print("Config")
    print("=" * 80)
    print(cfg)
    print("=" * 80)


def reset_cfg(cfg, args):
    """根据命令行参数更新配置"""
    if args.root:
        cfg.DATASET.ROOT = args.root
    
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    
    if args.resume:
        cfg.RESUME = args.resume
    
    if args.seed is not None:
        cfg.SEED = args.seed
    
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    
    # BiomedCoOp 特定配置
    if args.n_ctx is not None:
        cfg.TRAINER.BIOMEDCOOP.N_CTX = args.n_ctx
    
    if args.ctx_init:
        cfg.TRAINER.BIOMEDCOOP.CTX_INIT = args.ctx_init
    
    if args.csc is not None:
        cfg.TRAINER.BIOMEDCOOP.CSC = args.csc
    
    if args.class_token_position:
        cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = args.class_token_position
    
    if args.sccm_lambda is not None:
        cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = args.sccm_lambda
    
    if args.kdsp_lambda is not None:
        cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = args.kdsp_lambda
    
    if args.tau is not None:
        cfg.TRAINER.BIOMEDCOOP.TAU = args.tau
    
    if args.n_prompts is not None:
        cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = args.n_prompts
    
    if args.prec:
        cfg.TRAINER.BIOMEDCOOP.PREC = args.prec


def extend_cfg(cfg):
    """
    扩展配置，添加 BiomedCoOp 特定配置
    """
    from yacs.config import CfgNode as CN
    
    # 添加 BiomedCoOp 配置节点
    cfg.TRAINER.BIOMEDCOOP = CN()
    cfg.TRAINER.BIOMEDCOOP.N_CTX = 4  # 上下文token数量
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"  # 上下文初始化文本
    cfg.TRAINER.BIOMEDCOOP.CSC = False  # 是否使用类别特定的上下文
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"  # 类别token位置
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 1.0  # SCCM损失权重
    cfg.TRAINER.BIOMEDCOOP.KDSP_LAMBDA = 1.0  # KDSP损失权重
    cfg.TRAINER.BIOMEDCOOP.TAU = 1.0  # 用于选择prompt的阈值
    cfg.TRAINER.BIOMEDCOOP.N_PROMPTS = 4  # 使用的prompt数量
    cfg.TRAINER.BIOMEDCOOP.PREC = "amp"  # 精度：fp16, fp32, amp


def setup_cfg(args):
    """设置配置"""
    cfg = get_cfg_default()
    extend_cfg(cfg)
    
    # 1. 从数据集配置文件加载
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    
    # 2. 从方法配置文件加载
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    # 3. 从命令行参数更新
    reset_cfg(cfg, args)
    
    # 4. 从可选参数列表更新
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="训练 BiomedCoOp PMC-CLIP 模型")
    
    # 基本参数
    parser.add_argument("--root", type=str, default="", help="数据集根目录")
    parser.add_argument("--output-dir", type=str, default="", help="输出目录")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的检查点路径")
    parser.add_argument("--seed", type=int, default=-1, help="随机种子（-1表示随机）")
    
    # 数据集和方法配置
    parser.add_argument("--dataset-config-file", type=str, default="", help="数据集配置文件路径")
    parser.add_argument("--config-file", type=str, default="", help="方法配置文件路径")
    
    # 模型参数
    parser.add_argument("--trainer", type=str, default="BiomedCoOp_PMCCLIP", help="训练器名称")
    parser.add_argument("--backbone", type=str, default="", help="backbone名称")
    
    # BiomedCoOp 特定参数
    parser.add_argument("--n-ctx", type=int, default=None, help="上下文token数量（默认4）")
    parser.add_argument("--ctx-init", type=str, default=None, help="上下文初始化文本")
    parser.add_argument("--csc", type=bool, default=None, help="是否使用类别特定的上下文")
    parser.add_argument("--class-token-position", type=str, default=None, 
                       choices=["end", "middle", "front"], help="类别token位置")
    parser.add_argument("--sccm-lambda", type=float, default=None, help="SCCM损失权重（默认1.0）")
    parser.add_argument("--kdsp-lambda", type=float, default=None, help="KDSP损失权重（默认1.0）")
    parser.add_argument("--tau", type=float, default=None, help="用于选择prompt的阈值（默认1.0）")
    parser.add_argument("--n-prompts", type=int, default=None, help="使用的prompt数量（默认4）")
    parser.add_argument("--prec", type=str, default=None, choices=["fp16", "fp32", "amp"],
                       help="训练精度（默认amp）")
    
    # 其他参数
    parser.add_argument("--opts", nargs="*", help="其他配置选项（key=value格式）")
    
    args = parser.parse_args()
    
    # 设置配置
    cfg = setup_cfg(args)
    
    # 打印参数和配置
    print_args(args, cfg)
    
    # 设置随机种子
    if cfg.SEED >= 0:
        print(f"设置固定随机种子: {cfg.SEED}")
        set_random_seed(cfg.SEED)
    
    # 设置日志
    setup_logger(cfg.OUTPUT_DIR)
    
    # 打印环境信息
    print("\n" + "=" * 80)
    print("Environment Info")
    print("=" * 80)
    print(collect_env_info())
    print("=" * 80 + "\n")
    
    # 构建训练器
    trainer = build_trainer(cfg)
    
    # 开始训练
    trainer.train()
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"输出目录: {cfg.OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

