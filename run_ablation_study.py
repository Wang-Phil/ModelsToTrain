#!/usr/bin/env python3
"""
消融实验脚本
对比不同组件组合的效果

Components:
1. CLIP Loss only
2. SupCon Loss only
3. SupCon + CLIP
4. SupCon + CLIP + Class Loss
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time
import threading
import queue

def create_ablation_configs(base_config_path="train_clip_config.json"):
    """
    创建消融实验的配置文件
    """
    
    # 读取基础配置
    if Path(base_config_path).exists():
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = json.load(f)[0]  # 取第一个配置作为基础
    else:
        # 默认配置
        base_config = {
            "image_encoder_name": "resnet18",
            "text_encoder_name": "clip:ViT-B/32",
            "embed_dim": 512,
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "temperature": 0.07,
            "img_size": 224,
            "augmentation": "standard",
            "num_workers": 4,
            "use_amp": True,
            "save_best": True,
            "use_cv": True,
            "n_splits": 5,
            "random_state": 42,
            "early_stopping_patience": None,
            "early_stopping_min_delta": 0.001,
            "early_stopping_monitor": "val_loss",
            "class_texts_file": "class_texts_hip_prosthesis.json",
            "use_weighted_sampling": True,
            "weight_method": "inverse_freq",
            "weight_smooth_factor": 1.0,
            "freeze_image_encoder": False,
            "freeze_text_encoder": False,
        }
    
    # 定义消融实验配置
    ablation_configs = [
        {
            "name": "clip_only",
            "display_name": "CLIP Loss only",
            "description": "仅使用CLIP损失（图像-文本对比学习）",
            "config": {
                **base_config,
                "use_superclip_loss": False,
                "use_supcon_loss": False,
                "use_lsal_loss": False,
                "comment": "Ablation: CLIP Loss only"
            }
        },
        {
            "name": "supcon_only",
            "display_name": "SupCon Loss only",
            "description": "仅使用SupCon损失（有监督对比学习）",
            "config": {
                **base_config,
                "use_superclip_loss": False,
                "use_supcon_loss": True,
                "use_lsal_loss": False,
                "supcon_temperature": 0.07,
                "supcon_loss_weight": 1.0,
                "clip_loss_weight": 0.0,  # 不使用CLIP Loss
                "class_loss_weight": 0.0,  # 不使用分类损失
                "comment": "Ablation: SupCon Loss only"
            }
        },
        {
            "name": "supcon_clip",
            "display_name": "SupCon + CLIP",
            "description": "SupCon损失 + CLIP损失（无分类损失）",
            "config": {
                **base_config,
                "use_superclip_loss": False,
                "use_supcon_loss": True,
                "use_lsal_loss": False,
                "supcon_temperature": 0.07,
                "supcon_loss_weight": 1.0,
                "clip_loss_weight": 1.0,
                "class_loss_weight": 0.0,  # 不使用分类损失
                "comment": "Ablation: SupCon + CLIP"
            }
        },
        {
            "name": "supcon_clip_class",
            "display_name": "SupCon + CLIP + Class Loss",
            "description": "完整方法：SupCon损失 + CLIP损失 + 分类损失",
            "config": {
                **base_config,
                "use_superclip_loss": False,
                "use_supcon_loss": True,
                "use_lsal_loss": False,
                "supcon_temperature": 0.07,
                "supcon_loss_weight": 1.0,
                "clip_loss_weight": 1.0,
                "class_loss_weight": 1.0,  # 使用分类损失
                "use_focal_loss": True,
                "focal_alpha": 0.25,
                "focal_gamma": 2.0,
                "comment": "Ablation: SupCon + CLIP + Class Loss (Full Method)"
            }
        }
    ]
    
    return ablation_configs


def run_single_experiment(config_dict, data_dir, output_base_dir, gpu_id, log_dir):
    """
    运行单个实验配置
    """
    name = config_dict["name"]
    config = config_dict["config"].copy()  # 复制配置，避免修改原始配置
    
    # 确保使用命令行传入的GPU ID，覆盖配置文件中的gpu_id
    config["gpu_id"] = gpu_id
    
    print(f"\n{'='*80}")
    print(f"运行实验: {config_dict['display_name']}")
    print(f"描述: {config_dict['description']}")
    print(f"{'='*80}")
    
    # 创建临时配置文件
    temp_config_file = Path(f"/tmp/ablation_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(temp_config_file, 'w', encoding='utf-8') as f:
        json.dump([config], f, indent=2, ensure_ascii=False)
    
    # 设置输出目录
    output_dir = Path(output_base_dir) / f"ablation_{name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建训练命令
    cmd = [
        "python", "train_clip.py",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--config-file", str(temp_config_file),
        "--multi-config",
        "--gpu-id", str(gpu_id)
    ]
    
    # 创建日志文件
    log_file = Path(log_dir) / f"ablation_{name}_gpu{gpu_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    print(f"日志文件: {log_file}")
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    # 运行训练
    start_time = time.time()
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent
            )
        
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        if result.returncode == 0:
            print(f"✓ 实验 {config_dict['display_name']} 完成")
            print(f"  耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
            return True, output_dir, elapsed_time
        else:
            print(f"✗ 实验 {config_dict['display_name']} 失败 (退出码: {result.returncode})")
            print(f"  查看日志: {log_file}")
            return False, output_dir, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ 实验 {config_dict['display_name']} 出错: {e}")
        return False, output_dir, elapsed_time
    finally:
        # 清理临时配置文件
        if temp_config_file.exists():
            temp_config_file.unlink()


def extract_results_from_checkpoint(checkpoint_dir):
    """
    从checkpoint目录提取结果
    """
    checkpoint_dir = Path(checkpoint_dir)
    cv_summary_path = checkpoint_dir / 'cv_summary.json'
    
    if not cv_summary_path.exists():
        return None
    
    try:
        with open(cv_summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        avg_results = summary.get('average_results', {})
        
        results = {
            'accuracy': avg_results.get('avg_best_val_acc', 0),
            'mAP': avg_results.get('avg_best_val_mAP', 0),
            'precision': avg_results.get('avg_best_precision', avg_results.get('avg_best_val_mAP', 0)),
            'recall': avg_results.get('avg_best_recall', 0),
            'f1': avg_results.get('avg_best_f1', 0),
            'std_accuracy': avg_results.get('std_best_val_acc', 0),
            'std_mAP': avg_results.get('std_best_val_mAP', 0),
            'std_precision': avg_results.get('std_best_precision', avg_results.get('std_best_val_mAP', 0)),
            'std_recall': avg_results.get('std_best_recall', 0),
            'std_f1': avg_results.get('std_best_f1', 0),
        }
        
        return results
    except Exception as e:
        print(f"警告: 无法读取结果 {cv_summary_path}: {e}")
        return None


def generate_ablation_table(results_dict, output_file="ablation_table.md"):
    """
    生成消融实验表格（Markdown格式）
    """
    output_path = Path(output_file)
    
    # 生成 Markdown 表格
    md_content = "# 消融实验结果\n\n"
    md_content += "## 表：消融实验对比\n\n"
    md_content += "| Components | Accuracy (%) | mAP (%) |\n"
    md_content += "|------------|--------------|----------|\n"
    
    # 按顺序排列
    order = ['clip_only', 'supcon_only', 'supcon_clip', 'supcon_clip_class']
    
    for name in order:
        if name not in results_dict:
            continue
            
        info = results_dict[name]
        display_name = info['display_name']
        results = info.get('results')
        
        if results is None:
            md_content += f"| {display_name} | - | - |\n"
        else:
            acc = results.get('accuracy', 0)
            acc_std = results.get('std_accuracy', 0)
            mAP = results.get('mAP', 0)
            mAP_std = results.get('std_mAP', 0)
            
            # 格式化：平均值 ± 标准差
            acc_str = f"{acc:.2f} ± {acc_std:.2f}" if acc_std > 0 else f"{acc:.2f}"
            mAP_str = f"{mAP:.2f} ± {mAP_std:.2f}" if mAP_std > 0 else f"{mAP:.2f}"
            
            # 如果是完整方法，加粗
            if name == 'supcon_clip_class':
                md_content += f"| **{display_name}** | **{acc_str}** | **{mAP_str}** |\n"
            else:
                md_content += f"| {display_name} | {acc_str} | {mAP_str} |\n"
    
    md_content += "\n"
    md_content += "**说明：**\n"
    md_content += "- 所有结果基于 5 折交叉验证\n"
    md_content += "- 数值格式：平均值 ± 标准差\n"
    md_content += "- Accuracy: 最佳验证准确率\n"
    md_content += "- mAP: 宏平均平均精度\n"
    
    # 保存 Markdown 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n✓ 消融实验表格已保存到: {output_path}")
    
    # 同时生成 LaTeX 表格
    latex_file = output_path.with_suffix('.tex')
    generate_latex_ablation_table(results_dict, latex_file)
    
    return output_path


def generate_latex_ablation_table(results_dict, output_file):
    """
    生成 LaTeX 表格
    """
    latex_content = r"""\begin{table}[htbp]
\centering
\caption{消融实验结果}
\label{tab:ablation_study}
\begin{tabular}{lcc}
\toprule
\textbf{Components} & \textbf{Accuracy (\%)} & \textbf{mAP (\%)} \\
\midrule
"""
    
    # 按顺序排列
    order = ['clip_only', 'supcon_only', 'supcon_clip', 'supcon_clip_class']
    
    for name in order:
        if name not in results_dict:
            continue
            
        info = results_dict[name]
        display_name = info['display_name']
        results = info.get('results')
        
        if results is None:
            latex_content += f"{display_name} & - & - \\\\\n"
        else:
            acc = results.get('accuracy', 0)
            acc_std = results.get('std_accuracy', 0)
            mAP = results.get('mAP', 0)
            mAP_std = results.get('std_mAP', 0)
            
            acc_str = f"${acc:.2f} \\pm {acc_std:.2f}$" if acc_std > 0 else f"${acc:.2f}$"
            mAP_str = f"${mAP:.2f} \\pm {mAP_std:.2f}$" if mAP_std > 0 else f"${mAP:.2f}$"
            
            if name == 'supcon_clip_class':
                latex_content += f"\\textbf{{{display_name}}} & \\textbf{{{acc_str}}} & \\textbf{{{mAP_str}}} \\\\\n"
            else:
                latex_content += f"{display_name} & {acc_str} & {mAP_str} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item 所有结果基于 5 折交叉验证
\item 数值格式：平均值 $\pm$ 标准差
\item Accuracy: 最佳验证准确率
\item mAP: 宏平均平均精度
\end{tablenotes}
\end{table}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"✓ LaTeX 表格已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='运行消融实验')
    parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output-dir', type=str, default='checkpoints/ablation_study',
                       help='输出目录（默认: checkpoints/ablation_study）')
    parser.add_argument('--gpu-id', type=int, default=0, help='单个GPU ID（默认: 0）')
    parser.add_argument('--gpus', type=str, default=None, 
                       help='GPU ID列表，用逗号分隔（例如: 7,8,9）。如果指定，会自动并行运行')
    parser.add_argument('--log-dir', type=str, default='logs/ablation_study',
                       help='日志目录（默认: logs/ablation_study）')
    parser.add_argument('--base-config', type=str, default='train_clip_config.json',
                       help='基础配置文件（默认: train_clip_config.json）')
    parser.add_argument('--skip-existing', action='store_true',
                       help='跳过已有结果的实验')
    
    args = parser.parse_args()
    
    # 解析GPU列表
    gpu_list = []
    if args.gpus:
        gpu_list = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    elif args.gpu_id is not None:
        gpu_list = [args.gpu_id]
    else:
        gpu_list = [0]
    
    # 创建输出和日志目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建消融实验配置
    ablation_configs = create_ablation_configs(args.base_config)
    
    # 过滤出需要运行的实验
    experiments_to_run = []
    for config_dict in ablation_configs:
        name = config_dict['name']
        output_dir = Path(args.output_dir) / f"ablation_{name}"
        if args.skip_existing and (output_dir / 'cv_summary.json').exists():
            continue
        experiments_to_run.append(config_dict)
    
    print(f"\n{'='*80}")
    print(f"消融实验")
    print(f"{'='*80}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"日志目录: {args.log_dir}")
    print(f"GPU 列表: {gpu_list}")
    print(f"实验总数: {len(ablation_configs)}")
    print(f"需要运行: {len(experiments_to_run)}")
    if len(experiments_to_run) < len(ablation_configs):
        print(f"跳过（已有结果）: {len(ablation_configs) - len(experiments_to_run)}")
    print(f"{'='*80}\n")
    
    # 存储所有结果
    all_results = {}
    total_start_time = time.time()
    
    # 如果只有一个GPU或只有一个实验，顺序运行
    if len(gpu_list) == 1 or len(experiments_to_run) <= 1:
        # 顺序运行
        run_count = 0
        for i, config_dict in enumerate(ablation_configs, 1):
            name = config_dict['name']
            print(f"\n[{i}/{len(ablation_configs)}] 开始实验: {config_dict['display_name']}")
            
            # 检查是否已存在结果
            output_dir = Path(args.output_dir) / f"ablation_{name}"
            if args.skip_existing and (output_dir / 'cv_summary.json').exists():
                print(f"  跳过（结果已存在）")
                results = extract_results_from_checkpoint(output_dir)
                all_results[name] = {
                    'display_name': config_dict['display_name'],
                    'description': config_dict['description'],
                    'results': results,
                    'output_dir': str(output_dir),
                    'status': 'skipped'
                }
                continue
            
            # 分配GPU
            gpu_id = gpu_list[run_count % len(gpu_list)]
            run_count += 1
            print(f"  使用 GPU: {gpu_id}")
            
            # 运行实验
            success, output_dir, elapsed_time = run_single_experiment(
                config_dict,
                args.data_dir,
                args.output_dir,
                gpu_id,
                args.log_dir
            )
            
            # 提取结果
            results = None
            if success:
                results = extract_results_from_checkpoint(output_dir)
            
            all_results[name] = {
                'display_name': config_dict['display_name'],
                'description': config_dict['description'],
                'results': results,
                'output_dir': str(output_dir),
                'status': 'success' if success else 'failed',
                'elapsed_time': elapsed_time,
                'gpu_id': gpu_id
            }
    else:
        # 并行运行
        print(f"使用 {len(gpu_list)} 个GPU并行运行 {len(experiments_to_run)} 个实验\n")
        
        task_queue = queue.Queue()
        for config_dict in experiments_to_run:
            task_queue.put(config_dict)
        
        results_lock = threading.Lock()
        
        def worker_thread(worker_id, gpu_id):
            while True:
                try:
                    config_dict = task_queue.get_nowait()
                except queue.Empty:
                    break
                
                name = config_dict['name']
                print(f"\n[GPU {gpu_id}] 开始实验: {config_dict['display_name']}")
                
                success, output_dir, elapsed_time = run_single_experiment(
                    config_dict,
                    args.data_dir,
                    args.output_dir,
                    gpu_id,
                    args.log_dir
                )
                
                results = None
                if success:
                    results = extract_results_from_checkpoint(output_dir)
                
                with results_lock:
                    all_results[name] = {
                        'display_name': config_dict['display_name'],
                        'description': config_dict['description'],
                        'results': results,
                        'output_dir': str(output_dir),
                        'status': 'success' if success else 'failed',
                        'elapsed_time': elapsed_time,
                        'gpu_id': gpu_id
                    }
                
                status_str = "✓ 完成" if success else "✗ 失败"
                print(f"[GPU {gpu_id}] {status_str}: {config_dict['display_name']}")
                task_queue.task_done()
        
        threads = []
        for worker_id, gpu_id in enumerate(gpu_list):
            thread = threading.Thread(target=worker_thread, args=(worker_id, gpu_id))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        # 处理跳过的实验
        for config_dict in ablation_configs:
            name = config_dict['name']
            if name not in all_results:
                output_dir = Path(args.output_dir) / f"ablation_{name}"
                if args.skip_existing and (output_dir / 'cv_summary.json').exists():
                    results = extract_results_from_checkpoint(output_dir)
                    all_results[name] = {
                        'display_name': config_dict['display_name'],
                        'description': config_dict['description'],
                        'results': results,
                        'output_dir': str(output_dir),
                        'status': 'skipped'
                    }
    
    # 生成消融实验表格
    print(f"\n{'='*80}")
    print("生成消融实验表格...")
    print(f"{'='*80}")
    
    table_file = generate_ablation_table(
        all_results,
        output_file=Path(args.output_dir) / "ablation_table.md"
    )
    
    # 保存详细结果 JSON
    json_file = Path(args.output_dir) / "ablation_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ 详细结果已保存到: {json_file}")
    
    # 打印总结
    total_time = time.time() - total_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'='*80}")
    print("实验总结")
    print(f"{'='*80}")
    print(f"总耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"\n实验结果:")
    order = ['clip_only', 'supcon_only', 'supcon_clip', 'supcon_clip_class']
    for name in order:
        if name not in all_results:
            continue
        info = all_results[name]
        status = info['status']
        if status == 'success':
            results = info.get('results')
            if results:
                print(f"  ✓ {info['display_name']}: Accuracy={results.get('accuracy', 0):.2f}%, mAP={results.get('mAP', 0):.2f}%")
            else:
                print(f"  ✓ {info['display_name']}: 完成（结果提取失败）")
        elif status == 'skipped':
            print(f"  ⊘ {info['display_name']}: 跳过")
        else:
            print(f"  ✗ {info['display_name']}: 失败")
    print(f"\n消融实验表格: {table_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

