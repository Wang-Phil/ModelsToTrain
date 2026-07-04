#!/usr/bin/env python3
"""
从已有实验结果生成对比表格
用于在实验完成后生成论文表格

使用方法:
    python generate_comparison_table.py --output-dir checkpoints/comparison_experiments
"""

import json
import argparse
from pathlib import Path

def extract_results_from_checkpoint(checkpoint_dir):
    """从checkpoint目录提取结果"""
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


def generate_comparison_table_from_results(results_dict, output_file="comparison_table.md"):
    """从结果字典生成对比表格"""
    output_path = Path(output_file)
    
    # 生成 Markdown 表格
    md_content = "# 主要方法对比实验结果\n\n"
    md_content += "## 表1：主要方法对比\n\n"
    md_content += "| Method | Accuracy (%) | mAP (%) | Precision (%) | Recall (%) | F1 (%) |\n"
    md_content += "|--------|--------------|---------|---------------|------------|--------|\n"
    
    for name, info in results_dict.items():
        display_name = info['display_name']
        results = info.get('results')
        
        if results is None:
            md_content += f"| {display_name} | - | - | - | - | - |\n"
        else:
            acc = results.get('accuracy', 0)
            acc_std = results.get('std_accuracy', 0)
            mAP = results.get('mAP', 0)
            mAP_std = results.get('std_mAP', 0)
            prec = results.get('precision', 0)
            prec_std = results.get('std_precision', 0)
            recall = results.get('recall', 0)
            recall_std = results.get('std_recall', 0)
            f1 = results.get('f1', 0)
            f1_std = results.get('std_f1', 0)
            
            acc_str = f"{acc:.2f} ± {acc_std:.2f}" if acc_std > 0 else f"{acc:.2f}"
            mAP_str = f"{mAP:.2f} ± {mAP_std:.2f}" if mAP_std > 0 else f"{mAP:.2f}"
            prec_str = f"{prec:.2f} ± {prec_std:.2f}" if prec_std > 0 else f"{prec:.2f}"
            recall_str = f"{recall:.2f} ± {recall_std:.2f}" if recall_std > 0 else f"{recall:.2f}"
            f1_str = f"{f1:.2f} ± {f1_std:.2f}" if f1_std > 0 else f"{f1:.2f}"
            
            md_content += f"| {display_name} | {acc_str} | {mAP_str} | {prec_str} | {recall_str} | {f1_str} |\n"
    
    md_content += "\n"
    md_content += "**说明：**\n"
    md_content += "- 所有结果基于 5 折交叉验证\n"
    md_content += "- 数值格式：平均值 ± 标准差\n"
    md_content += "- Accuracy: 最佳验证准确率\n"
    md_content += "- mAP: 宏平均平均精度\n"
    md_content += "- Precision/Recall/F1: 宏平均指标\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✓ 对比表格已保存到: {output_path}")
    
    # 生成 LaTeX 表格
    latex_file = output_path.with_suffix('.tex')
    generate_latex_table(results_dict, latex_file)
    
    return output_path


def generate_latex_table(results_dict, output_file):
    """生成 LaTeX 表格"""
    latex_content = r"""\begin{table}[htbp]
\centering
\caption{主要方法对比实验结果}
\label{tab:main_comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Accuracy (\%)} & \textbf{mAP (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1 (\%)} \\
\midrule
"""
    
    for name, info in results_dict.items():
        display_name = info['display_name']
        results = info.get('results')
        
        if results is None:
            latex_content += f"{display_name} & - & - & - & - & - \\\\\n"
        else:
            acc = results.get('accuracy', 0)
            acc_std = results.get('std_accuracy', 0)
            mAP = results.get('mAP', 0)
            mAP_std = results.get('std_mAP', 0)
            prec = results.get('precision', 0)
            prec_std = results.get('std_precision', 0)
            recall = results.get('recall', 0)
            recall_std = results.get('std_recall', 0)
            f1 = results.get('f1', 0)
            f1_std = results.get('std_f1', 0)
            
            acc_str = f"${acc:.2f} \\pm {acc_std:.2f}$" if acc_std > 0 else f"${acc:.2f}$"
            mAP_str = f"${mAP:.2f} \\pm {mAP_std:.2f}$" if mAP_std > 0 else f"${mAP:.2f}$"
            prec_str = f"${prec:.2f} \\pm {prec_std:.2f}$" if prec_std > 0 else f"${prec:.2f}$"
            recall_str = f"${recall:.2f} \\pm {recall_std:.2f}$" if recall_std > 0 else f"${recall:.2f}$"
            f1_str = f"${f1:.2f} \\pm {f1_std:.2f}$" if f1_std > 0 else f"${f1:.2f}$"
            
            if "Ours" in display_name or "supcon_clip_class" in name:
                latex_content += f"\\textbf{{{display_name}}} & {acc_str} & {mAP_str} & {prec_str} & {recall_str} & {f1_str} \\\\\n"
            else:
                latex_content += f"{display_name} & {acc_str} & {mAP_str} & {prec_str} & {recall_str} & {f1_str} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\item 所有结果基于 5 折交叉验证
\item 数值格式：平均值 $\pm$ 标准差
\item Accuracy: 最佳验证准确率
\item mAP: 宏平均平均精度
\item Precision/Recall/F1: 宏平均指标
\end{tablenotes}
\end{table}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"✓ LaTeX 表格已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='从已有实验结果生成对比表格')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='包含所有实验结果的目录')
    parser.add_argument('--output-file', type=str, default='comparison_table.md',
                       help='输出表格文件（默认: comparison_table.md）')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # 定义方法名称映射
    method_mapping = {
        'comparison_baseline_clip': {
            'display_name': 'CLIP (Baseline)',
            'description': '标准 CLIP 损失函数（图像-文本对比学习）'
        },
        'comparison_superclip': {
            'display_name': 'SuperCLIP',
            'description': 'SuperCLIP 损失函数（分类损失 + CLIP对比损失）'
        },
        'comparison_supcon_only': {
            'display_name': 'SupCon',
            'description': '有监督对比学习（仅 SupCon Loss）'
        },
        'comparison_supcon_clip': {
            'display_name': 'SupCon + CLIP',
            'description': 'Multi-Task Learning（SupCon + CLIP，无分类损失）'
        },
        'comparison_supcon_clip_class': {
            'display_name': 'SupCon + CLIP + Class (Ours)',
            'description': '完整 Multi-Task Learning（SupCon + CLIP + Classification Loss）'
        }
    }
    
    # 收集所有结果
    all_results = {}
    
    for exp_dir in sorted(output_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        if not exp_name.startswith('comparison_'):
            continue
        
        # 提取方法名称
        method_key = exp_name
        
        if method_key in method_mapping:
            info = method_mapping[method_key]
            results = extract_results_from_checkpoint(exp_dir)
            
            all_results[method_key] = {
                'display_name': info['display_name'],
                'description': info['description'],
                'results': results,
                'output_dir': str(exp_dir)
            }
            
            if results:
                print(f"✓ 已读取: {info['display_name']} - mAP={results.get('mAP', 0):.2f}%")
            else:
                print(f"⚠ 无法读取结果: {info['display_name']}")
    
    if not all_results:
        print(f"错误: 在 {output_dir} 中未找到任何实验结果")
        return
    
    # 按顺序排列（确保我们的方法在最后）
    ordered_results = {}
    order = [
        'comparison_baseline_clip',
        'comparison_superclip',
        'comparison_supcon_only',
        'comparison_supcon_clip',
        'comparison_supcon_clip_class'
    ]
    
    for key in order:
        if key in all_results:
            ordered_results[key] = all_results[key]
    
    # 生成表格
    print(f"\n生成对比表格...")
    generate_comparison_table_from_results(ordered_results, args.output_file)
    
    print(f"\n完成！")


if __name__ == "__main__":
    main()


