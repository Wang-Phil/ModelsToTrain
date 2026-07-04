#!/usr/bin/env python3
"""
从已有实验结果生成消融实验表格
用于在实验完成后生成论文表格

使用方法:
    python generate_ablation_table.py --output-dir checkpoints/ablation_study
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


def generate_ablation_table_from_results(results_dict, output_file="ablation_table.md"):
    """从结果字典生成消融实验表格"""
    output_path = Path(output_file)
    
    # 生成 Markdown 表格
    md_content = "# 消融实验结果\n\n"
    md_content += "## 表：消融实验对比\n\n"
    md_content += "| Components | Accuracy (%) | mAP (%) |\n"
    md_content += "|------------|--------------|----------|\n"
    
    # 按顺序排列
    order = ['clip_only', 'supcon_only', 'supcon_clip', 'supcon_clip_class']
    method_mapping = {
        'ablation_clip_only': {'key': 'clip_only', 'display_name': 'CLIP Loss only'},
        'ablation_supcon_only': {'key': 'supcon_only', 'display_name': 'SupCon Loss only'},
        'ablation_supcon_clip': {'key': 'supcon_clip', 'display_name': 'SupCon + CLIP'},
        'ablation_supcon_clip_class': {'key': 'supcon_clip_class', 'display_name': 'SupCon + CLIP + Class Loss'}
    }
    
    # 重新组织结果字典
    reorganized_results = {}
    for exp_dir_name, info in method_mapping.items():
        key = info['key']
        if key in results_dict:
            reorganized_results[key] = results_dict[key]
        else:
            # 尝试从目录名匹配
            for name, data in results_dict.items():
                if name == exp_dir_name or name == key:
                    reorganized_results[key] = data
                    break
    
    for name in order:
        if name not in reorganized_results:
            continue
            
        info = reorganized_results[name]
        if isinstance(info, dict):
            if 'display_name' in info:
                display_name = info['display_name']
                results = info.get('results')
            else:
                # 如果直接是结果字典
                display_name = method_mapping.get(f'ablation_{name}', {}).get('display_name', name)
                results = info
        else:
            display_name = method_mapping.get(f'ablation_{name}', {}).get('display_name', name)
            results = info
        
        if results is None or not isinstance(results, dict) or 'accuracy' not in results:
            md_content += f"| {display_name} | - | - |\n"
        else:
            acc = results.get('accuracy', 0)
            acc_std = results.get('std_accuracy', 0)
            mAP = results.get('mAP', 0)
            mAP_std = results.get('std_mAP', 0)
            
            acc_str = f"{acc:.2f} ± {acc_std:.2f}" if acc_std > 0 else f"{acc:.2f}"
            mAP_str = f"{mAP:.2f} ± {mAP_std:.2f}" if mAP_std > 0 else f"{mAP:.2f}"
            
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
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✓ 消融实验表格已保存到: {output_path}")
    
    # 生成 LaTeX 表格
    latex_file = output_path.with_suffix('.tex')
    generate_latex_ablation_table_from_results(reorganized_results, latex_file, method_mapping)
    
    return output_path


def generate_latex_ablation_table_from_results(results_dict, output_file, method_mapping):
    """生成 LaTeX 表格"""
    latex_content = r"""\begin{table}[htbp]
\centering
\caption{消融实验结果}
\label{tab:ablation_study}
\begin{tabular}{lcc}
\toprule
\textbf{Components} & \textbf{Accuracy (\%)} & \textbf{mAP (\%)} \\
\midrule
"""
    
    order = ['clip_only', 'supcon_only', 'supcon_clip', 'supcon_clip_class']
    
    for name in order:
        if name not in results_dict:
            continue
            
        info = results_dict[name]
        if isinstance(info, dict):
            if 'display_name' in info:
                display_name = info['display_name']
                results = info.get('results')
            else:
                display_name = method_mapping.get(f'ablation_{name}', {}).get('display_name', name)
                results = info
        else:
            display_name = method_mapping.get(f'ablation_{name}', {}).get('display_name', name)
            results = info
        
        if results is None or not isinstance(results, dict) or 'accuracy' not in results:
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
    parser = argparse.ArgumentParser(description='从已有实验结果生成消融实验表格')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='包含所有实验结果的目录')
    parser.add_argument('--output-file', type=str, default='ablation_table.md',
                       help='输出表格文件（默认: ablation_table.md）')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # 定义方法名称映射
    method_mapping = {
        'ablation_clip_only': {
            'key': 'clip_only',
            'display_name': 'CLIP Loss only',
            'description': '仅使用CLIP损失（图像-文本对比学习）'
        },
        'ablation_supcon_only': {
            'key': 'supcon_only',
            'display_name': 'SupCon Loss only',
            'description': '仅使用SupCon损失（有监督对比学习）'
        },
        'ablation_supcon_clip': {
            'key': 'supcon_clip',
            'display_name': 'SupCon + CLIP',
            'description': 'SupCon损失 + CLIP损失（无分类损失）'
        },
        'ablation_supcon_clip_class': {
            'key': 'supcon_clip_class',
            'display_name': 'SupCon + CLIP + Class Loss',
            'description': '完整方法：SupCon损失 + CLIP损失 + 分类损失'
        }
    }
    
    # 收集所有结果
    all_results = {}
    
    for exp_dir in sorted(output_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        if not exp_name.startswith('ablation_'):
            continue
        
        if exp_name in method_mapping:
            info = method_mapping[exp_name]
            key = info['key']
            results = extract_results_from_checkpoint(exp_dir)
            
            all_results[key] = {
                'display_name': info['display_name'],
                'description': info['description'],
                'results': results,
                'output_dir': str(exp_dir)
            }
            
            if results:
                print(f"✓ 已读取: {info['display_name']} - Accuracy={results.get('accuracy', 0):.2f}%, mAP={results.get('mAP', 0):.2f}%")
            else:
                print(f"⚠ 无法读取结果: {info['display_name']}")
    
    if not all_results:
        print(f"错误: 在 {output_dir} 中未找到任何实验结果")
        return
    
    # 生成表格
    print(f"\n生成消融实验表格...")
    generate_ablation_table_from_results(all_results, args.output_file)
    
    print(f"\n完成！")


if __name__ == "__main__":
    main()


