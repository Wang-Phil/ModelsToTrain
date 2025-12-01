#!/usr/bin/env python3
"""
生成 Cross-Star 消融实验的 LaTeX 表格
包含 baseline 和三个 Cross-Star 变体
"""

import json
import os
import sys
from pathlib import Path

def get_model_data(model_path):
    """从 cv_summary.json 读取模型数据"""
    json_file = Path(model_path) / 'cv_summary.json'
    if not json_file.exists():
        return None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {
            'name': data.get('model', 'unknown'),
            'mAP': (data.get('average_mAP', 0), data.get('std_mAP', 0)),
            'Precision': (data.get('average_precision', 0), data.get('std_precision', 0)),
            'Recall': (data.get('average_recall', 0), data.get('std_recall', 0)),
            'F1': (data.get('average_f1', 0), data.get('std_f1', 0)),
            'Accuracy': (data.get('average_best_val_acc', 0), data.get('std_best_val_acc', 0)),
        }

def get_baseline_data(baseline_path):
    """从 baseline 的 cv_summary.json 读取数据"""
    json_file = Path(baseline_path) / 'cv_summary.json'
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {
                'name': 'baseline',
                'mAP': (data.get('average_mAP', 0), data.get('std_mAP', 0)),
                'Precision': (data.get('average_precision', 0), data.get('std_precision', 0)),
                'Recall': (data.get('average_recall', 0), data.get('std_recall', 0)),
                'F1': (data.get('average_f1', 0), data.get('std_f1', 0)),
                'Accuracy': (data.get('average_best_val_acc', 0), data.get('std_best_val_acc', 0)),
            }
    return None

def get_model_display_name(model_name):
    """根据模型名称返回显示名称"""
    name_mapping = {
        'baseline': 'Baseline',
        'starnet_s1_cross_star': 'Cross-Star (D)',
        'starnet_s1_cross_star_add': 'Cross-Star Add (D1)',
        'starnet_s1_cross_star_samescale': 'Cross-Star SameScale (D2)',
    }
    return name_mapping.get(model_name, model_name.replace('_', ' ').title())

def generate_latex_table_only(models_data, baseline_data, output_file, title="Cross-Star Ablation Study"):
    """仅生成表格代码（不包含完整文档），用于插入到现有LaTeX文档中"""
    
    # 合并 baseline 和模型数据
    all_data = []
    if baseline_data:
        all_data.append(baseline_data)
    all_data.extend(models_data)
    
    # 排序：baseline 保持在最上面，其他模型按 mAP 从低到高排序（最好的在最后）
    baseline_list = [m for m in all_data if m['name'] == 'baseline']
    other_models = [m for m in all_data if m['name'] != 'baseline']
    other_models_sorted = sorted(other_models, key=lambda x: x['mAP'][0])  # 按 mAP 从低到高排序
    all_data_sorted = baseline_list + other_models_sorted
    
    # 找出每个指标列的最优值（排除baseline）
    non_baseline = [m for m in all_data_sorted if m['name'] != 'baseline']
    max_map = max(m['mAP'][0] for m in non_baseline)
    max_prec = max(m['Precision'][0] for m in non_baseline)
    max_recall = max(m['Recall'][0] for m in non_baseline)
    max_f1 = max(m['F1'][0] for m in non_baseline)
    max_acc = max(m['Accuracy'][0] for m in non_baseline)
    
    latex_code = r"""\begin{table}[h]
\centering
\caption{""" + title + r""" (Mean $\pm$ Std over 5 folds)}
\label{tab:cross_star_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{mAP (\%)} $\uparrow$ & \textbf{Precision (\%)} $\uparrow$ & \textbf{Recall (\%)} $\uparrow$ & \textbf{F1 (\%)} $\uparrow$ & \textbf{Accuracy (\%)} $\uparrow$ \\
\midrule
"""
    
    for model in all_data_sorted:
        name = model['name']
        display_name = get_model_display_name(name)
        map_val, map_std = model['mAP']
        prec_val, prec_std = model['Precision']
        recall_val, recall_std = model['Recall']
        f1_val, f1_std = model['F1']
        acc_val, acc_std = model['Accuracy']
        
        # Baseline 不加粗，其他模型的最优值加粗
        is_baseline = (name == 'baseline')
        is_best_map = not is_baseline and (abs(map_val - max_map) < 0.01)
        is_best_prec = not is_baseline and (abs(prec_val - max_prec) < 0.01)
        is_best_recall = not is_baseline and (abs(recall_val - max_recall) < 0.01)
        is_best_f1 = not is_baseline and (abs(f1_val - max_f1) < 0.01)
        is_best_acc = not is_baseline and (abs(acc_val - max_acc) < 0.01)
        
        # 格式化各项指标
        map_str = f"\\textbf{{{map_val:.2f} $\\pm$ {map_std:.2f}}}" if is_best_map else f"{map_val:.2f} $\\pm$ {map_std:.2f}"
        prec_str = f"\\textbf{{{prec_val:.2f} $\\pm$ {prec_std:.2f}}}" if is_best_prec else f"{prec_val:.2f} $\\pm$ {prec_std:.2f}"
        recall_str = f"\\textbf{{{recall_val:.2f} $\\pm$ {recall_std:.2f}}}" if is_best_recall else f"{recall_val:.2f} $\\pm$ {recall_std:.2f}"
        f1_str = f"\\textbf{{{f1_val:.2f} $\\pm$ {f1_std:.2f}}}" if is_best_f1 else f"{f1_val:.2f} $\\pm$ {f1_std:.2f}"
        acc_str = f"\\textbf{{{acc_val:.2f} $\\pm$ {acc_std:.2f}}}" if is_best_acc else f"{acc_val:.2f} $\\pm$ {acc_std:.2f}"
        
        latex_code += f"{display_name} & {map_str} & {prec_str} & {recall_str} & {f1_str} & {acc_str} \\\\\n"
    
    latex_code += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"✓ LaTeX 表格代码已生成: {output_file}")

def generate_ablation_latex_table(models_data, baseline_data, output_file, title="Cross-Star Ablation Study"):
    """生成完整的 LaTeX 文档"""
    
    # 合并 baseline 和模型数据
    all_data = []
    if baseline_data:
        all_data.append(baseline_data)
    all_data.extend(models_data)
    
    # 排序：baseline 保持在最上面，其他模型按 mAP 从低到高排序（最好的在最后）
    baseline_list = [m for m in all_data if m['name'] == 'baseline']
    other_models = [m for m in all_data if m['name'] != 'baseline']
    other_models_sorted = sorted(other_models, key=lambda x: x['mAP'][0])  # 按 mAP 从低到高排序
    all_data_sorted = baseline_list + other_models_sorted
    
    # 找出每个指标列的最优值（排除baseline）
    non_baseline = [m for m in all_data_sorted if m['name'] != 'baseline']
    max_map = max(m['mAP'][0] for m in non_baseline)
    max_prec = max(m['Precision'][0] for m in non_baseline)
    max_recall = max(m['Recall'][0] for m in non_baseline)
    max_f1 = max(m['F1'][0] for m in non_baseline)
    max_acc = max(m['Accuracy'][0] for m in non_baseline)
    
    latex_code = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{geometry}
\geometry{margin=1.5cm}
\usepackage{xcolor}

\title{""" + title + r"""}
\author{5-Fold Cross-Validation Results}
\date{\today}

\begin{document}

\maketitle

\begin{table}[h]
\centering
\caption{""" + title + r""" (Mean $\pm$ Std over 5 folds)}
\label{tab:cross_star_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{mAP (\%)} $\uparrow$ & \textbf{Precision (\%)} $\uparrow$ & \textbf{Recall (\%)} $\uparrow$ & \textbf{F1 (\%)} $\uparrow$ & \textbf{Accuracy (\%)} $\uparrow$ \\
\midrule
"""
    
    for model in all_data_sorted:
        name = model['name']
        display_name = get_model_display_name(name)
        map_val, map_std = model['mAP']
        prec_val, prec_std = model['Precision']
        recall_val, recall_std = model['Recall']
        f1_val, f1_std = model['F1']
        acc_val, acc_std = model['Accuracy']
        
        # Baseline 不加粗，其他模型的最优值加粗
        is_baseline = (name == 'baseline')
        is_best_map = not is_baseline and (abs(map_val - max_map) < 0.01)
        is_best_prec = not is_baseline and (abs(prec_val - max_prec) < 0.01)
        is_best_recall = not is_baseline and (abs(recall_val - max_recall) < 0.01)
        is_best_f1 = not is_baseline and (abs(f1_val - max_f1) < 0.01)
        is_best_acc = not is_baseline and (abs(acc_val - max_acc) < 0.01)
        
        # 格式化各项指标
        map_str = f"\\textbf{{{map_val:.2f} $\\pm$ {map_std:.2f}}}" if is_best_map else f"{map_val:.2f} $\\pm$ {map_std:.2f}"
        prec_str = f"\\textbf{{{prec_val:.2f} $\\pm$ {prec_std:.2f}}}" if is_best_prec else f"{prec_val:.2f} $\\pm$ {prec_std:.2f}"
        recall_str = f"\\textbf{{{recall_val:.2f} $\\pm$ {recall_std:.2f}}}" if is_best_recall else f"{recall_val:.2f} $\\pm$ {recall_std:.2f}"
        f1_str = f"\\textbf{{{f1_val:.2f} $\\pm$ {f1_std:.2f}}}" if is_best_f1 else f"{f1_val:.2f} $\\pm$ {f1_std:.2f}"
        acc_str = f"\\textbf{{{acc_val:.2f} $\\pm$ {acc_std:.2f}}}" if is_best_acc else f"{acc_val:.2f} $\\pm$ {acc_std:.2f}"
        
        latex_code += f"{display_name} & {map_str} & {prec_str} & {recall_str} & {f1_str} & {acc_str} \\\\\n"
    
    latex_code += r"""\bottomrule
\end{tabular}%
}
\end{table}

\vspace{0.5cm}

\textbf{Notes:}
\begin{itemize}
    \item All metrics are reported as mean $\pm$ standard deviation over 5-fold cross-validation
    \item \textbf{Bold} values indicate the best performance for each metric (excluding baseline)
    \item mAP: Mean Average Precision
    \item Baseline: Original StarNet without Cross-Star operation
    \item Cross-Star (D): Baseline Cross-Star with cross multiplication: Concat((x\_3A * x\_7B), (x\_7A * x\_3B))
    \item Cross-Star Add (D1): Cross-Star with addition: Concat((x\_3A + x\_7B), (x\_7A + x\_3B))
    \item Cross-Star SameScale (D2): Cross-Star with same-scale multiplication: Concat((x\_3A * x\_3B), (x\_7A * x\_7B))
\end{itemize}

\end{document}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"✓ LaTeX 文件已生成: {output_file}")

def main():
    # 模型路径
    base_dir = Path('checkpoints/final_starnet_models')
    baseline_path = base_dir / 'starnet_s1'
    
    models = [
        ('starnet_s1_cross_star', base_dir / 'starnet_s1_cross_star'),
        ('starnet_s1_cross_star_add', base_dir / 'starnet_s1_cross_star_add'),
        ('starnet_s1_cross_star_samescale', base_dir / 'starnet_s1_cross_star_samescale'),
    ]
    
    # 输出目录
    output_dir = base_dir / 'cross_star_ablation'
    output_dir.mkdir(exist_ok=True)
    
    # 获取 baseline 数据
    print(f"正在获取 baseline 数据: {baseline_path}")
    baseline_data = get_baseline_data(baseline_path)
    
    if baseline_data:
        print(f"✓ 成功获取 baseline 数据")
        print(f"  - mAP: {baseline_data['mAP'][0]:.2f}% ± {baseline_data['mAP'][1]:.2f}%")
    else:
        print("⚠ 警告: 未能获取 baseline 数据，将不包含在表格中")
        baseline_data = None
    
    # 获取模型数据
    models_data = []
    print(f"\n正在读取模型数据...")
    for model_name, model_path in models:
        print(f"  读取: {model_name}")
        data = get_model_data(model_path)
        if data:
            models_data.append(data)
            print(f"    ✓ mAP: {data['mAP'][0]:.2f}% ± {data['mAP'][1]:.2f}%")
        else:
            print(f"    ✗ 无法读取数据")
    
    if not models_data:
        print("错误: 未能读取任何模型数据")
        return
    
    print(f"\n✓ 成功读取 {len(models_data)} 个模型的数据")
    
    # 生成表格
    tex_file_full = output_dir / 'cross_star_ablation_table.tex'
    tex_file_only = output_dir / 'cross_star_ablation_table_only.tex'
    
    print(f"\n正在生成完整 LaTeX 文档...")
    generate_ablation_latex_table(models_data, baseline_data, tex_file_full, title="Cross-Star Ablation Study")
    
    print(f"正在生成表格代码（用于插入）...")
    generate_latex_table_only(models_data, baseline_data, tex_file_only, title="Cross-Star Ablation Study")
    
    print(f"\n✓ 完成！")
    print(f"  - 完整文档: {tex_file_full}")
    print(f"  - 仅表格代码: {tex_file_only}")

if __name__ == '__main__':
    main()


