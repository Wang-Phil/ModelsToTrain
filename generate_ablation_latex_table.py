#!/usr/bin/env python3
"""
从 summary_report.txt 生成消融实验的 LaTeX 表格
包含 baseline 数据并重新命名模型
"""

import re
import os
import sys
import json
from pathlib import Path

def parse_summary_report(report_file):
    """解析 summary_report.txt 文件"""
    models_data = []
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取每个模型的数据
    pattern = r'模型: (\w+)\n.*?平均最佳验证mAP: ([\d.]+)% ± ([\d.]+)%\n.*?平均mAP: ([\d.]+)% ± ([\d.]+)%\n.*?平均Precision: ([\d.]+)% ± ([\d.]+)%\n.*?平均Recall: ([\d.]+)% ± ([\d.]+)%\n.*?平均F1 Score: ([\d.]+)% ± ([\d.]+)%\n.*?平均准确率: ([\d.]+)% ± ([\d.]+)%'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        model_name = match.group(1)
        best_map_mean = float(match.group(2))
        best_map_std = float(match.group(3))
        map_mean = float(match.group(4))
        map_std = float(match.group(5))
        prec_mean = float(match.group(6))
        prec_std = float(match.group(7))
        recall_mean = float(match.group(8))
        recall_std = float(match.group(9))
        f1_mean = float(match.group(10))
        f1_std = float(match.group(11))
        acc_mean = float(match.group(12))
        acc_std = float(match.group(13))
        
        models_data.append({
            'name': model_name,
            'mAP': (map_mean, map_std),
            'Precision': (prec_mean, prec_std),
            'Recall': (recall_mean, recall_std),
            'F1': (f1_mean, f1_std),
            'Accuracy': (acc_mean, acc_std),
        })
    
    return models_data

def get_baseline_data(baseline_path):
    """从 baseline 的 cv_summary.json 或日志中提取数据"""
    # 尝试从 JSON 文件读取
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
    
    # 如果 JSON 不存在，尝试从日志文件读取
    log_file = Path(baseline_path).parent.parent / 'logs' / 'final_starnet_models' / 'origin_starnet' / 'starnet_s1_gpu8.log'
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 从日志中提取数据
            map_match = re.search(r'平均最佳验证mAP: ([\d.]+)% ± ([\d.]+)%', content)
            prec_match = re.search(r'平均Precision: ([\d.]+)% ± ([\d.]+)%', content)
            recall_match = re.search(r'平均Recall: ([\d.]+)% ± ([\d.]+)%', content)
            f1_match = re.search(r'平均F1 Score: ([\d.]+)% ± ([\d.]+)%', content)
            acc_match = re.search(r'平均最佳验证准确率: ([\d.]+)% ± ([\d.]+)%', content)
            
            if all([map_match, prec_match, recall_match, f1_match, acc_match]):
                return {
                    'name': 'baseline',
                    'mAP': (float(map_match.group(1)), float(map_match.group(2))),
                    'Precision': (float(prec_match.group(1)), float(prec_match.group(2))),
                    'Recall': (float(recall_match.group(1)), float(recall_match.group(2))),
                    'F1': (float(f1_match.group(1)), float(f1_match.group(2))),
                    'Accuracy': (float(acc_match.group(1)), float(acc_match.group(2))),
                }
    
    return None

def get_model_display_name(model_name):
    """根据模型名称返回显示名称"""
    name_mapping = {
        'baseline': 'Baseline',
        'starnet_s1': 'Attn@All Stages',
        'starnet_s2': 'Attn@Stage 1+',
        'starnet_s3': 'Attn@Stage 2+',
        'starnet_s4': 'Attn@Stage 3',
    }
    return name_mapping.get(model_name, model_name.replace('_', ' ').title())

def generate_ablation_latex_table(models_data, baseline_data, output_file, title="Attention Ablation Study"):
    """生成消融实验的 LaTeX 表格代码"""
    
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
    
    # 找出每个指标列的最优值（用于加粗，排除baseline）
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
\label{tab:ablation}
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
        
        # 格式化模型名称
        name_str = display_name
        
        # 格式化各项指标
        map_str = f"\\textbf{{{map_val:.2f} $\\pm$ {map_std:.2f}}}" if is_best_map else f"{map_val:.2f} $\\pm$ {map_std:.2f}"
        prec_str = f"\\textbf{{{prec_val:.2f} $\\pm$ {prec_std:.2f}}}" if is_best_prec else f"{prec_val:.2f} $\\pm$ {prec_std:.2f}"
        recall_str = f"\\textbf{{{recall_val:.2f} $\\pm$ {recall_std:.2f}}}" if is_best_recall else f"{recall_val:.2f} $\\pm$ {recall_std:.2f}"
        f1_str = f"\\textbf{{{f1_val:.2f} $\\pm$ {f1_std:.2f}}}" if is_best_f1 else f"{f1_val:.2f} $\\pm$ {f1_std:.2f}"
        acc_str = f"\\textbf{{{acc_val:.2f} $\\pm$ {acc_std:.2f}}}" if is_best_acc else f"{acc_val:.2f} $\\pm$ {acc_std:.2f}"
        
        latex_code += f"{name_str} & {map_str} & {prec_str} & {recall_str} & {f1_str} & {acc_str} \\\\\n"
    
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
    \item Baseline: Original StarNet without spatial attention
    \item Attn@All Stages: Spatial attention applied to all stages (Stage 0, 1, 2, 3)
    \item Attn@Stage 1+: Spatial attention applied from Stage 1 onwards (Stage 1, 2, 3)
    \item Attn@Stage 2+: Spatial attention applied from Stage 2 onwards (Stage 2, 3)
    \item Attn@Stage 3: Spatial attention applied only to Stage 3
\end{itemize}

\end{document}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"✓ LaTeX 文件已生成: {output_file}")

def generate_latex_table_only(models_data, baseline_data, output_file, title="Attention Ablation Study"):
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
\label{tab:ablation}
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

def main():
    # 默认使用 attention_ablation 目录
    if len(sys.argv) > 1:
        report_file = sys.argv[1]
    else:
        report_file = 'checkpoints/final_starnet_models/attention_ablation/summary_report.txt'
    
    # Baseline 路径
    baseline_path = 'checkpoints/final_starnet_models/starnet_s1'
    
    # 检查报告文件是否存在
    if not os.path.exists(report_file):
        print(f"错误: 找不到报告文件: {report_file}")
        print(f"用法: {sys.argv[0]} [report_file_path]")
        return
    
    # 确定输出目录和文件名
    report_path = Path(report_file)
    output_dir = report_path.parent
    tex_file_full = output_dir / 'ablation_table.tex'
    tex_file_only = output_dir / 'ablation_table_only.tex'
    
    # 解析报告
    print(f"正在解析报告文件: {report_file}")
    models_data = parse_summary_report(report_file)
    
    if not models_data:
        print("错误: 未能从报告中提取数据")
        print("请检查报告文件格式是否正确")
        return
    
    print(f"✓ 成功解析 {len(models_data)} 个模型的数据")
    for model in models_data:
        print(f"  - {model['name']}")
    
    # 获取 baseline 数据
    print(f"\n正在获取 baseline 数据: {baseline_path}")
    baseline_data = get_baseline_data(baseline_path)
    
    if baseline_data:
        print(f"✓ 成功获取 baseline 数据")
        print(f"  - mAP: {baseline_data['mAP'][0]:.2f}% ± {baseline_data['mAP'][1]:.2f}%")
    else:
        print("⚠ 警告: 未能获取 baseline 数据，将不包含在表格中")
        baseline_data = None
    
    # 生成完整的 LaTeX 文档
    print(f"\n正在生成完整 LaTeX 文档...")
    generate_ablation_latex_table(models_data, baseline_data, tex_file_full, title="Attention Ablation Study")
    
    # 生成仅表格代码（用于插入到现有文档）
    print(f"正在生成表格代码（用于插入）...")
    generate_latex_table_only(models_data, baseline_data, tex_file_only, title="Attention Ablation Study")
    
    print(f"\n✓ 完成！")
    print(f"  - 完整文档: {tex_file_full}")
    print(f"  - 仅表格代码: {tex_file_only}")

if __name__ == '__main__':
    main()

