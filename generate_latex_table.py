#!/usr/bin/env python3
"""
从 summary_report.txt 生成 LaTeX 表格并导出为 PDF
"""

import re
import os
import subprocess
from pathlib import Path

def parse_summary_report(report_file):
    """解析 summary_report.txt 文件"""
    models_data = []
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取每个模型的数据
    pattern = r'模型: (\w+)\n.*?平均mAP: ([\d.]+)% ± ([\d.]+)%\n.*?平均Precision: ([\d.]+)% ± ([\d.]+)%\n.*?平均Recall: ([\d.]+)% ± ([\d.]+)%\n.*?平均F1 Score: ([\d.]+)% ± ([\d.]+)%\n.*?平均准确率: ([\d.]+)% ± ([\d.]+)%\n.*?参数量: ([\d.]+)M\n.*?FLOPs: ([\d.]+)M'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        model_name = match.group(1)
        map_mean = float(match.group(2))
        map_std = float(match.group(3))
        prec_mean = float(match.group(4))
        prec_std = float(match.group(5))
        recall_mean = float(match.group(6))
        recall_std = float(match.group(7))
        f1_mean = float(match.group(8))
        f1_std = float(match.group(9))
        acc_mean = float(match.group(10))
        acc_std = float(match.group(11))
        params = float(match.group(12))
        flops = float(match.group(13))
        
        models_data.append({
            'name': model_name,
            'mAP': (map_mean, map_std),
            'Precision': (prec_mean, prec_std),
            'Recall': (recall_mean, recall_std),
            'F1': (f1_mean, f1_std),
            'Accuracy': (acc_mean, acc_std),
            'Params': params,
            'FLOPs': flops
        })
    
    return models_data

def format_model_name(name):
    """格式化模型名称，使其在 LaTeX 中更美观"""
    # 将下划线替换为空格，并处理特殊名称
    name = name.replace('_', ' ')
    # 首字母大写
    name = name.title()
    # 特殊处理
    name = name.replace('Starnet', 'StarNet')
    name = name.replace('Mobilenet', 'MobileNet')
    name = name.replace('Resnet', 'ResNet')
    name = name.replace('Densenet', 'DenseNet')
    name = name.replace('Googlenet', 'GoogleNet')
    name = name.replace('Inceptionv3', 'InceptionV3')
    return name

def generate_latex_table(models_data, output_file):
    """生成 LaTeX 表格代码"""
    
    latex_code = r"""\documentclass[10pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{geometry}
\geometry{margin=1cm}
\usepackage{longtable}

\title{模型性能对比表}
\author{五折交叉验证结果}
\date{\today}

\begin{document}

\maketitle

\begin{longtable}{lccccccc}
\toprule
\textbf{Model} & \textbf{mAP (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1 (\%)} & \textbf{Accuracy (\%)} & \textbf{Params (M)} & \textbf{FLOPs (M)} \\
\midrule
\endfirsthead

\multicolumn{8}{c}%
{{\bfseries \tablename\ \thetable{} -- 续上页}} \\
\toprule
\textbf{Model} & \textbf{mAP (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1 (\%)} & \textbf{Accuracy (\%)} & \textbf{Params (M)} & \textbf{FLOPs (M)} \\
\midrule
\endhead

\bottomrule
\endfoot

\bottomrule
\endlastfoot
"""
    
    # 按 Precision 降序排序
    models_data_sorted = sorted(models_data, key=lambda x: x['Precision'][0], reverse=True)
    
    # 找出每个指标列的最大值（用于加粗最优值）
    max_map = max(m['mAP'][0] for m in models_data_sorted)
    max_prec = max(m['Precision'][0] for m in models_data_sorted)
    max_recall = max(m['Recall'][0] for m in models_data_sorted)
    max_f1 = max(m['F1'][0] for m in models_data_sorted)
    max_acc = max(m['Accuracy'][0] for m in models_data_sorted)
    # 对于参数量和FLOPs，最小值是最优的（越小越好）
    min_params = min(m['Params'] for m in models_data_sorted)
    min_flops = min(m['FLOPs'] for m in models_data_sorted)
    
    for i, model in enumerate(models_data_sorted):
        name = format_model_name(model['name'])
        map_val, map_std = model['mAP']
        prec_val, prec_std = model['Precision']
        recall_val, recall_std = model['Recall']
        f1_val, f1_std = model['F1']
        acc_val, acc_std = model['Accuracy']
        params = model['Params']
        flops = model['FLOPs']
        
        # 判断是否为最优模型（第一行）
        is_best_model = (i == 0)
        
        # 判断每个指标是否为最优值
        is_best_map = (abs(map_val - max_map) < 0.01)  # 考虑浮点误差
        is_best_prec = (abs(prec_val - max_prec) < 0.01)
        is_best_recall = (abs(recall_val - max_recall) < 0.01)
        is_best_f1 = (abs(f1_val - max_f1) < 0.01)
        is_best_acc = (abs(acc_val - max_acc) < 0.01)
        is_best_params = (abs(params - min_params) < 0.01)
        is_best_flops = (abs(flops - min_flops) < 0.01)
        
        # 模型名称：最优模型加粗
        if is_best_model:
            name_str = f"\\bfseries {name} \\normalfont"
        else:
            name_str = name
        
        # mAP：最优值加粗
        if is_best_map:
            map_str = f"\\bfseries {map_val:.2f} $\\pm$ {map_std:.2f} \\normalfont"
        else:
            map_str = f"{map_val:.2f} $\\pm$ {map_std:.2f}"
        
        # Precision：最优值加粗
        if is_best_prec:
            prec_str = f"\\bfseries {prec_val:.2f} $\\pm$ {prec_std:.2f} \\normalfont"
        else:
            prec_str = f"{prec_val:.2f} $\\pm$ {prec_std:.2f}"
        
        # Recall：最优值加粗
        if is_best_recall:
            recall_str = f"\\bfseries {recall_val:.2f} $\\pm$ {recall_std:.2f} \\normalfont"
        else:
            recall_str = f"{recall_val:.2f} $\\pm$ {recall_std:.2f}"
        
        # F1：最优值加粗
        if is_best_f1:
            f1_str = f"\\bfseries {f1_val:.2f} $\\pm$ {f1_std:.2f} \\normalfont"
        else:
            f1_str = f"{f1_val:.2f} $\\pm$ {f1_std:.2f}"
        
        # Accuracy：最优值加粗
        if is_best_acc:
            acc_str = f"\\bfseries {acc_val:.2f} $\\pm$ {acc_std:.2f} \\normalfont"
        else:
            acc_str = f"{acc_val:.2f} $\\pm$ {acc_std:.2f}"
        
        # Params：最小值加粗（越小越好）
        if is_best_params:
            params_str = f"\\bfseries {params:.2f} \\normalfont"
        else:
            params_str = f"{params:.2f}"
        
        # FLOPs：最小值加粗（越小越好）
        if is_best_flops:
            flops_str = f"\\bfseries {flops:.2f} \\normalfont"
        else:
            flops_str = f"{flops:.2f}"
        
        # 组合成一行
        latex_code += f"{name_str} & {map_str} & {prec_str} & {recall_str} & {f1_str} & {acc_str} & {params_str} & {flops_str} \\\\\n"
    
    latex_code += r"""
\end{longtable}

\vspace{1cm}

\textbf{说明:}
\begin{itemize}
    \item 所有指标均为五折交叉验证的平均值 $\pm$ 标准差
    \item mAP: 宏平均精确率 (Macro Average Precision)
    \item 加粗表示：模型名称（精确度最高的模型）和各指标的最优值
    \item 对于参数量和FLOPs，最小值（越小越好）为最优值
    \item 模型按精确度 (Precision) 降序排列
\end{itemize}

\end{document}
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"LaTeX 文件已生成: {output_file}")

def compile_latex_to_pdf(tex_file):
    """编译 LaTeX 文件为 PDF"""
    tex_path = Path(tex_file)
    tex_dir = tex_path.parent
    tex_name = tex_path.stem
    
    # 切换到 LaTeX 文件所在目录
    original_dir = os.getcwd()
    os.chdir(tex_dir)
    
    try:
        # 运行 pdflatex（运行两次以确保交叉引用正确）
        print(f"正在编译 LaTeX 文件: {tex_file}")
        for i in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_name + '.tex'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"警告: pdflatex 第 {i+1} 次运行返回错误码 {result.returncode}")
                print("错误输出:")
                print(result.stderr)
        
        pdf_file = tex_dir / (tex_name + '.pdf')
        if pdf_file.exists():
            print(f"✓ PDF 文件已生成: {pdf_file}")
            return str(pdf_file)
        else:
            print("✗ PDF 文件未生成")
            return None
    except FileNotFoundError:
        print("✗ 错误: 未找到 pdflatex 命令")
        print("   请确保已安装 LaTeX 发行版（如 TeX Live 或 MiKTeX）")
        return None
    finally:
        os.chdir(original_dir)
        # 清理辅助文件
        for ext in ['.aux', '.log', '.out']:
            aux_file = tex_dir / (tex_name + ext)
            if aux_file.exists():
                try:
                    aux_file.unlink()
                except:
                    pass

def main():
    # 文件路径
    report_file = 'checkpoints/cv_multi_models/summary_report.txt'
    output_dir = Path('checkpoints/cv_multi_models')
    tex_file = output_dir / 'model_comparison_table.tex'
    pdf_file = output_dir / 'model_comparison_table.pdf'
    
    # 检查报告文件是否存在
    if not os.path.exists(report_file):
        print(f"错误: 找不到报告文件: {report_file}")
        return
    
    # 解析报告
    print("正在解析报告文件...")
    models_data = parse_summary_report(report_file)
    
    if not models_data:
        print("错误: 未能从报告中提取数据")
        return
    
    print(f"成功解析 {len(models_data)} 个模型的数据")
    
    # 生成 LaTeX 表格
    print("正在生成 LaTeX 表格...")
    generate_latex_table(models_data, tex_file)
    
    # 编译为 PDF
    print("\n正在编译为 PDF...")
    pdf_path = compile_latex_to_pdf(tex_file)
    
    if pdf_path:
        print(f"\n✓ 完成！PDF 文件已保存到: {pdf_path}")
    else:
        print(f"\n⚠ LaTeX 文件已生成: {tex_file}")
        print("   您可以手动使用 pdflatex 编译该文件")

if __name__ == '__main__':
    main()

