#!/usr/bin/env python3
"""
汇总所有实验结果的脚本
从output目录下的各个子目录中读取cv_summary.json文件，生成Excel汇总表
"""

import json
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

def read_cv_summary(summary_path: Path) -> Optional[Dict]:
    """读取cv_summary.json文件"""
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"警告: 无法读取 {summary_path}: {e}")
        return None

def extract_metrics(summary_data: Dict, model_name: str) -> Dict:
    """从summary数据中提取所需的指标"""
    # 尝试两种格式：新的格式（average_results）和旧的格式（直接在根级别）
    avg_results = summary_data.get('average_results', {})
    
    # 如果是旧格式（字段在根级别），直接从summary_data读取
    if not avg_results:
        # 旧格式：使用 average_ 前缀
        avg_best_val_mAP = summary_data.get('average_best_val_mAP') or summary_data.get('avg_best_val_mAP', 0)
        avg_best_val_acc = summary_data.get('average_best_val_acc') or summary_data.get('avg_best_val_acc', 0)
        avg_val_mAP = summary_data.get('average_val_mAP') or summary_data.get('avg_val_mAP', 0)
        avg_val_acc = summary_data.get('average_val_acc') or summary_data.get('avg_val_acc', 0)
        avg_val_loss = summary_data.get('average_val_loss') or summary_data.get('avg_val_loss', 0)
        avg_best_precision = avg_best_val_mAP  # 旧格式可能没有单独的precision
        avg_best_recall = summary_data.get('average_best_recall') or summary_data.get('avg_best_recall', 0)
        avg_best_f1 = summary_data.get('average_best_f1') or summary_data.get('avg_best_f1', 0)
    else:
        # 新格式：在 average_results 中
        avg_best_val_mAP = avg_results.get('avg_best_val_mAP', 0)
        avg_best_val_acc = avg_results.get('avg_best_val_acc', 0)
        avg_val_mAP = avg_results.get('avg_val_mAP', 0)
        avg_val_acc = avg_results.get('avg_val_acc', 0)
        avg_val_loss = avg_results.get('avg_val_loss', 0)
        avg_best_precision = avg_results.get('avg_best_precision')
        if avg_best_precision is None:
            avg_best_precision = avg_best_val_mAP  # 如果没有precision，使用mAP
        avg_best_recall = avg_results.get('avg_best_recall', 0)
        avg_best_f1 = avg_results.get('avg_best_f1', 0)
    
    return {
        '模型': model_name,
        '平均最佳验证 mAP': round(avg_best_val_mAP, 4) if avg_best_val_mAP is not None else 0,
        '平均最佳验证准确率': round(avg_best_val_acc, 4) if avg_best_val_acc is not None else 0,
        '平均 mAP': round(avg_val_mAP, 4) if avg_val_mAP is not None else 0,
        '平均 Precision': round(avg_best_precision, 4) if avg_best_precision is not None else 0,
        '平均 Recall': round(avg_best_recall, 4) if avg_best_recall is not None else 0,
        '平均 F1 Score': round(avg_best_f1, 4) if avg_best_f1 is not None else 0,
        '平均最终验证准确率': round(avg_val_acc, 4) if avg_val_acc is not None else 0,
        '平均最终验证损失': round(avg_val_loss, 6) if avg_val_loss is not None else 0,
    }

def main():
    # 输出目录
    output_dir = Path('/home/ln/wangweicheng/ModelsTotrain/output')
    
    if not output_dir.exists():
        print(f"错误: 输出目录不存在: {output_dir}")
        return
    
    # 存储所有结果
    results = []
    
    # 遍历所有子目录
    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        # 查找cv_summary.json文件
        summary_file = subdir / 'cv_summary.json'
        
        if not summary_file.exists():
            print(f"警告: {subdir.name} 中没有找到 cv_summary.json，跳过")
            continue
        
        # 读取summary文件
        summary_data = read_cv_summary(summary_file)
        if summary_data is None:
            continue
        
        # 提取指标
        model_name = subdir.name
        metrics = extract_metrics(summary_data, model_name)
        results.append(metrics)
        print(f"✓ 已读取: {model_name}")
    
    if not results:
        print("错误: 没有找到任何结果文件")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按照表头顺序排列列
    columns_order = [
        '模型',
        '平均最佳验证 mAP',
        '平均最佳验证准确率',
        '平均 mAP',
        '平均 Precision',
        '平均 Recall',
        '平均 F1 Score',
        '平均最终验证准确率',
        '平均最终验证损失',
    ]
    
    df = df[columns_order]
    
    # 保存为Excel文件
    excel_path = output_dir / 'experiment_results_summary.xlsx'
    df.to_excel(excel_path, index=False, engine='openpyxl')
    
    print(f"\n✓ 汇总完成！")
    print(f"✓ 共汇总了 {len(results)} 个实验结果")
    print(f"✓ Excel文件保存至: {excel_path}")
    
    # 打印汇总表预览
    print("\n汇总表预览:")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

