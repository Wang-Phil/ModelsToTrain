import json, os
from datetime import datetime

base = "checkpoints/cross_block_liter/ablation_study"
report_file = os.path.join(base, "summary_report.txt")

# 获取所有模型目录
models = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and not d.startswith('.')]

# 收集所有模型的结果
results = []
for model in models:
    summary = os.path.join(base, model, "cv_summary.json")
    if os.path.exists(summary):
        try:
            data = json.load(open(summary, encoding='utf-8'))
            results.append({
                'model': model,
                'data': data
            })
        except Exception as exc:
            print(f"警告: 无法读取 {model} 的结果文件 - {exc}")
            results.append({
                'model': model,
                'data': None,
                'error': str(exc)
            })
    else:
        results.append({
            'model': model,
            'data': None,
            'error': 'cv_summary.json 缺失'
        })

# 按 mAP 排序（降序）
results.sort(key=lambda x: x['data'].get('average_best_val_mAP', 0) if x['data'] else -1, reverse=True)

# 生成报告
with open(report_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("SKNet 尺寸组合消融实验汇总报告\n")
    f.write("=" * 80 + "\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    # 如果有训练配置信息，显示第一个模型的信息
    if results and results[0]['data']:
        first_data = results[0]['data']
        f.write("训练配置:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  数据目录: single_label_data\n")
        f.write(f"  训练轮数: {first_data.get('epochs', 'N/A')}\n")
        f.write(f"  批次大小: {first_data.get('batch_size', 'N/A')}\n")
        f.write(f"  学习率: {first_data.get('lr', 'N/A')}\n")
        f.write(f"  优化器: {first_data.get('optimizer', 'N/A')}\n")
        f.write(f"  损失函数: {first_data.get('loss', 'N/A')}\n")
        f.write(f"  数据增强: {first_data.get('augmentation', 'N/A')}\n")
        f.write(f"  交叉验证折数: {first_data.get('n_splits', 5)}\n")
        f.write("\n")
    
    f.write("=" * 80 + "\n")
    f.write("各模型结果 (按 mAP 降序排列)\n")
    f.write("=" * 80 + "\n\n")
    
    # 表格头部
    f.write(f"{'模型':<20} {'mAP (%)':<18} {'Acc (%)':<18} {'Precision (%)':<18} {'Recall (%)':<18} {'F1 (%)':<18} {'Params':<12} {'FLOPs':<12}\n")
    f.write("-" * 80 + "\n")
    
    for result in results:
        model = result['model']
        data = result['data']
        
        if data:
            # 提取关键指标
            avg_mAP = data.get('average_best_val_mAP', data.get('average_mAP', 0))
            std_mAP = data.get('std_best_val_mAP', data.get('std_mAP', 0))
            avg_acc = data.get('average_best_val_acc', 0)
            std_acc = data.get('std_best_val_acc', 0)
            avg_precision = data.get('average_precision', 0)
            std_precision = data.get('std_precision', 0)
            avg_recall = data.get('average_recall', 0)
            std_recall = data.get('std_recall', 0)
            avg_f1 = data.get('average_f1', 0)
            std_f1 = data.get('std_f1', 0)
            params = data.get('params_millions', 0)
            flops = data.get('flops_millions', 0)
            
            # 表格行
            f.write(f"{model:<20} {avg_mAP:>6.2f}±{std_mAP:>5.2f}  {avg_acc:>6.2f}±{std_acc:>5.2f}  "
                   f"{avg_precision:>6.2f}±{std_precision:>5.2f}  {avg_recall:>6.2f}±{std_recall:>5.2f}  "
                   f"{avg_f1:>6.2f}±{std_f1:>5.2f}  {params:>6.2f}M  {flops:>6.2f}M\n")
        else:
            f.write(f"{model:<20} {'错误: ' + result.get('error', '未知错误'):<78}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("详细结果:\n")
    f.write("=" * 80 + "\n\n")
    
    # 详细结果
    for result in results:
        model = result['model']
        data = result['data']
        
        f.write(f"\n模型: {model}\n")
        f.write("-" * 80 + "\n")
        
        if data:
            avg_mAP = data.get('average_best_val_mAP', data.get('average_mAP', 0))
            std_mAP = data.get('std_best_val_mAP', data.get('std_mAP', 0))
            avg_acc = data.get('average_best_val_acc', 0)
            std_acc = data.get('std_best_val_acc', 0)
            avg_precision = data.get('average_precision', 0)
            std_precision = data.get('std_precision', 0)
            avg_recall = data.get('average_recall', 0)
            std_recall = data.get('std_recall', 0)
            avg_f1 = data.get('average_f1', 0)
            std_f1 = data.get('std_f1', 0)
            params = data.get('params_millions', 0)
            flops = data.get('flops_millions', 0)
            
            f.write(f"  平均最佳验证mAP: {avg_mAP:.2f}% ± {std_mAP:.2f}% [主要指标]\n")
            f.write(f"  平均最佳验证准确率: {avg_acc:.2f}% ± {std_acc:.2f}%\n")
            f.write(f"  平均mAP: {data.get('average_mAP', avg_mAP):.2f}% ± {data.get('std_mAP', std_mAP):.2f}%\n")
            f.write(f"  平均Precision: {avg_precision:.2f}% ± {std_precision:.2f}%\n")
            f.write(f"  平均Recall: {avg_recall:.2f}% ± {std_recall:.2f}%\n")
            f.write(f"  平均F1 Score: {avg_f1:.2f}% ± {std_f1:.2f}%\n")
            f.write(f"  参数量: {params:.2f}M\n")
            f.write(f"  FLOPs: {flops:.2f}M\n")
        else:
            f.write(f"  状态: {result.get('error', 'cv_summary.json 缺失')}\n")
    
    # 统计信息
    valid_results = [r for r in results if r['data']]
    if valid_results:
        f.write("\n" + "=" * 80 + "\n")
        f.write("统计信息:\n")
        f.write("=" * 80 + "\n")
        f.write(f"  有效模型数量: {len(valid_results)}/{len(results)}\n")
        
        mAPs = [r['data'].get('average_best_val_mAP', r['data'].get('average_mAP', 0)) for r in valid_results]
        if mAPs:
            f.write(f"  mAP 范围: {min(mAPs):.2f}% ~ {max(mAPs):.2f}%\n")
            f.write(f"  mAP 平均值: {sum(mAPs)/len(mAPs):.2f}%\n")
            best_model = max(valid_results, key=lambda x: x['data'].get('average_best_val_mAP', x['data'].get('average_mAP', 0)))
            f.write(f"  最佳模型: {best_model['model']} (mAP: {best_model['data'].get('average_best_val_mAP', best_model['data'].get('average_mAP', 0)):.2f}%)\n")

print(f"汇总报告已生成: {report_file}")
print(f"共处理 {len(results)} 个模型，其中 {len([r for r in results if r['data']])} 个有效")
