
import json, os
base = "checkpoints/cv_multi_models"
report_file = os.path.join(base, "summary_report.txt")
models = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
with open(report_file, "w", encoding="utf-8") as f:
    f.write("=========================================\n")
    f.write("多模型五折交叉验证汇总报告\n")
    f.write("生成时间: " + __import__("datetime").datetime.now().strftime("%c") + "\n")
    f.write("=========================================\n\n")
    f.write("模型结果:\n")
    f.write("=========================================\n")
    for model in sorted(models):
        summary = os.path.join(base, model, "cv_summary.json")
        f.write(f"\n模型: {model}\n")
        f.write("----------------------------------------\n")
        if os.path.exists(summary):
            try:
                data = json.load(open(summary))
                f.write(f"  平均mAP: {data.get('average_mAP',0):.2f}% ± {data.get('std_mAP',0):.2f}%\n")
                f.write(f"  平均Precision: {data.get('average_precision',0):.2f}% ± {data.get('std_precision',0):.2f}%\n")
                f.write(f"  平均Recall: {data.get('average_recall',0):.2f}% ± {data.get('std_recall',0):.2f}%\n")
                f.write(f"  平均F1 Score: {data.get('average_f1',0):.2f}% ± {data.get('std_f1',0):.2f}%\n")
                f.write(f"  平均准确率: {data.get('average_best_val_acc',0):.2f}% ± {data.get('std_best_val_acc',0):.2f}%\n")
                f.write(f"  参数量: {data.get('params_millions',0):.2f}M\n")
                f.write(f"  FLOPs: {data.get('flops_millions',0):.2f}M\n")
            except Exception as exc:
                f.write(f"  错误: 无法读取结果文件 - {exc}\n")
        else:
            f.write("  状态: cv_summary.json 缺失\n")
print(f"汇总报告已刷新: {report_file}")
