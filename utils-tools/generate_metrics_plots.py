import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def load_cv_summary(json_path):
    """加载交叉验证摘要文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def aggregate_confusion_matrices(cv_data):
    """聚合所有 fold 的混淆矩阵"""
    n_folds = cv_data['n_splits']
    class_names = None
    all_cm = []
    
    for fold_result in cv_data['fold_results']:
        cm = np.array(fold_result['metrics']['confusion_matrix'])
        all_cm.append(cm)
        if class_names is None:
            class_names = fold_result['metrics']['class_names']
    
    # 计算平均混淆矩阵 - 修复：确保正确堆叠
    all_cm = np.array(all_cm)  # shape: (n_folds, n_classes, n_classes)
    avg_cm = np.mean(all_cm, axis=0)  # 沿 fold 维度求平均
    
    print(f"混淆矩阵形状检查:")
    print(f"  - 单个 fold 混淆矩阵形状: {all_cm[0].shape}")
    print(f"  - 所有 fold 堆叠后形状: {all_cm.shape}")
    print(f"  - 平均混淆矩阵形状: {avg_cm.shape}")
    print(f"  - 平均混淆矩阵前几行:\n{avg_cm[:3, :]}")
    
    return avg_cm, class_names, all_cm

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 确保是numpy数组
    cm = np.array(cm)
    
    # 计算百分比
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    cm_percent = cm / row_sums * 100
    
    # 调试信息
    print(f"  混淆矩阵形状: {cm.shape}")
    print(f"  行和: {row_sums.flatten()}")
    print(f"  百分比矩阵所有行都有数据: {np.all(row_sums > 0)}")
    
    # 使用imshow绘制，确保所有数据都显示
    im = ax.imshow(cm_percent, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    # 添加数值标注
    n_classes = len(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, f'{cm_percent[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    # 设置刻度
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # 添加网格线
    ax.set_xticks(np.arange(n_classes+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n_classes+1)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", size=0)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到: {save_path}")

def plot_confusion_matrix_absolute(cm, class_names, save_path, title="Confusion Matrix (Absolute)"):
    """绘制绝对数值的混淆矩阵"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 确保是numpy数组
    cm = np.array(cm)
    
    # 如果是浮点数（平均混淆矩阵），四舍五入为整数显示
    if cm.dtype == np.float64 or cm.dtype == np.float32:
        cm_display = np.round(cm).astype(int)
    else:
        cm_display = cm.astype(int)
    
    # 调试信息
    print(f"  绝对数值混淆矩阵形状: {cm_display.shape}")
    print(f"  所有行都有数据: {np.all(cm_display.sum(axis=1) > 0)}")
    
    # 使用imshow绘制，确保所有数据都显示
    max_val = cm_display.max() if cm_display.max() > 0 else 1
    im = ax.imshow(cm_display, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max_val)
    
    # 添加数值标注
    n_classes = len(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            text_color = "white" if cm_display[i, j] > max_val / 2 else "black"
            text = ax.text(j, i, f'{cm_display[i, j]}',
                          ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')
    
    # 设置刻度
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # 添加网格线
    ax.set_xticks(np.arange(n_classes+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n_classes+1)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", size=0)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"绝对数值混淆矩阵已保存到: {save_path}")

def calculate_roc_from_cm(cm, class_names):
    """从混淆矩阵计算 ROC 曲线（近似方法）"""
    n_classes = len(class_names)
    roc_data = {}
    
    # 对每个类别计算 TPR 和 FPR
    for i, class_name in enumerate(class_names):
        # True Positives
        tp = cm[i, i]
        # False Positives
        fp = cm[:, i].sum() - tp
        # False Negatives
        fn = cm[i, :].sum() - tp
        # True Negatives
        tn = cm.sum() - tp - fp - fn
        
        # 计算 TPR (Sensitivity) 和 FPR
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        
        # 计算 AUC（单点，这只是近似值）
        # 真正的 ROC 曲线需要多个阈值点
        roc_data[class_name] = {
            'tpr': tpr,
            'fpr': fpr,
            'auc': 1 - (fpr / 2)  # 简化的 AUC 近似
        }
    
    return roc_data

def plot_roc_curves_from_cm(roc_data, class_names, save_path, title="ROC Curves (from Confusion Matrix)"):
    """从混淆矩阵数据绘制 ROC 曲线（近似）"""
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        data = roc_data[class_name]
        # 绘制单点（因为只有混淆矩阵，没有多个阈值）
        plt.plot(data['fpr'], data['tpr'], 'o', 
                label=f'{class_name} (AUC≈{data["auc"]:.3f})',
                color=colors[i], markersize=8)
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=9, ncol=1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲线已保存到: {save_path}")
    print("注意: 这是基于混淆矩阵的近似 ROC，真正的 ROC 曲线需要预测概率数据")

def plot_per_class_metrics(cv_data, save_path):
    """绘制每个类别的性能指标"""
    n_folds = cv_data['n_splits']
    class_names = cv_data['fold_results'][0]['metrics']['class_names']
    n_classes = len(class_names)
    
    # 聚合所有 fold 的每类指标
    precision_per_class = np.zeros((n_folds, n_classes))
    recall_per_class = np.zeros((n_folds, n_classes))
    f1_per_class = np.zeros((n_folds, n_classes))
    
    for fold_idx, fold_result in enumerate(cv_data['fold_results']):
        metrics = fold_result['metrics']
        precision_per_class[fold_idx] = metrics['precision_per_class']
        recall_per_class[fold_idx] = metrics['recall_per_class']
        f1_per_class[fold_idx] = metrics['f1_per_class']
    
    # 计算平均值和标准差
    precision_mean = np.mean(precision_per_class, axis=0)
    precision_std = np.std(precision_per_class, axis=0)
    recall_mean = np.mean(recall_per_class, axis=0)
    recall_std = np.std(recall_per_class, axis=0)
    f1_mean = np.mean(f1_per_class, axis=0)
    f1_std = np.std(f1_per_class, axis=0)
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(n_classes)
    width = 0.25
    
    bars1 = ax.bar(x - width, precision_mean, width, yerr=precision_std, 
                   label='Precision', alpha=0.8, capsize=5)
    bars2 = ax.bar(x, recall_mean, width, yerr=recall_std, 
                   label='Recall', alpha=0.8, capsize=5)
    bars3 = ax.bar(x + width, f1_mean, width, yerr=f1_std, 
                   label='F1-Score', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics (Mean ± Std across 5 folds)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"每类性能指标图已保存到: {save_path}")

def main():
    # 设置路径
    json_path = Path("/home/ln/wangweicheng/ModelsTotrain/checkpoints/final_starnet_models/final_model/starnet_sa_s1/cv_summary.json")
    output_dir = json_path.parent
    
    print(f"正在加载数据: {json_path}")
    cv_data = load_cv_summary(json_path)
    
    model_name = cv_data.get('model', 'model')
    print(f"模型: {model_name}")
    print(f"交叉验证折数: {cv_data['n_splits']}")
    
    # 聚合混淆矩阵
    print("\n正在聚合混淆矩阵...")
    avg_cm, class_names, all_cm = aggregate_confusion_matrices(cv_data)
    
    # 生成混淆矩阵图（百分比）
    cm_percent_path = output_dir / f"{model_name}_confusion_matrix_percent.png"
    plot_confusion_matrix(avg_cm, class_names, cm_percent_path, 
                         title=f"{model_name} - Average Confusion Matrix (Percentage)")
    
    # 生成混淆矩阵图（绝对数值）
    cm_abs_path = output_dir / f"{model_name}_confusion_matrix_absolute.png"
    plot_confusion_matrix_absolute(avg_cm, class_names, cm_abs_path,
                                  title=f"{model_name} - Average Confusion Matrix (Count)")
    
    # 从混淆矩阵计算 ROC 数据（近似）
    print("\n正在计算 ROC 曲线（基于混淆矩阵近似）...")
    roc_data = calculate_roc_from_cm(avg_cm, class_names)
    
    # 生成 ROC 曲线图
    roc_path = output_dir / f"{model_name}_roc_curves.png"
    plot_roc_curves_from_cm(roc_data, class_names, roc_path,
                            title=f"{model_name} - ROC Curves (Approximate from CM)")
    
    # 生成每类性能指标图
    print("\n正在生成每类性能指标图...")
    metrics_path = output_dir / f"{model_name}_per_class_metrics.png"
    plot_per_class_metrics(cv_data, metrics_path)
    
    # 打印摘要信息
    print("\n" + "="*60)
    print("生成完成！")
    print("="*60)
    print(f"输出文件:")
    print(f"  1. {cm_percent_path}")
    print(f"  2. {cm_abs_path}")
    print(f"  3. {roc_path}")
    print(f"  4. {metrics_path}")
    print("\n注意: ROC 曲线是基于混淆矩阵的近似值。")
    print("要获得准确的 ROC 曲线，需要保存预测概率数据。")

if __name__ == "__main__":
    main()

