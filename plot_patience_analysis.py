"""
绘制 Early Stopping Patience 对模型性能的影响
双Y轴图表：mAP vs Patience 和 Validation Loss vs Patience
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据
patience_values = [20, 30, 40, 50]
val_loss = [0.678, 0.618, 0.560, 0.615]
mAP = [60.5, 69.5, 72.5, 71.8]

# 创建图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左Y轴：mAP (%)
color1 = '#FF6B35'  # 橙红色
ax1.set_xlabel('Patience (Epochs)', fontsize=12, fontweight='bold')
ax1.set_ylabel('mAP (%)', fontsize=12, fontweight='bold', color=color1)
line1 = ax1.plot(patience_values, mAP, color=color1, marker='o', 
                 linewidth=2.5, markersize=10, label='mAP (%)', zorder=3)
ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_ylim([58, 75])  # 设置mAP的Y轴范围

# 右Y轴：Validation Loss
ax2 = ax1.twinx()
color2 = '#004E89'  # 深蓝色
ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold', color=color2)
line2 = ax2.plot(patience_values, val_loss, color=color2, marker='s', 
                 linewidth=2.5, markersize=10, label='Validation Loss', zorder=3)
ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
ax2.set_ylim([0.54, 0.70])  # 设置Loss的Y轴范围，反转以显示"越低越好"

# 添加数据点标注
for i, (p, m, l) in enumerate(zip(patience_values, mAP, val_loss)):
    # mAP标注
    ax1.annotate(f'{m:.1f}%', 
                xy=(p, m), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=9,
                color=color1,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color1, alpha=0.7))
    
    # Loss标注
    ax2.annotate(f'{l:.3f}', 
                xy=(p, l), 
                xytext=(5, -15), 
                textcoords='offset points',
                fontsize=9,
                color=color2,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color2, alpha=0.7))

# 标记最佳点 (Patience=40)
best_idx = 2  # Patience=40的索引
ax1.axvline(x=patience_values[best_idx], color='green', linestyle='--', 
            linewidth=2, alpha=0.6, label='Optimal (Patience=40)')
ax1.scatter(patience_values[best_idx], mAP[best_idx], 
           color='green', s=200, marker='*', zorder=5, 
           edgecolors='darkgreen', linewidths=2)
ax2.scatter(patience_values[best_idx], val_loss[best_idx], 
           color='green', s=200, marker='*', zorder=5, 
           edgecolors='darkgreen', linewidths=2)

# 添加最佳点标注
ax1.annotate('Best\nmAP: 72.5%\nLoss: 0.560', 
            xy=(patience_values[best_idx], mAP[best_idx]), 
            xytext=(15, 20), 
            textcoords='offset points',
            fontsize=10,
            color='green',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                     edgecolor='green', alpha=0.8, linewidth=2),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

# 设置X轴
ax1.set_xlim([18, 52])
ax1.set_xticks(patience_values)
ax1.set_xticklabels(patience_values, fontsize=11)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper left', fontsize=10, framealpha=0.9)

# 添加标题
plt.title('Early Stopping Patience Analysis\nmAP vs Patience & Validation Loss vs Patience', 
          fontsize=14, fontweight='bold', pad=20)

# 调整布局
plt.tight_layout()

# 保存图片
output_path = Path('checkpoints/cv_multi_models/patience_analysis.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已保存到: {output_path}")

# 显示图表
plt.show()

print("\n" + "=" * 80)
print("Patience 分析总结")
print("=" * 80)
print(f"{'Patience':<12} {'Validation Loss':<18} {'mAP (%)':<12} {'分析'}")
print("-" * 80)
for p, l, m in zip(patience_values, val_loss, mAP):
    if p == 40:
        analysis = "✅ 最佳点！损失最低，mAP最高"
    elif p == 20:
        analysis = "❌ 性能最差，损失最高"
    elif p == 30:
        analysis = "📈 性能显著提升，损失大幅下降"
    else:  # p == 50
        analysis = "📉 性能下降，损失反弹"
    print(f"{p:<12} {l:<18.3f} {m:<12.1f} {analysis}")
print("=" * 80)

