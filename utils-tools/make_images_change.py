import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 初始数据：每个类别的图像数量
categories = ['Acetabular Loosening', 'Dislocation', 'Fracture', 'Good Place', 
              'Infection', 'Native Hip', 'Spacer', 'Stem Loosening', 'Wear']
initial_counts = [287, 57, 125, 261, 85, 744, 53, 90, 25]

# 将数据存储为 DataFrame
df = pd.DataFrame({'Category': categories, 'Initial Count': initial_counts})

# 下采样和上采样
desired_count = 250
df['Final Count'] = df['Initial Count'].apply(lambda x: desired_count if x > desired_count else x)
df['Final Count'] = df['Final Count'].apply(lambda x: desired_count if x < desired_count else x)

# 设置图像大小
plt.figure(figsize=(12, 6))
bar_width = 0.4
index = np.arange(len(categories))

# 绘制初始数量的柱状图
plt.bar(index, df['Initial Count'], bar_width, color='#79BBFF', label='Initial Count')

# 绘制最终数量的柱状图
plt.bar(index + bar_width, df['Final Count'], bar_width, color='#FEC194', label='Final Count')

# 添加一条虚线，表示 250 处
plt.axhline(y=250, color='gray', linestyle='--', linewidth=1, label='250 Threshold')

# 添加标题和标签
plt.title('Image Count Change Before and After Sampling', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Image Count', fontsize=12)
plt.xticks(index + bar_width / 2, df['Category'], rotation=45, ha='right')

# 添加图例
plt.legend()

# 保存图像到当前路径
plt.tight_layout()
plt.savefig('image_count_comparison_with_arrows_and_threshold.png')  # 保存图片为PNG格式
plt.show()
