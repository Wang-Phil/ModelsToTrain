"""
准备和验证交叉验证数据集
检查数据质量、统计类别分布、验证数据格式
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from PIL import Image
import json

def check_dataset_structure(data_dir):
    """检查数据集结构"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return False
    
    print(f"检查数据集结构: {data_dir}")
    print("=" * 80)
    
    # 获取所有类别文件夹
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        print("错误: 未找到类别子文件夹")
        return False
    
    print(f"\n找到 {len(class_dirs)} 个类别:")
    print("-" * 80)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    class_stats = {}
    total_images = 0
    invalid_images = []
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = []
        
        # 查找所有图像文件
        for ext in image_extensions:
            images.extend(list(class_dir.glob(f'*{ext}')))
        
        # 验证图像文件
        valid_images = []
        for img_path in images:
            try:
                img = Image.open(img_path)
                img.verify()  # 验证图像完整性
                valid_images.append(img_path)
            except Exception as e:
                invalid_images.append((str(img_path), str(e)))
        
        class_stats[class_name] = {
            'total_files': len(images),
            'valid_images': len(valid_images),
            'invalid_images': len(images) - len(valid_images)
        }
        
        total_images += len(valid_images)
        
        print(f"  {class_name:30s} {len(valid_images):5d} 张图像")
    
    print("-" * 80)
    print(f"总计: {total_images} 张有效图像")
    
    # 检查类别分布
    print("\n类别分布统计:")
    print("-" * 80)
    min_count = min(s['valid_images'] for s in class_stats.values())
    max_count = max(s['valid_images'] for s in class_stats.values())
    avg_count = total_images / len(class_stats)
    
    print(f"  最少: {min_count} 张")
    print(f"  最多: {max_count} 张")
    print(f"  平均: {avg_count:.1f} 张")
    print(f"  类别不平衡比例: {max_count / min_count:.2f}:1")
    
    # 检查类别不平衡
    if max_count / min_count > 3:
        print("\n  ⚠️  警告: 类别严重不平衡，建议使用加权损失函数或Focal Loss")
    
    # 检查最小样本数（对于5折交叉验证）
    min_samples_per_fold = min_count / 5
    if min_samples_per_fold < 1:
        print(f"\n  ⚠️  警告: 某些类别在5折交叉验证中每折可能少于1个样本")
    elif min_samples_per_fold < 5:
        print(f"\n  ⚠️  警告: 某些类别在5折交叉验证中每折样本数较少（<5）")
    else:
        print(f"\n  ✓ 5折交叉验证: 每折最少约 {min_samples_per_fold:.1f} 个样本")
    
    # 报告无效图像
    if invalid_images:
        print(f"\n发现 {len(invalid_images)} 个无效图像文件:")
        for img_path, error in invalid_images[:10]:  # 只显示前10个
            print(f"  {img_path}: {error}")
        if len(invalid_images) > 10:
            print(f"  ... 还有 {len(invalid_images) - 10} 个无效文件")
    
    # 保存统计信息
    stats_file = data_dir.parent / f'{data_dir.name}_stats.json'
    stats = {
        'data_dir': str(data_dir),
        'num_classes': len(class_stats),
        'total_images': total_images,
        'class_stats': class_stats,
        'min_count': min_count,
        'max_count': max_count,
        'avg_count': avg_count,
        'imbalance_ratio': max_count / min_count,
        'invalid_images': len(invalid_images)
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计信息已保存到: {stats_file}")
    
    return True


def visualize_class_distribution(data_dir, output_file=None):
    """可视化类别分布"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("matplotlib未安装，跳过可视化")
        return
    
    data_dir = Path(data_dir)
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    class_names = []
    class_counts = []
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = []
        for ext in image_extensions:
            images.extend(list(class_dir.glob(f'*{ext}')))
        
        # 验证图像
        valid_count = 0
        for img_path in images:
            try:
                img = Image.open(img_path)
                img.verify()
                valid_count += 1
            except:
                pass
        
        class_names.append(class_name)
        class_counts.append(valid_count)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 柱状图
    bars = ax1.bar(range(len(class_names)), class_counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('类别', fontsize=12)
    ax1.set_ylabel('图像数量', fontsize=12)
    ax1.set_title('类别分布（柱状图）', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, class_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=9)
    
    # 饼图
    colors = plt.cm.Set3(range(len(class_names)))
    wedges, texts, autotexts = ax2.pie(class_counts, labels=class_names, autopct='%1.1f%%',
                                       startangle=90, colors=colors)
    ax2.set_title('类别分布（饼图）', fontsize=14, fontweight='bold')
    
    # 调整文本大小
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
    
    plt.tight_layout()
    
    if output_file is None:
        output_file = data_dir.parent / f'{data_dir.name}_distribution.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n类别分布图已保存到: {output_file}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='准备和验证交叉验证数据集')
    parser.add_argument('--data-dir', type=str, default='single_label_data',
                        help='数据目录路径')
    parser.add_argument('--visualize', action='store_true',
                        help='生成类别分布可视化图表')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("交叉验证数据集准备和验证工具")
    print("=" * 80)
    
    # 检查数据集结构
    if not check_dataset_structure(args.data_dir):
        sys.exit(1)
    
    # 可视化
    if args.visualize:
        print("\n生成类别分布可视化...")
        visualize_class_distribution(args.data_dir)
    
    print("\n" + "=" * 80)
    print("数据集检查完成！")
    print("=" * 80)
    print("\n现在可以使用以下命令进行交叉验证训练:")
    print(f"\npython train_cross_validation.py \\")
    print(f"    --data-dir {args.data_dir} \\")
    print(f"    --model resnet50 \\")
    print(f"    --pretrained \\")
    print(f"    --n-splits 5 \\")
    print(f"    --epochs 50 \\")
    print(f"    --batch-size 32 \\")
    print(f"    --lr 0.001 \\")
    print(f"    --optimizer adam \\")
    print(f"    --loss focal \\")
    print(f"    --augmentation standard \\")
    print(f"    --output-dir checkpoints/cv_results \\")
    print(f"    --device cuda:0 \\")
    print(f"    --seed 42")


if __name__ == '__main__':
    main()

