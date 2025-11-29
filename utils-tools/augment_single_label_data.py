"""
对single_label_data进行数据增强
- 仅对训练集进行增强，验证集和测试集保持原始数据
- 使用上采样和下采样，使每个类别达到500张图像
- 应用8种图像增强技术
"""

import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm


class ImageAugmenter:
    """图像增强器，实现8种增强技术
    
    参数配置说明（根据医学影像特点优化）：
    - 旋转：5°至8°（略微收紧，避免过大角度）
    - 缩放：2.5%至10%（保持不变，合理范围）
    - 平移：5%至10%（保持不变，提升位置鲁棒性）
    - 水平翻转：0.3～0.4（降低概率，医学影像需谨慎）
    - 亮度：±15%～20%（收紧上限，避免极端失真）
    - 对比度：±10%～20%（保持不变，关键参数）
    - 高斯噪声：σ = 0.01～0.03（收紧上限，避免掩盖病理信息）
    """
    
    def __init__(self, enable_horizontal_flip=True, flip_prob=0.4):
        """
        Args:
            enable_horizontal_flip: 是否启用水平翻转（对于单侧疾病应禁用）
            flip_prob: 水平翻转概率（如果启用）
        """
        self.rotation_range = (5, 8)  # 旋转角度范围：5°至8°（略微收紧）
        self.scaling_range = (0.025, 0.10)  # 缩放范围：2.5%至10%（保持不变）
        self.translation_range = (0.05, 0.10)  # 平移范围：5%至10%（保持不变）
        self.enable_horizontal_flip = enable_horizontal_flip  # 是否启用水平翻转
        self.flip_prob = flip_prob  # 水平翻转概率：0.3～0.4（降低，需谨慎）
        self.brightness_range = (0.15, 0.20)  # 亮度变化范围：±15%～20%（收紧上限）
        self.contrast_range = (0.10, 0.20)  # 对比度变化范围：±10%～20%（保持不变）
        self.noise_sigma_range = (0.01, 0.03)  # 高斯噪声：σ = 0.01～0.03（收紧上限）
    
    def rotate(self, image):
        """旋转：5°至8°的随机角度（略微收紧，避免过大角度）"""
        angle = random.uniform(-self.rotation_range[1], -self.rotation_range[0])
        if random.random() < 0.5:
            angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
        return image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    
    def scale(self, image):
        """缩放：2.5%至10%的随机放大或缩小"""
        scale_factor = random.uniform(1 - self.scaling_range[1], 1 + self.scaling_range[1])
        if random.random() < 0.5:
            scale_factor = random.uniform(1 - self.scaling_range[0], 1 + self.scaling_range[0])
        
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # 缩放图像
        scaled = image.resize((new_width, new_height), Image.LANCZOS)
        
        # 裁剪或填充回原始尺寸
        if scale_factor > 1:
            # 放大：裁剪中心部分
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            return scaled.crop((left, top, left + width, top + height))
        else:
            # 缩小：填充黑色
            result = Image.new(image.mode, (width, height), (0, 0, 0))
            paste_x = (width - new_width) // 2
            paste_y = (height - new_height) // 2
            result.paste(scaled, (paste_x, paste_y))
            return result
    
    def translate(self, image):
        """平移：沿水平或垂直方向随机移动图像像素的5%至10%"""
        width, height = image.size
        tx_range = int(width * self.translation_range[0]), int(width * self.translation_range[1])
        ty_range = int(height * self.translation_range[0]), int(height * self.translation_range[1])
        
        tx = random.randint(-tx_range[1], tx_range[1])
        ty = random.randint(-ty_range[1], ty_range[1])
        
        # 使用仿射变换
        result = Image.new(image.mode, (width, height), (0, 0, 0))
        result.paste(image, (tx, ty))
        return result
    
    def horizontal_flip(self, image):
        """水平翻转：概率0.3～0.4（需谨慎，对于单侧疾病应禁用）"""
        if not self.enable_horizontal_flip:
            return image
        if random.random() < self.flip_prob:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image
    
    def adjust_brightness(self, image):
        """亮度变化：±15%～20%（收紧上限，避免极端失真）"""
        factor = random.uniform(1 - self.brightness_range[1], 1 + self.brightness_range[1])
        if random.random() < 0.5:
            factor = random.uniform(1 - self.brightness_range[0], 1 + self.brightness_range[0])
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image):
        """对比度变化：±10%～20%"""
        factor = random.uniform(1 - self.contrast_range[1], 1 + self.contrast_range[1])
        if random.random() < 0.5:
            factor = random.uniform(1 - self.contrast_range[0], 1 + self.contrast_range[0])
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def add_gaussian_noise(self, image):
        """高斯噪声：σ = 0.01～0.03（收紧上限，避免掩盖病理信息）"""
        sigma = random.uniform(self.noise_sigma_range[0], self.noise_sigma_range[1])
        
        # 转换为numpy数组
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # 添加高斯噪声
        noise = np.random.normal(0, sigma, img_array.shape)
        noisy_img = img_array + noise
        
        # 裁剪到[0, 1]范围
        noisy_img = np.clip(noisy_img, 0, 1)
        
        # 转换回PIL Image
        noisy_img = (noisy_img * 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
    def augment_image(self, image):
        """应用随机增强（随机选择几种增强技术）"""
        # 随机选择要应用的增强技术（至少应用1-3种）
        augmentations = []
        
        # 旋转
        if random.random() < 0.7:
            image = self.rotate(image)
        
        # 缩放
        if random.random() < 0.6:
            image = self.scale(image)
        
        # 平移
        if random.random() < 0.6:
            image = self.translate(image)
        
        # 水平翻转
        image = self.horizontal_flip(image)
        
        # 亮度调整
        if random.random() < 0.7:
            image = self.adjust_brightness(image)
        
        # 对比度调整
        if random.random() < 0.7:
            image = self.adjust_contrast(image)
        
        # 高斯噪声
        if random.random() < 0.5:
            image = self.add_gaussian_noise(image)
        
        return image


def load_dataset_from_folder(data_dir):
    """从按类别组织的文件夹结构加载数据集
    
    Args:
        data_dir: 数据目录，包含按类别组织的子文件夹
        
    Returns:
        DataFrame with columns: filename, label
    """
    valid_data = []
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 标准化标签
    def normalize_label(label):
        label = str(label).strip()
        label = ' '.join(label.split())
        label_mapping = {
            'Acetabular  Loosening': 'Acetabular Loosening',
            'Stem  Loosening': 'Stem Loosening',
            'Stem Loosening ': 'Stem Loosening',
            'Good Position': 'Good Place',
            'Linear Wear': 'Wear',
        }
        return label_mapping.get(label, label)
    
    # 遍历每个类别文件夹
    for label_folder in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(label_path):
            continue
        
        label = normalize_label(label_folder)
        
        # 遍历该类别下的所有图像文件
        for filename in os.listdir(label_path):
            file_path = os.path.join(label_path, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_extensions:
                    # 保存相对路径：label/filename
                    relative_path = os.path.join(label, filename)
                    valid_data.append({
                        'filename': relative_path,
                        'label': label
                    })
    
    return pd.DataFrame(valid_data)


def load_dataset(csv_path=None, image_dir=None, data_dir=None):
    """加载数据集
    
    支持两种方式：
    1. 从CSV文件加载（需要csv_path和image_dir）
    2. 从按类别组织的文件夹加载（需要data_dir）
    
    Args:
        csv_path: CSV文件路径（可选）
        image_dir: 图像目录（可选，与CSV配合使用）
        data_dir: 按类别组织的数据目录（可选）
        
    Returns:
        DataFrame with columns: filename, label
    """
    if data_dir is not None:
        # 从文件夹结构加载
        print(f"从文件夹结构加载数据: {data_dir}")
        return load_dataset_from_folder(data_dir)
    elif csv_path is not None and image_dir is not None:
        # 从CSV文件加载
        print(f"从CSV文件加载数据: {csv_path}")
        df = pd.read_csv(csv_path, header=None, names=['filename', 'label'])
        
        # 标准化标签
        def normalize_label(label):
            label = str(label).strip()
            label = ' '.join(label.split())
            label_mapping = {
                'Acetabular  Loosening': 'Acetabular Loosening',
                'Stem  Loosening': 'Stem Loosening',
                'Stem Loosening ': 'Stem Loosening',
                'Good Position': 'Good Place',
                'Linear Wear': 'Wear',
            }
            return label_mapping.get(label, label)
        
        df['label'] = df['label'].apply(normalize_label)
        
        # 验证文件存在
        valid_data = []
        for idx, row in df.iterrows():
            img_path = os.path.join(image_dir, row['filename'])
            if os.path.exists(img_path):
                valid_data.append(row)
            else:
                print(f"警告: 文件不存在: {img_path}")
        
        return pd.DataFrame(valid_data)
    else:
        raise ValueError("必须提供 data_dir 或 (csv_path + image_dir)")


def split_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    """划分训练集、验证集和测试集"""
    # 先分出测试集
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    # 再从训练+验证集中分出验证集
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=random_state, stratify=train_val['label']
    )
    
    return train, val, test


def balance_class_to_target(df, target_count, data_dir, augmenter, output_dir, split_name):
    """平衡类别到目标数量（上采样或下采样），并复制/生成图像
    
    Args:
        df: 数据DataFrame，包含filename和label列
        target_count: 目标数量
        data_dir: 数据目录（按类别组织的文件夹或单个图像目录）
        augmenter: 图像增强器
        output_dir: 输出目录
        split_name: 数据集分割名称（train/val/test）
    """
    balanced_data = []
    
    for label in df['label'].unique():
        label_data = df[df['label'] == label].copy()
        current_count = len(label_data)
        
        print(f"\n处理类别: {label}")
        print(f"  当前数量: {current_count}, 目标数量: {target_count}")
        
        label_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(label_dir, exist_ok=True)
        
        if current_count > target_count:
            # 下采样：随机选择target_count个样本
            print(f"  执行下采样...")
            sampled = label_data.sample(n=target_count, random_state=42)
            
            # 复制选中的原始图像
            for idx, row in sampled.iterrows():
                # filename可能是相对路径（label/filename）或绝对路径
                if os.path.sep in row['filename'] and os.path.exists(row['filename']):
                    # 已经是完整路径
                    src_path = row['filename']
                else:
                    # 相对路径，需要拼接
                    src_path = os.path.join(data_dir, row['filename'])
                
                dst_path = os.path.join(label_dir, os.path.basename(row['filename']))
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"    警告: 源文件不存在: {src_path}")
            
            # 更新filename为basename
            sampled['filename'] = sampled['filename'].apply(lambda x: os.path.basename(x))
            balanced_data.append(sampled)
            
        elif current_count < target_count:
            # 上采样：使用数据增强生成更多样本
            print(f"  执行上采样（数据增强）...")
            needed = target_count - current_count
            
            # 先复制所有原始数据
            for idx, row in label_data.iterrows():
                # filename可能是相对路径（label/filename）或绝对路径
                if os.path.sep in row['filename'] and os.path.exists(row['filename']):
                    # 已经是完整路径
                    src_path = row['filename']
                else:
                    # 相对路径，需要拼接
                    src_path = os.path.join(data_dir, row['filename'])
                
                dst_path = os.path.join(label_dir, os.path.basename(row['filename']))
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"    警告: 源文件不存在: {src_path}")
            
            # 更新filename为basename
            label_data['filename'] = label_data['filename'].apply(lambda x: os.path.basename(x))
            balanced_data.append(label_data)
            
            # 生成增强样本
            augmented_count = 0
            iteration = 0
            max_iterations = 1000  # 防止无限循环
            
            while augmented_count < needed and iteration < max_iterations:
                # 随机选择一个原始样本进行增强
                sample = label_data.sample(n=1, random_state=None).iloc[0]
                
                # 构建源图像路径
                if os.path.sep in sample['filename'] and os.path.exists(sample['filename']):
                    img_path = sample['filename']
                else:
                    img_path = os.path.join(data_dir, label, sample['filename'])
                
                try:
                    # 加载图像
                    image = Image.open(img_path).convert('RGB')
                    
                    # 应用增强
                    augmented_image = augmenter.augment_image(image)
                    
                    # 保存增强后的图像
                    base_name = os.path.basename(sample['filename'])
                    name, ext = os.path.splitext(base_name)
                    aug_filename = f"aug_{augmented_count:05d}_{name}{ext}"
                    aug_path = os.path.join(label_dir, aug_filename)
                    augmented_image.save(aug_path)
                    
                    # 添加到数据列表
                    new_row = pd.DataFrame({
                        'filename': [aug_filename],
                        'label': [label]
                    })
                    balanced_data.append(new_row)
                    
                    augmented_count += 1
                    if augmented_count % 50 == 0:
                        print(f"    已生成 {augmented_count}/{needed} 个增强样本...")
                
                except Exception as e:
                    print(f"    警告: 处理 {sample['filename']} 时出错: {e}")
                
                iteration += 1
            
            if augmented_count < needed:
                print(f"  警告: 仅生成了 {augmented_count}/{needed} 个增强样本")
        else:
            # 数量正好，直接复制所有原始数据
            for idx, row in label_data.iterrows():
                # filename可能是相对路径（label/filename）或绝对路径
                if os.path.sep in row['filename'] and os.path.exists(row['filename']):
                    # 已经是完整路径
                    src_path = row['filename']
                else:
                    # 相对路径，需要拼接
                    src_path = os.path.join(data_dir, row['filename'])
                
                dst_path = os.path.join(label_dir, os.path.basename(row['filename']))
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"    警告: 源文件不存在: {src_path}")
            
            # 更新filename为basename
            label_data['filename'] = label_data['filename'].apply(lambda x: os.path.basename(x))
            balanced_data.append(label_data)
    
    result_df = pd.concat(balanced_data, ignore_index=True)
    return result_df


def copy_original_images(df, data_dir, output_dir, split_name):
    """复制原始图像到输出目录"""
    print(f"\n复制 {split_name} 集的原始图像...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"复制 {split_name}"):
        # filename可能是相对路径（label/filename）或绝对路径
        if os.path.sep in row['filename'] and os.path.exists(row['filename']):
            # 已经是完整路径
            src_path = row['filename']
        else:
            # 相对路径，需要拼接
            src_path = os.path.join(data_dir, row['filename'])
        
        dst_dir = os.path.join(output_dir, split_name, row['label'])
        dst_path = os.path.join(dst_dir, os.path.basename(row['filename']))
        
        os.makedirs(dst_dir, exist_ok=True)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"警告: 源文件不存在: {src_path}")


def create_augmented_dataset(csv_path=None, image_dir=None, data_dir=None, output_dir=None, 
                             target_count=500, test_size=0.2, val_size=0.1, random_state=42,
                             enable_horizontal_flip=True, flip_prob=0.4):
    """创建增强后的数据集
    
    Args:
        csv_path: CSV文件路径（可选，与image_dir配合使用）
        image_dir: 图像目录（可选，与csv_path配合使用）
        data_dir: 按类别组织的数据目录（优先使用）
        output_dir: 输出目录
        target_count: 每个类别的目标数量
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        enable_horizontal_flip: 是否启用水平翻转（对于单侧疾病应禁用）
        flip_prob: 水平翻转概率（如果启用）
    """
    
    print("=" * 60)
    print("开始创建增强数据集")
    print("=" * 60)
    print("\n数据增强参数配置:")
    print(f"  - 旋转角度: 5°至8°")
    print(f"  - 缩放范围: 2.5%至10%")
    print(f"  - 平移范围: 5%至10%")
    print(f"  - 水平翻转: {'启用' if enable_horizontal_flip else '禁用'} (概率: {flip_prob})")
    print(f"  - 亮度变化: ±15%～20%")
    print(f"  - 对比度变化: ±10%～20%")
    print(f"  - 高斯噪声: σ = 0.01～0.03")
    print("=" * 60)
    
    # 加载数据集
    print("\n1. 加载数据集...")
    df = load_dataset(csv_path=csv_path, image_dir=image_dir, data_dir=data_dir)
    print(f"   总样本数: {len(df)}")
    print(f"   类别数: {df['label'].nunique()}")
    print(f"   类别分布:")
    print(df['label'].value_counts().to_string())
    
    # 确定数据目录
    if data_dir is not None:
        actual_data_dir = data_dir
    else:
        actual_data_dir = image_dir
    
    # 划分数据集
    print("\n2. 划分数据集...")
    train_df, val_df, test_df = split_dataset(df, test_size, val_size, random_state)
    print(f"   训练集: {len(train_df)} 样本")
    print(f"   验证集: {len(val_df)} 样本")
    print(f"   测试集: {len(test_df)} 样本")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化增强器
    augmenter = ImageAugmenter(enable_horizontal_flip=enable_horizontal_flip, flip_prob=flip_prob)
    
    # 处理验证集和测试集（不增强，只复制）
    print("\n3. 处理验证集和测试集（不增强）...")
    copy_original_images(val_df, actual_data_dir, output_dir, 'val')
    copy_original_images(test_df, actual_data_dir, output_dir, 'test')
    
    # 处理训练集（增强并平衡）
    print("\n4. 处理训练集（增强并平衡到每个类别500张）...")
    
    # 平衡训练集到目标数量（包含复制原始图像和生成增强图像）
    balanced_train_df = balance_class_to_target(
        train_df, target_count, actual_data_dir, augmenter, output_dir, 'train'
    )
    
    # 保存数据集CSV文件
    print("\n5. 保存数据集信息...")
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, header=False)
    balanced_train_df.to_csv(os.path.join(output_dir, 'train_balanced.csv'), index=False, header=False)
    
    # 打印最终统计
    print("\n" + "=" * 60)
    print("数据集创建完成！")
    print("=" * 60)
    print(f"\n最终统计:")
    print(f"  训练集（增强后）: {len(balanced_train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")
    print(f"  测试集: {len(test_df)} 样本")
    
    print(f"\n训练集类别分布:")
    print(balanced_train_df['label'].value_counts().to_string())
    
    print(f"\n输出目录: {output_dir}")
    print(f"  - train/: 训练集（已增强和平衡）")
    print(f"  - val/: 验证集（原始数据）")
    print(f"  - test/: 测试集（原始数据）")
    print(f"  - train.csv: 训练集原始列表")
    print(f"  - train_balanced.csv: 训练集增强后列表")
    print(f"  - val.csv: 验证集列表")
    print(f"  - test.csv: 测试集列表")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='对single_label_data进行数据增强')
    parser.add_argument('--data_dir', type=str,
                       default='/data/wangweicheng/ModelsToTrains/single_label_data',
                       help='按类别组织的数据目录（优先使用，每个类别一个子文件夹）')
    parser.add_argument('--csv', type=str, default=None,
                       help='单标签CSV文件路径（可选，与--image_dir配合使用）')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='图像文件目录（可选，与--csv配合使用）')
    parser.add_argument('--output_dir', type=str,
                       default='/data/wangweicheng/ModelsToTrains/data/augmented',
                       help='输出目录')
    parser.add_argument('--target_count', type=int, default=500,
                       help='每个类别的目标图像数量（默认500）')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例（默认0.2）')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='验证集比例（默认0.1）')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子（默认42）')
    parser.add_argument('--enable_horizontal_flip', action='store_true', default=True,
                       help='启用水平翻转（默认启用，对于单侧疾病应禁用）')
    parser.add_argument('--disable_horizontal_flip', action='store_false', dest='enable_horizontal_flip',
                       help='禁用水平翻转（对于单侧疾病使用此选项）')
    parser.add_argument('--flip_prob', type=float, default=0.4,
                       help='水平翻转概率（默认0.4，范围0.3-0.5）')
    
    args = parser.parse_args()
    
    # 检查输入
    if args.data_dir is not None:
        if not os.path.exists(args.data_dir):
            print(f"错误: 数据目录不存在: {args.data_dir}")
            exit(1)
    elif args.csv is not None and args.image_dir is not None:
        if not os.path.exists(args.csv):
            print(f"错误: CSV文件不存在: {args.csv}")
            exit(1)
        if not os.path.exists(args.image_dir):
            print(f"错误: 图像目录不存在: {args.image_dir}")
            exit(1)
    else:
        print("错误: 必须提供 --data_dir 或 (--csv + --image_dir)")
        exit(1)
    
    # 创建增强数据集
    create_augmented_dataset(
        csv_path=args.csv,
        image_dir=args.image_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_count=args.target_count,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        enable_horizontal_flip=args.enable_horizontal_flip,
        flip_prob=args.flip_prob
    )

