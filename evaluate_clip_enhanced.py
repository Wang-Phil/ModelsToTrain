"""
增强版CLIP模型评估脚本
计算mAP、准确率、精确率、召回率、F1分数等指标
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

# 设置环境变量
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
if 'HF_ENDPOINT' not in os.environ:
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# 导入增强版CLIP模型
from models.clip_enhanced import EnhancedCLIPModel

# 导入原始CLIP模型（用于创建teacher模型）
from models.clip import CLIPModel


class CLIPDataset(Dataset):
    """CLIP数据集 - 从按类别组织的文件夹中加载图像"""
    
    def __init__(self, root_dir, transform=None, class_texts_dict=None, class_texts_file=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # 加载类别文本描述
        self.class_texts_map = {}
        if class_texts_file is not None:
            with open(class_texts_file, 'r', encoding='utf-8') as f:
                self.class_texts_map = json.load(f)
        elif class_texts_dict is not None:
            self.class_texts_map = class_texts_dict
        
        # 获取所有类别
        excluded_folders = {'split_fewshot', '__pycache__', '.ipynb_checkpoints'}
        classes = sorted([
            d.name for d in self.root_dir.iterdir() 
            if d.is_dir() and d.name not in excluded_folders and not d.name.startswith('split_')
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # 收集所有图像文件
        for cls_name in classes:
            cls_dir = self.root_dir / cls_name
            for img_file in cls_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_file), self.class_to_idx[cls_name]))
        
        print(f"数据集: {len(self.samples)} 个样本, {len(classes)} 个类别")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224):
    """获取数据变换"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def calculate_mAP(y_true, y_pred_proba, num_classes):
    """
    计算mAP (mean Average Precision)
    
    Args:
        y_true: 真实标签 [n_samples]
        y_pred_proba: 预测概率 [n_samples, n_classes]
        num_classes: 类别数量
    
    Returns:
        mAP: 平均精度均值
    """
    # 将标签转换为one-hot编码
    y_true_binary = label_binarize(y_true, classes=range(num_classes))
    
    # 如果只有两个类别，需要添加负类
    if num_classes == 2:
        y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])
        y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])
    
    # 计算每个类别的AP
    ap_scores = []
    for i in range(num_classes):
        if y_true_binary[:, i].sum() > 0:  # 如果该类别有样本
            ap = average_precision_score(y_true_binary[:, i], y_pred_proba[:, i])
            ap_scores.append(ap)
    
    # 计算mAP
    mAP = np.mean(ap_scores) if len(ap_scores) > 0 else 0.0
    return mAP


def evaluate_model(
    model_path,
    data_dir,
    image_encoder_name='resnet50',
    text_encoder_name='pubmedbert',
    embed_dim=512,
    temperature=0.07,
    class_texts_file=None,
    img_size=224,
    batch_size=32,
    num_workers=4,
    gpu_id=0,
    use_test_set=False,
    test_split=0.2
):
    """评估模型"""
    
    # 设置设备
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    transform = get_transforms(img_size)
    full_dataset = CLIPDataset(data_dir, transform=transform, class_texts_file=class_texts_file)
    
    # 获取类别名称列表
    class_names = [full_dataset.idx_to_class[i] for i in range(len(full_dataset.class_to_idx))]
    num_classes = len(class_names)
    
    print(f"\n类别列表 ({num_classes}个):")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # 划分数据集（如果需要测试集）
    if use_test_set:
        from sklearn.model_selection import train_test_split
        indices = list(range(len(full_dataset)))
        labels = [full_dataset.samples[i][1] for i in indices]
        _, test_indices = train_test_split(
            indices,
            test_size=test_split,
            random_state=42,
            stratify=labels
        )
        test_dataset = Subset(full_dataset, test_indices)
        print(f"\n使用测试集: {len(test_indices)} 个样本")
    else:
        test_dataset = full_dataset
        print(f"\n使用全部数据: {len(full_dataset)} 个样本")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型（需要知道原始配置）
    # 尝试从checkpoint中获取配置，如果没有则使用默认值
    model_config = checkpoint.get('model_config', {})
    
    # 从checkpoint中获取配置，如果没有则使用函数参数
    use_classification_loss = model_config.get('use_classification_loss', True)
    use_contrastive_loss = model_config.get('use_contrastive_loss', True)
    use_sccm_loss = model_config.get('use_sccm_loss', False)
    use_kdsp_loss = model_config.get('use_kdsp_loss', False)
    
    # 创建teacher模型（如果需要）
    teacher_model = None
    if use_kdsp_loss:
        # 从checkpoint或参数中获取teacher配置
        # 注意：这里需要根据实际训练时的配置来设置
        teacher_image_encoder = 'biomedclip'  # 默认值，应该与训练时一致
        teacher_text_encoder = 'biomedclip_text'  # 默认值，应该与训练时一致
        print(f"创建teacher模型: {teacher_image_encoder} + {teacher_text_encoder}")
        teacher_model = EnhancedCLIPModel(
            image_encoder_name=teacher_image_encoder,
            text_encoder_name=teacher_text_encoder,
            embed_dim=embed_dim,
            temperature=temperature,
            class_texts=class_names,
            class_texts_file=class_texts_file,
            use_classification_loss=False,
            use_contrastive_loss=False,
            use_sccm_loss=False,
            use_kdsp_loss=False
        )
        teacher_model.eval()
        teacher_model.to(device)
    
    # 创建模型
    model = EnhancedCLIPModel(
        image_encoder_name=image_encoder_name,
        text_encoder_name=text_encoder_name,
        embed_dim=embed_dim,
        temperature=temperature,
        class_texts=class_names,
        class_texts_file=class_texts_file,
        teacher_model=teacher_model,
        use_classification_loss=use_classification_loss,
        use_contrastive_loss=use_contrastive_loss,
        use_sccm_loss=use_sccm_loss,
        use_kdsp_loss=use_kdsp_loss,
        classification_loss_weight=model_config.get('classification_loss_weight', 1.0),
        contrastive_loss_weight=model_config.get('contrastive_loss_weight', 1.0),
        sccm_loss_weight=model_config.get('sccm_loss_weight', 1.0),
        kdsp_loss_weight=model_config.get('kdsp_loss_weight', 1.0)
    )
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        print("警告: checkpoint中没有找到model_state_dict，尝试直接加载...")
        state_dict = checkpoint
    
    # 过滤掉teacher模型的参数（如果存在）
    # 因为评估时可能不需要teacher模型
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith('teacher_model.'):
            filtered_state_dict[key] = value
    
    # 使用strict=False来允许缺失的键（如teacher模型参数）
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        print(f"警告: 以下键在模型中缺失（已忽略）: {len(missing_keys)} 个")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"  - {key}")
        else:
            for key in missing_keys[:5]:
                print(f"  - {key}")
            print(f"  ... 还有 {len(missing_keys) - 5} 个")
    
    if unexpected_keys:
        print(f"警告: 以下键在checkpoint中但不在模型中（已忽略）: {len(unexpected_keys)} 个")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"  - {key}")
        else:
            for key in unexpected_keys[:5]:
                print(f"  - {key}")
            print(f"  ... 还有 {len(unexpected_keys) - 5} 个")
    
    print("✓ 模型权重加载成功")
    
    model.eval()
    model.to(device)
    
    # 评估
    print("\n开始评估...")
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='评估中'):
            images = images.to(device)
            labels = labels.to(device)
            
            # 获取预测
            logits = model(images, labels=None)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # 计算指标
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    
    # 准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n准确率 (Accuracy): {accuracy*100:.2f}%")
    
    # 精确率、召回率、F1分数
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # 宏平均
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    # 加权平均
    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = np.average(f1, weights=support)
    
    print(f"\n宏平均 (Macro Average):")
    print(f"  精确率 (Precision): {precision_macro*100:.2f}%")
    print(f"  召回率 (Recall): {recall_macro*100:.2f}%")
    print(f"  F1分数: {f1_macro*100:.2f}%")
    
    print(f"\n加权平均 (Weighted Average):")
    print(f"  精确率 (Precision): {precision_weighted*100:.2f}%")
    print(f"  召回率 (Recall): {recall_weighted*100:.2f}%")
    print(f"  F1分数: {f1_weighted*100:.2f}%")
    
    # mAP
    mAP = calculate_mAP(all_labels, all_probabilities, num_classes)
    print(f"\nmAP (mean Average Precision): {mAP*100:.2f}%")
    
    # 每个类别的详细指标
    print(f"\n每个类别的详细指标:")
    print(f"{'类别':<25} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<10}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {precision[i]*100:>8.2f}% {recall[i]*100:>8.2f}% {f1[i]*100:>8.2f}% {support[i]:>8}")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\n混淆矩阵 (Confusion Matrix):")
    print(f"{'预测':>15}", end='')
    for i in range(num_classes):
        print(f"{i:>8}", end='')
    print()
    for i in range(num_classes):
        print(f"{class_names[i][:14]:>15}", end='')
        for j in range(num_classes):
            print(f"{cm[i, j]:>8}", end='')
        print()
    
    # 保存结果
    results = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'mAP': float(mAP),
        'per_class_metrics': {
            class_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(num_classes)
        },
        'confusion_matrix': cm.tolist()
    }
    
    # 保存到文件
    output_file = Path(model_path).parent / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='增强版CLIP模型评估脚本')
    
    parser.add_argument('--model-path', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    parser.add_argument('--image-encoder', type=str, default='resnet50', help='图像编码器名称')
    parser.add_argument('--text-encoder', type=str, default='pubmedbert', help='文本编码器名称')
    parser.add_argument('--embed-dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--temperature', type=float, default=0.07, help='温度参数')
    parser.add_argument('--class-texts-file', type=str, default=None, help='类别文本描述JSON文件路径')
    parser.add_argument('--img-size', type=int, default=224, help='图像大小')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载工作进程数')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--use-test-set', action='store_true', help='使用测试集（从数据中划分）')
    parser.add_argument('--test-split', type=float, default=0.2, help='测试集比例')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        image_encoder_name=args.image_encoder,
        text_encoder_name=args.text_encoder,
        embed_dim=args.embed_dim,
        temperature=args.temperature,
        class_texts_file=args.class_texts_file,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpu_id=args.gpu_id,
        use_test_set=args.use_test_set,
        test_split=args.test_split
    )

