"""
Grad-CAM可视化脚本
支持多种模型架构，自动识别目标层，生成模型关注热力图
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

# 导入模型
from train_multiclass import (
    ImageFolderDataset, create_model, get_data_augmentation
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GradCAM:
    """Grad-CAM实现"""
    
    def __init__(self, model, target_layers, use_cuda=True):
        self.model = model
        self.target_layers = target_layers if isinstance(target_layers, list) else [target_layers]
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            self.model = self.model.cuda()
        
        self.model.eval()
        self.gradients = []
        self.activations = []
        self.hooks = []
        
        # 注册前向和反向钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子函数"""
        def forward_hook(module, input, output):
            self.activations.append(output.detach())
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())
        
        for layer in self.target_layers:
            self.hooks.append(layer.register_forward_hook(forward_hook))
            if hasattr(layer, 'register_full_backward_hook'):
                self.hooks.append(layer.register_full_backward_hook(backward_hook))
            else:
                self.hooks.append(layer.register_backward_hook(backward_hook))
    
    def _release_hooks(self):
        """释放钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _generate_cam(self, activations, gradients):
        """生成CAM"""
        # 计算权重（梯度的全局平均池化）
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 加权激活图
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU激活
        
        # 归一化到[0, 1]
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def __call__(self, input_tensor, target_category=None):
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入张量 [B, C, H, W]
            target_category: 目标类别索引，如果为None则使用预测类别
        
        Returns:
            cam: CAM热力图 [B, H, W] 或 [H, W] (如果batch_size=1)
        """
        self.gradients = []
        self.activations = []
        
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 确定目标类别
        if target_category is None:
            target_category = torch.argmax(output, dim=1).cpu().numpy()
        
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)
        elif isinstance(target_category, torch.Tensor):
            target_category = target_category.cpu().numpy()
        
        # 反向传播
        self.model.zero_grad()
        loss = 0
        for i, cat in enumerate(target_category):
            loss += output[i, int(cat)]
        loss.backward(retain_graph=True)
        
        # 生成CAM
        cams = []
        for act, grad in zip(self.activations, self.gradients):
            cam = self._generate_cam(act, grad)
            # 调整大小到输入图像大小
            if len(cam.shape) == 2:  # [H, W]
                cam_resized = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))
            else:  # [B, H, W]
                cam_resized = np.array([
                    cv2.resize(cam[i], (input_tensor.size(3), input_tensor.size(2)))
                    for i in range(cam.shape[0])
                ])
            cams.append(cam_resized)
        
        # 如果有多个层，取平均
        if len(cams) > 1:
            cam = np.mean(cams, axis=0)
        else:
            cam = cams[0]
        
        # 如果batch_size=1，返回2D数组
        if cam.ndim == 3 and cam.shape[0] == 1:
            cam = cam[0]
        
        return cam
    
    def __del__(self):
        self._release_hooks()


def get_target_layers(model, model_name):
    """
    根据模型名称自动获取目标层
    
    Args:
        model: 模型实例
        model_name: 模型名称
    
    Returns:
        target_layers: 目标层列表
    """
    model_name = model_name.lower()
    target_layers = []
    
    try:
        # ResNet系列
        if 'resnet' in model_name:
            target_layers = [model.layer4[-1]]
        
        # ConvNeXtV2系列
        elif 'convnextv2' in model_name or 'convnext' in model_name:
            if hasattr(model, 'stages'):
                target_layers = [model.stages[-1][-1]]
            elif hasattr(model, 'downsample_layers'):
                # 查找最后一个stage的最后一个block
                if hasattr(model, 'stages') and len(model.stages) > 0:
                    target_layers = [model.stages[-1][-1]]
        
        # StarNeXt系列
        elif 'starnext' in model_name:
            if hasattr(model, 'stages'):
                target_layers = [model.stages[-1][-1]]
        
        # StarNet系列
        elif 'starnet' in model_name:
            if hasattr(model, 'stages'):
                target_layers = [model.stages[-1][-1]]
        
        # DenseNet系列
        elif 'densenet' in model_name:
            target_layers = [model.features[-1]]
        
        # MobileNet系列
        elif 'mobilenet' in model_name:
            if hasattr(model, 'features'):
                target_layers = [model.features[-1]]
        
        # EfficientNet系列
        elif 'efficientnet' in model_name:
            if hasattr(model, 'blocks'):
                target_layers = [model.blocks[-1]]
            elif hasattr(model, 'features'):
                target_layers = [model.features[-1]]
        
        # Inception系列
        elif 'inception' in model_name:
            target_layers = [model.Mixed_7c]
        
        # GoogleNet
        elif 'googlenet' in model_name:
            target_layers = [model.inception5b]
        
        # MogaNet系列
        elif 'moganet' in model_name:
            if hasattr(model, 'stages'):
                target_layers = [model.stages[-1][-1]]
        
        # 如果找不到，尝试使用最后一个卷积层
        if not target_layers:
            # 递归查找最后一个卷积层
            def find_last_conv(module):
                last_conv = None
                for child in module.children():
                    if isinstance(child, nn.Conv2d):
                        last_conv = child
                    else:
                        found = find_last_conv(child)
                        if found:
                            last_conv = found
                return last_conv
            
            last_conv = find_last_conv(model)
            if last_conv:
                target_layers = [last_conv]
                print(f"警告: 自动找到最后一个卷积层: {last_conv}")
            else:
                raise ValueError(f"无法自动找到目标层，请手动指定")
        
    except Exception as e:
        print(f"警告: 获取目标层时出错: {e}")
        raise
    
    return target_layers


def show_cam_on_image(img, mask, colormap=cv2.COLORMAP_JET, alpha=0.4):
    """
    在图像上叠加CAM热力图
    
    Args:
        img: 原始图像 [H, W, 3] (0-255)
        mask: CAM掩码 [H, W] (0-1)
        colormap: OpenCV颜色映射
        alpha: 叠加透明度
    
    Returns:
        result: 叠加后的图像
    """
    # 确保图像在正确范围
    if img.max() > 1:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    
    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    
    # 叠加
    result = (1 - alpha) * img + alpha * heatmap
    result = np.clip(result, 0, 1)
    
    return result


def visualize_gradcam_single_image(
    model,
    image_path,
    target_layers,
    target_category=None,
    img_size=224,
    save_path=None,
    use_cuda=True
):
    """
    对单张图像生成Grad-CAM可视化
    
    Args:
        model: 模型
        image_path: 图像路径
        target_layers: 目标层
        target_category: 目标类别
        img_size: 图像大小
        save_path: 保存路径
        use_cuda: 是否使用CUDA
    
    Returns:
        cam_image: CAM可视化图像
    """
    # 加载和预处理图像
    img = Image.open(image_path).convert('RGB')
    original_img = np.array(img)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0)
    
    # 创建Grad-CAM
    gradcam = GradCAM(model, target_layers, use_cuda=use_cuda)
    
    # 生成CAM
    cam = gradcam(input_tensor, target_category=target_category)
    
    # 调整原始图像大小
    img_resized = cv2.resize(original_img, (img_size, img_size))
    img_resized = img_resized.astype(np.float32) / 255.0
    
    # 叠加CAM
    cam_image = show_cam_on_image(img_resized, cam)
    
    # 保存
    if save_path:
        plt.figure(figsize=(12, 5))
        
        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(img_resized)
        plt.title('Original Image', fontsize=12)
        plt.axis('off')
        
        # 热力图
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title('Grad-CAM Heatmap', fontsize=12)
        plt.axis('off')
        plt.colorbar()
        
        # 叠加图像
        plt.subplot(1, 3, 3)
        plt.imshow(cam_image)
        plt.title('Overlay', fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 也保存单独的叠加图像
        cam_image_uint8 = (cam_image * 255).astype(np.uint8)
        cv2.imwrite(save_path.replace('.png', '_overlay.png'), 
                   cv2.cvtColor(cam_image_uint8, cv2.COLOR_RGB2BGR))
    
    return cam_image


def visualize_gradcam_batch(
    model,
    dataloader,
    target_layers,
    class_names,
    output_dir,
    num_samples=None,
    use_cuda=True
):
    """
    批量生成Grad-CAM可视化
    
    Args:
        model: 模型
        dataloader: 数据加载器
        target_layers: 目标层
        class_names: 类别名称列表
        output_dir: 输出目录
        num_samples: 处理的样本数（None表示全部）
        use_cuda: 是否使用CUDA
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建Grad-CAM
    gradcam = GradCAM(model, target_layers, use_cuda=use_cuda)
    
    model.eval()
    processed = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc='生成Grad-CAM')):
        if num_samples and processed >= num_samples:
            break
        
        if use_cuda:
            images = images.cuda()
        
        # 获取预测（需要梯度）
        images.requires_grad = True
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1).detach()
        
        # 对每个样本生成CAM
        for i in range(images.size(0)):
            if num_samples and processed >= num_samples:
                break
            
            image = images[i:i+1].clone()
            image.requires_grad = True
            label = labels[i].item()
            pred = predictions[i].item()
            
            # 生成CAM（使用预测类别）
            cam = gradcam(image, target_category=[pred])
            
            # 确保cam是2D数组
            if cam.ndim == 3:
                cam = cam[0]
            
            # 获取原始图像（反归一化）
            img_tensor = image[0].detach().cpu()
            img = img_tensor.clone()
            for t, m, s in zip(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
                t.mul_(s).add_(m)
            img = torch.clamp(img, 0, 1)
            img_np = img.permute(1, 2, 0).numpy()
            
            # 叠加CAM
            cam_image = show_cam_on_image(img_np, cam)
            
            # 保存
            class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
            pred_name = class_names[pred] if pred < len(class_names) else f'Class_{pred}'
            
            save_name = f"gradcam_{processed:04d}_true_{class_name}_pred_{pred_name}.png"
            save_path = os.path.join(output_dir, save_name)
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img_np)
            axes[0].set_title(f'Original\nTrue: {class_name}', fontsize=10)
            axes[0].axis('off')
            
            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap', fontsize=10)
            axes[1].axis('off')
            
            axes[2].imshow(cam_image)
            axes[2].set_title(f'Overlay\nPred: {pred_name}', fontsize=10)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            processed += 1
    
    print(f"\n已生成 {processed} 个Grad-CAM可视化，保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM可视化脚本')
    
    # 模型相关
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--model-name', type=str, default=None,
                        help='模型名称（如果无法从checkpoint获取）')
    
    # 数据相关
    parser.add_argument('--image-path', type=str, default=None,
                        help='单张图像路径（单图像模式）')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据目录（批量模式）')
    parser.add_argument('--img-size', type=int, default=224,
                        help='图像大小')
    
    # 输出相关
    parser.add_argument('--output-dir', type=str, default='gradcam_results',
                        help='输出目录')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='处理的样本数（批量模式）')
    
    # 其他
    parser.add_argument('--target-category', type=int, default=None,
                        help='目标类别索引（None表示使用预测类别）')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批次大小（批量模式）')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用CPU')
    
    args = parser.parse_args()
    
    # 设置设备
    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 获取模型信息
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    else:
        model_name = args.model_name
        if model_name is None:
            raise ValueError("无法确定模型名称，请使用--model-name指定")
    
    if 'class_to_idx' in checkpoint:
        class_to_idx = checkpoint['class_to_idx']
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = sorted(class_to_idx.keys())
        num_classes = len(class_names)
    else:
        num_classes = 9  # 默认值
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    print(f"模型: {model_name}")
    print(f"类别数: {num_classes}")
    print(f"类别: {class_names}")
    
    # 创建模型
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 获取目标层
    print("查找目标层...")
    try:
        target_layers = get_target_layers(model, model_name)
        print(f"目标层: {target_layers}")
    except Exception as e:
        print(f"错误: {e}")
        print("请手动指定目标层")
        return
    
    # 单图像模式
    if args.image_path:
        print(f"\n处理单张图像: {args.image_path}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        save_path = os.path.join(
            args.output_dir,
            f"gradcam_{os.path.basename(args.image_path)}"
        )
        
        visualize_gradcam_single_image(
            model=model,
            image_path=args.image_path,
            target_layers=target_layers,
            target_category=args.target_category,
            img_size=args.img_size,
            save_path=save_path,
            use_cuda=use_cuda
        )
        
        print(f"结果已保存到: {save_path}")
    
    # 批量模式
    elif args.data_dir:
        print(f"\n批量处理: {args.data_dir}")
        
        # 数据加载
        _, val_transform = get_data_augmentation('none', img_size=args.img_size)
        dataset = ImageFolderDataset(args.data_dir, transform=val_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        visualize_gradcam_batch(
            model=model,
            dataloader=dataloader,
            target_layers=target_layers,
            class_names=class_names,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            use_cuda=use_cuda
        )
    
    else:
        print("错误: 请指定--image-path（单图像模式）或--data-dir（批量模式）")


if __name__ == '__main__':
    main()

