"""
CLIP模型Grad-CAM可视化脚本
支持可视化图像中哪些区域与特定文本描述最相关
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
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

# 导入CLIP模型
from models.clip import CLIPModel, ImageEncoder, TextEncoder

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CLIPGradCAM:
    """CLIP模型的Grad-CAM实现"""
    
    def __init__(self, model: CLIPModel, target_layer, use_cuda=True):
        """
        Args:
            model: CLIP模型实例
            target_layer: 目标层（通常是图像编码器backbone的最后一层）
            use_cuda: 是否使用CUDA
        """
        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            self.model = self.model.cuda()
        
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            # 保存激活值
            if isinstance(output, torch.Tensor):
                self.activations = output.detach()
            elif isinstance(output, tuple):
                self.activations = output[0].detach()
        
        def backward_hook(module, grad_input, grad_output):
            # 保存梯度
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        # 注册钩子
        if hasattr(self.target_layer, 'register_forward_hook'):
            self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
            if hasattr(self.target_layer, 'register_full_backward_hook'):
                self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
            else:
                self.hooks.append(self.target_layer.register_backward_hook(backward_hook))
    
    def _release_hooks(self):
        """释放钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _generate_cam(self, activations, gradients):
        """生成CAM热力图"""
        if gradients is None or activations is None:
            raise ValueError("Gradients or activations are None")
        
        # 计算权重（梯度的全局平均池化）
        # 处理不同维度的激活和梯度
        if len(activations.shape) == 4:  # [B, C, H, W]
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
        elif len(activations.shape) == 3:  # [B, N, C] or [B, H*W, C]
            # 对于序列输出，计算全局平均
            weights = torch.mean(gradients, dim=1, keepdim=True)
            cam = torch.sum(weights * activations, dim=-1, keepdim=False)
            # 如果是空间序列，可能需要reshape
            if cam.ndim == 2:
                # 尝试reshape为空间维度（假设是正方形）
                sqrt_size = int(np.sqrt(cam.shape[1]))
                if sqrt_size * sqrt_size == cam.shape[1]:
                    cam = cam.view(cam.shape[0], sqrt_size, sqrt_size)
                else:
                    # 无法reshape，返回1D
                    cam = cam.mean(dim=1, keepdim=True).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")
        
        # ReLU激活
        cam = F.relu(cam)
        
        # 归一化到[0, 1]
        if cam.ndim == 4:
            cam = cam.squeeze(1)
        
        # 对每个样本归一化
        if cam.ndim == 3:  # [B, H, W]
            for i in range(cam.shape[0]):
                cam_i = cam[i]
                cam_i = cam_i - cam_i.min()
                cam_i = cam_i / (cam_i.max() + 1e-8)
                cam[i] = cam_i
        elif cam.ndim == 2:  # [H, W]
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
        else:
            # 如果维度不对，创建一个简单的热力图
            cam = torch.ones(1, 224, 224) * 0.5
        
        return cam.cpu().numpy()
    
    def __call__(self, images, text_features, target_text_idx=None):
        """
        生成Grad-CAM热力图
        
        Args:
            images: 输入图像 [B, C, H, W]
            text_features: 文本特征 [num_texts, embed_dim] 或 [B, embed_dim]
            target_text_idx: 目标文本索引（用于多文本情况），如果为None则使用最相似的文本
        
        Returns:
            cam: CAM热力图 [B, H, W] 或 [H, W] (如果batch_size=1)
        """
        self.gradients = None
        self.activations = None
        
        # 确保输入在正确的设备上
        device = next(self.model.parameters()).device
        images = images.to(device)
        if isinstance(text_features, torch.Tensor):
            text_features = text_features.to(device)
        
        images.requires_grad = True
        
        # 前向传播获取图像特征
        image_features = self.model.image_encoder(images)
        
        # 如果是多个文本特征，选择目标文本
        if text_features.ndim == 2 and text_features.shape[0] > 1:
            if target_text_idx is None:
                # 计算所有文本的相似度，选择最相似的
                similarity = image_features @ text_features.T  # [B, num_texts]
                target_text_idx = torch.argmax(similarity, dim=1)  # [B]
                target_text_features = text_features[target_text_idx]  # [B, embed_dim]
            else:
                if isinstance(target_text_idx, int):
                    target_text_features = text_features[target_text_idx:target_text_idx+1]
                else:
                    target_text_features = text_features[target_text_idx]
        else:
            target_text_features = text_features
        
        # 计算相似度作为损失（我们想要最大化这个相似度）
        similarity = (image_features * target_text_features).sum(dim=1)  # [B]
        loss = similarity.sum()
        
        # 反向传播
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        
        # 生成CAM
        if self.gradients is None or self.activations is None:
            raise ValueError("未能捕获梯度或激活值。请检查目标层是否正确注册了钩子。")
        
        cam = self._generate_cam(self.activations, self.gradients)
        
        # 调整大小到输入图像大小
        img_h, img_w = images.size(2), images.size(3)
        if cam.ndim == 2:  # [H, W]
            cam_resized = cv2.resize(cam, (img_w, img_h))
        elif cam.ndim == 3:  # [B, H, W]
            cam_resized = np.array([
                cv2.resize(cam[i], (img_w, img_h))
                for i in range(cam.shape[0])
            ])
        else:
            # 如果维度不对，创建一个默认热力图
            cam_resized = np.ones((img_h, img_w)) * 0.5
        
        # 如果batch_size=1，返回2D数组
        if cam_resized.ndim == 3 and cam_resized.shape[0] == 1:
            cam_resized = cam_resized[0]
        
        return cam_resized
    
    def __del__(self):
        self._release_hooks()


def get_target_layer_for_clip(model: CLIPModel, image_encoder_name: str):
    """
    根据图像编码器类型自动获取目标层
    
    Args:
        model: CLIP模型实例
        image_encoder_name: 图像编码器名称
    
    Returns:
        target_layer: 目标层
    """
    image_encoder = model.image_encoder
    backbone = image_encoder.backbone
    
    image_encoder_name = image_encoder_name.lower()
    
    try:
        # ResNet系列
        if 'resnet' in image_encoder_name:
            if hasattr(backbone, 'layer4'):
                return backbone.layer4[-1]
            elif hasattr(backbone, 'model'):
                # 如果是Sequential包装的
                if hasattr(backbone.model, 'layer4'):
                    return backbone.model.layer4[-1]
        
        # StarNet Dual-Pyramid RCF
        elif 'starnet_dual_pyramid_rcf' in image_encoder_name:
            if hasattr(backbone, 'model'):
                # 获取最后一个stage的最后一个block
                if hasattr(backbone.model, 'local'):
                    if hasattr(backbone.model.local, 'blocks_list'):
                        # blocks_list是一个列表，每个元素是一个stage的block列表
                        last_stage_blocks = backbone.model.local.blocks_list[-1]
                        if isinstance(last_stage_blocks, (list, nn.ModuleList)):
                            return last_stage_blocks[-1]
                        else:
                            return last_stage_blocks
                    elif hasattr(backbone.model.local, 'stages'):
                        return backbone.model.local.stages[-1][-1]
                # 如果找不到，尝试使用norm层之前的部分
                elif hasattr(backbone.model, 'norm'):
                    # 获取norm层之前的最后一个卷积层
                    return backbone.model.norm
        
        # 标准 StarNet 系列
        elif 'starnet' in image_encoder_name:
            if hasattr(backbone, 'stages'):
                return backbone.stages[-1][-1]
            elif hasattr(backbone, 'model'):
                if hasattr(backbone.model, 'stages'):
                    return backbone.model.stages[-1][-1]
        
        # ConvNeXt系列
        elif 'convnext' in image_encoder_name:
            if hasattr(backbone, 'stages'):
                return backbone.stages[-1][-1]
            elif hasattr(backbone, 'model'):
                if hasattr(backbone.model, 'stages'):
                    return backbone.model.stages[-1][-1]
        
        # EfficientNet系列
        elif 'efficientnet' in image_encoder_name:
            if hasattr(backbone, 'blocks'):
                return backbone.blocks[-1]
            elif hasattr(backbone, 'features'):
                return backbone.features[-1]
        
        # ViT
        elif image_encoder_name == 'vit':
            if hasattr(backbone, 'encoder'):
                return backbone.encoder.layer[-1]
            elif hasattr(backbone, 'layers'):
                return backbone.layers[-1]
        
        # 如果找不到，尝试查找最后一个卷积层
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
        
        last_conv = find_last_conv(backbone)
        if last_conv:
            print(f"警告: 自动找到最后一个卷积层: {last_conv}")
            return last_conv
        
        # 如果都找不到，返回backbone本身（可能会影响性能）
        print(f"警告: 无法找到合适的目标层，使用整个backbone")
        return backbone
        
    except Exception as e:
        print(f"警告: 获取目标层时出错: {e}")
        # 返回backbone作为后备
        return backbone


def show_cam_on_image(img, mask, colormap=cv2.COLORMAP_JET, alpha=0.4):
    """
    在图像上叠加CAM热力图
    
    Args:
        img: 原始图像 [H, W, 3] (0-255 或 0-1)
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


def visualize_clip_gradcam(
    model: CLIPModel,
    image_path: str,
    class_texts: List[str],
    target_layer=None,
    target_text_idx: int = None,
    img_size: int = 224,
    save_path: str = None,
    use_cuda: bool = True,
    image_encoder_name: str = None
):
    """
    对单张图像生成CLIP Grad-CAM可视化
    
    Args:
        model: CLIP模型
        image_path: 图像路径
        class_texts: 类别文本描述列表
        target_layer: 目标层（如果为None则自动查找）
        target_text_idx: 目标文本索引（用于可视化特定类别的热力图）
        img_size: 图像大小
        save_path: 保存路径
        use_cuda: 是否使用CUDA
        image_encoder_name: 图像编码器名称（用于自动查找目标层）
    
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
    
    # 确保输入tensor在正确的设备上
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 编码文本
    model.eval()
    with torch.no_grad():
        text_features = model.text_encoder(texts=class_texts)  # [num_classes, embed_dim]
        if isinstance(text_features, torch.Tensor):
            text_features = text_features.to(device)
    
    # 获取目标层
    if target_layer is None:
        if image_encoder_name is None:
            image_encoder_name = model.image_encoder.model_name
        target_layer = get_target_layer_for_clip(model, image_encoder_name)
        print(f"自动找到目标层: {target_layer}")
    
    # 创建Grad-CAM
    gradcam = CLIPGradCAM(model, target_layer, use_cuda=use_cuda)
    
    # 如果没有指定目标文本，对所有文本生成热力图
    if target_text_idx is None:
        # 生成所有类别 heatmap 中相似度最高的那个
        cam = gradcam(input_tensor, text_features, target_text_idx=None)
        # 获取最相似的类别
        with torch.no_grad():
            image_features = model.image_encoder(input_tensor)
            similarity = model.compute_similarity(image_features, text_features)
            best_match_idx = torch.argmax(similarity, dim=1).item()
    else:
        cam = gradcam(input_tensor, text_features, target_text_idx=target_text_idx)
        best_match_idx = target_text_idx
    
    # 调整原始图像大小
    img_resized = cv2.resize(original_img, (img_size, img_size))
    img_resized = img_resized.astype(np.float32) / 255.0
    
    # 叠加CAM
    cam_image = show_cam_on_image(img_resized, cam)
    
    # 获取预测结果
    with torch.no_grad():
        predictions, probabilities = model.predict(input_tensor, class_texts)
        pred_idx = predictions[0].item()
        probs = probabilities[0].cpu().numpy()
    
    # 保存
    if save_path:
        # 创建多子图可视化
        fig = plt.figure(figsize=(16, 5))
        
        # 原始图像
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(img_resized)
        ax1.set_title('原始图像', fontsize=12)
        ax1.axis('off')
        
        # 热力图
        ax2 = plt.subplot(1, 4, 2)
        im = ax2.imshow(cam, cmap='jet')
        ax2.set_title(f'Grad-CAM热力图\n(目标: {class_texts[best_match_idx]})', fontsize=12)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046)
        
        # 叠加图像
        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(cam_image)
        ax3.set_title('叠加结果', fontsize=12)
        ax3.axis('off')
        
        # 预测概率（如果类别数不太多）
        ax4 = plt.subplot(1, 4, 4)
        if len(class_texts) <= 10:
            y_pos = np.arange(len(class_texts))
            colors = ['red' if i == pred_idx else 'blue' for i in range(len(class_texts))]
            ax4.barh(y_pos, probs, color=colors)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([txt[:20] + '...' if len(txt) > 20 else txt for txt in class_texts], fontsize=8)
            ax4.set_xlabel('概率')
            ax4.set_title('预测概率')
            ax4.invert_yaxis()
        else:
            # 如果类别太多，只显示top-5
            top5_indices = np.argsort(probs)[-5:][::-1]
            top5_probs = probs[top5_indices]
            y_pos = np.arange(5)
            colors = ['red' if i == pred_idx else 'blue' for i in top5_indices]
            ax4.barh(y_pos, top5_probs, color=colors)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([class_texts[i][:20] + '...' if len(class_texts[i]) > 20 else class_texts[i] 
                                 for i in top5_indices], fontsize=8)
            ax4.set_xlabel('概率')
            ax4.set_title('Top-5 预测概率')
            ax4.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 也保存单独的叠加图像
        cam_image_uint8 = (cam_image * 255).astype(np.uint8)
        overlay_path = save_path.replace('.png', '_overlay.png').replace('.jpg', '_overlay.jpg')
        cv2.imwrite(overlay_path, cv2.cvtColor(cam_image_uint8, cv2.COLOR_RGB2BGR))
        print(f"结果已保存到: {save_path}")
        print(f"叠加图像已保存到: {overlay_path}")
    
    return cam_image


def main():
    parser = argparse.ArgumentParser(description='CLIP模型Grad-CAM可视化脚本')
    
    # 模型相关
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--image-encoder', type=str, default=None,
                        help='图像编码器名称（如果无法从checkpoint获取）')
    parser.add_argument('--text-encoder', type=str, default='bert-base-chinese',
                        help='文本编码器名称')
    parser.add_argument('--embed-dim', type=int, default=512,
                        help='嵌入维度')
    
    # 数据相关
    parser.add_argument('--image-path', type=str, required=True,
                        help='图像路径')
    parser.add_argument('--class-texts', type=str, nargs='+', default=None,
                        help='类别文本描述列表（例如: "类别1" "类别2"）')
    parser.add_argument('--class-texts-file', type=str, default=None,
                        help='类别文本描述JSON文件路径')
    parser.add_argument('--target-text-idx', type=int, default=None,
                        help='目标文本索引（用于可视化特定类别的热力图，None表示使用最相似的）')
    parser.add_argument('--img-size', type=int, default=224,
                        help='图像大小')
    
    # 输出相关
    parser.add_argument('--output-dir', type=str, default='clip_gradcam_results',
                        help='输出目录')
    parser.add_argument('--output-name', type=str, default=None,
                        help='输出文件名（如果不指定则自动生成）')
    
    # 其他
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
    
    # 获取模型配置
    if 'image_encoder' in checkpoint:
        image_encoder_name = checkpoint['image_encoder']
    else:
        image_encoder_name = args.image_encoder
        if image_encoder_name is None:
            raise ValueError("无法确定图像编码器名称，请使用--image-encoder指定")
    
    if 'text_encoder' in checkpoint:
        text_encoder_name = checkpoint['text_encoder']
    else:
        text_encoder_name = args.text_encoder
    
    if 'embed_dim' in checkpoint:
        embed_dim = checkpoint['embed_dim']
    else:
        embed_dim = args.embed_dim
    
    print(f"图像编码器: {image_encoder_name}")
    print(f"文本编码器: {text_encoder_name}")
    print(f"嵌入维度: {embed_dim}")
    
    # 创建模型
    model = CLIPModel(
        image_encoder_name=image_encoder_name,
        text_encoder_name=text_encoder_name,
        embed_dim=embed_dim
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # 加载类别文本描述
    if args.class_texts_file:
        with open(args.class_texts_file, 'r', encoding='utf-8') as f:
            class_texts_dict = json.load(f)
        class_texts = list(class_texts_dict.values())
        print(f"从文件加载类别文本: {args.class_texts_file}")
        print(f"类别数: {len(class_texts)}")
    elif args.class_texts:
        class_texts = args.class_texts
        print(f"使用命令行指定的类别文本: {class_texts}")
    else:
        raise ValueError("请指定--class-texts或--class-texts-file")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成输出文件名
    if args.output_name:
        save_path = os.path.join(args.output_dir, args.output_name)
    else:
        image_name = Path(args.image_path).stem
        save_path = os.path.join(args.output_dir, f"clip_gradcam_{image_name}.png")
    
    # 生成可视化
    print(f"\n处理图像: {args.image_path}")
    visualize_clip_gradcam(
        model=model,
        image_path=args.image_path,
        class_texts=class_texts,
        target_layer=None,
        target_text_idx=args.target_text_idx,
        img_size=args.img_size,
        save_path=save_path,
        use_cuda=use_cuda,
        image_encoder_name=image_encoder_name
    )
    
    print(f"\n可视化完成！")


if __name__ == '__main__':
    main()

