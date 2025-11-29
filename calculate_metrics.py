"""
计算模型评估指标：mAP, Precision, Recall, F1, Params, FLOPs
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# thop是可选的，用于计算FLOPs
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def extract_logits(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    if hasattr(outputs, 'logits'):
        return outputs.logits
    return outputs


def calculate_classification_metrics(y_true, y_pred, num_classes):
    """
    计算分类任务的各项指标
    
    Args:
        y_true: 真实标签 [N]
        y_pred: 预测标签 [N]
        num_classes: 类别数
    
    Returns:
        dict: 包含各项指标的字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 计算每个类别的精确率、召回率、F1分数
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average=None, zero_division=0
    )
    
    # 宏平均（Macro Average）：对所有类别求平均
    precision_macro = np.mean(precision_per_class)
    recall_macro = np.mean(recall_per_class)
    f1_macro = np.mean(f1_per_class)
    
    # 微平均（Micro Average）：将所有类别的TP、FP、FN加起来计算
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # 加权平均（Weighted Average）：按样本数加权
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # mAP (mean Average Precision): 对于多分类任务，通常使用宏平均精确率
    mAP = precision_macro
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # 计算总体准确率
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    metrics = {
        'mAP': float(mAP * 100),  # 转换为百分比
        'precision_macro': float(precision_macro * 100),
        'recall_macro': float(recall_macro * 100),
        'f1_macro': float(f1_macro * 100),
        'precision_micro': float(precision_micro * 100),
        'recall_micro': float(recall_micro * 100),
        'f1_micro': float(f1_micro * 100),
        'precision_weighted': float(precision_weighted * 100),
        'recall_weighted': float(recall_weighted * 100),
        'f1_weighted': float(f1_weighted * 100),
        'accuracy': float(accuracy * 100),
        'precision_per_class': [float(p * 100) for p in precision_per_class],
        'recall_per_class': [float(r * 100) for r in recall_per_class],
        'f1_per_class': [float(f * 100) for f in f1_per_class],
        'support_per_class': support.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def calculate_model_complexity(model, input_size=(1, 3, 224, 224), device='cpu'):
    """
    计算模型的参数量和FLOPs
    
    Args:
        model: PyTorch模型
        input_size: 输入尺寸 (batch_size, channels, height, width)
        device: 设备
    
    Returns:
        dict: 包含参数量和FLOPs的字典
    """
    model = model.to(device)
    
    # 确保模型和所有子模块都处于eval模式
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.Dropout, nn.Dropout2d)):
            module.eval()
    
    # 创建虚拟输入（使用batch_size=2避免BatchNorm在batch_size=1时的问题）
    # 但保持原始batch_size用于参数量计算
    batch_size, channels, height, width = input_size
    dummy_input = torch.randn((max(2, batch_size), channels, height, width)).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算FLOPs（使用thop库）
    if THOP_AVAILABLE:
        try:
            # 使用torch.no_grad()确保所有操作都在推理模式下
            with torch.no_grad():
                # 检查模型forward签名，看是否需要额外参数
                import inspect
                sig = inspect.signature(model.forward)
                forward_params = list(sig.parameters.keys())
                
                # 如果forward方法需要epoch参数，传递epoch=0
                if 'epoch' in forward_params:
                    flops, _ = profile(model, inputs=(dummy_input, 0), verbose=False)
                else:
                    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
                
                flops_millions = flops / 1e6
                # 如果使用了batch_size=2，需要除以2来得到单个样本的FLOPs
                if batch_size == 1:
                    flops_millions = flops_millions / 2
        except Exception as e:
            print(f"警告: 无法计算FLOPs: {e}")
            flops_millions = 0.0
    else:
        flops_millions = 0.0
    
    # 转换为百万单位
    params_millions = total_params / 1e6
    
    complexity = {
        'params_total': int(total_params),
        'params_trainable': int(trainable_params),
        'params_millions': float(params_millions),
        'flops_millions': float(flops_millions)
    }
    
    return complexity


def evaluate_model_comprehensive(model, dataloader, criterion, device, num_classes, class_names=None):
    """
    全面评估模型，返回所有指标
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        num_classes: 类别数
        class_names: 类别名称列表（可选）
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = extract_logits(model(images))
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # 获取预测结果
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算损失
    avg_loss = running_loss / len(dataloader)
    
    # 计算分类指标
    metrics = calculate_classification_metrics(
        np.array(all_labels),
        np.array(all_preds),
        num_classes
    )
    
    # 添加损失
    metrics['loss'] = float(avg_loss)
    
    # 添加类别名称（如果有）
    if class_names:
        metrics['class_names'] = class_names
        # 为每个类别添加详细指标
        metrics['per_class_details'] = []
        for i, class_name in enumerate(class_names):
            metrics['per_class_details'].append({
                'class_name': class_name,
                'precision': metrics['precision_per_class'][i],
                'recall': metrics['recall_per_class'][i],
                'f1': metrics['f1_per_class'][i],
                'support': metrics['support_per_class'][i]
            })
    
    return metrics

