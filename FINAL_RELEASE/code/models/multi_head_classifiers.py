"""
多分类头实现：ArcFace, CosFace, LDAM, Softmax
用于单阶段训练多个分类头，共享 Backbone 特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ArcFace(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels=None):
        # 归一化特征和权重
        features_norm = F.normalize(features, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        
        # 计算余弦相似度
        cosine = F.linear(features_norm, weight_norm)
        
        if labels is None:
            return self.s * cosine
        
        # 计算角度
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # 添加 margin
        target_theta = theta[torch.arange(0, features.size(0)), labels]
        theta[torch.arange(0, features.size(0)), labels] = target_theta + self.m
        
        # 计算 logits
        logits = self.s * torch.cos(theta)
        
        return logits


class CosFace(nn.Module):
    """
    CosFace: Large Margin Cosine Loss
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels=None):
        # 归一化特征和权重
        features_norm = F.normalize(features, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        
        # 计算余弦相似度
        cosine = F.linear(features_norm, weight_norm)
        
        if labels is None:
            return self.s * cosine
        
        # 创建副本并添加 margin
        phi = cosine.clone()
        phi[torch.arange(0, features.size(0)), labels] = cosine[torch.arange(0, features.size(0)), labels] - self.m
        
        # 计算 logits
        logits = self.s * phi
        
        return logits


class LDAM_CB_Loss(nn.Module):
    """
    LDAM (Label Distribution Aware Margin) + Class-Balanced Loss
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30.0, beta=0.9999):
        super(LDAM_CB_Loss, self).__init__()
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        
        # LDAM: 计算每个类别的 margin
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list + 1e-12))
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer('m_list', torch.tensor(m_list, dtype=torch.float32))
        
        # Class-Balanced: 计算每个类别的权重
        effective_num = 1.0 - np.power(beta, cls_num_list)
        weights = (1.0 - beta) / (effective_num + 1e-12)
        weights = weights / np.sum(weights) * len(cls_num_list)
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
        self.s = s
        self.max_m = max_m
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_classes] - 未缩放的 logits
            targets: [B] - 类别标签
        """
        # 应用 LDAM margin
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = logits - batch_m * self.s
        
        # 计算交叉熵
        output = torch.where(index, x_m, logits)
        
        # 应用 Class-Balanced 权重
        loss = F.cross_entropy(output, targets, weight=self.weights, reduction='none')
        
        return loss.mean()

