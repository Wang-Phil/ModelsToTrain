"""
测试评估脚本的导入和基本功能
"""

import sys

def test_imports():
    """测试所有必要的导入"""
    print("测试导入...")
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import torchvision
        print("✓ torchvision")
    except ImportError as e:
        print(f"✗ torchvision: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn")
    except ImportError as e:
        print(f"✗ seaborn: {e}")
        return False
    
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
        print("✓ sklearn")
    except ImportError as e:
        print(f"✗ sklearn: {e}")
        return False
    
    try:
        from evaluate_model import (
            load_model, evaluate_model, calculate_metrics,
            plot_confusion_matrix, plot_roc_curves, plot_per_class_metrics
        )
        print("✓ evaluate_model")
    except ImportError as e:
        print(f"✗ evaluate_model: {e}")
        return False
    
    return True


def test_metrics_calculation():
    """测试指标计算功能"""
    print("\n测试指标计算...")
    
    try:
        from evaluate_model import calculate_metrics
        import numpy as np
        
        # 创建模拟数据
        num_classes = 3
        n_samples = 100
        
        y_true = np.random.randint(0, num_classes, n_samples)
        y_pred = y_true.copy()  # 完美预测
        y_probs = np.random.rand(n_samples, num_classes)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # 归一化
        
        class_names = [f'Class_{i}' for i in range(num_classes)]
        
        metrics = calculate_metrics(y_true, y_pred, y_probs, num_classes, class_names)
        
        # 检查关键指标是否存在
        required_keys = [
            'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
            'confusion_matrix', 'per_class_precision', 'per_class_recall',
            'per_class_f1', 'roc_auc_micro', 'roc_auc_macro'
        ]
        
        for key in required_keys:
            if key not in metrics:
                print(f"✗ 缺少指标: {key}")
                return False
        
        print("✓ 所有指标计算正常")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  加权F1: {metrics['f1_weighted']:.4f}")
        print(f"  ROC AUC (微平均): {metrics['roc_auc_micro']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 指标计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("评估脚本功能测试")
    print("=" * 60)
    
    success = True
    success &= test_imports()
    success &= test_metrics_calculation()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败，请检查错误信息")
    print("=" * 60)

