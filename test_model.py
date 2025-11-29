# 引入必要的库
import torch
import sys
import os # 导入 os 库以便更清晰地处理路径

# 设置路径（假设您的模型文件位于这个路径下）
sys.path.insert(0, '/home/ln/wangweicheng/ModelsTotrain')

# --- GPU/CUDA 检测与设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("检查 starnet_dual_pyramid_rcf 模型")
print(f"🚀 当前运行设备: {device}")
if device.type == 'cuda':
    print(f"   CUDA 设备名称: {torch.cuda.get_device_name(0)}")
print("=" * 80)

try:
    # 假设您的模型文件名为 starnet_dual_pyramid_rcf.py 且已在路径中
    from models.starnet_dual_pyramid_rcf import starnet_dual_pyramid_rcf, StarNet_DualPyramid_RCF
    
    # 1. 测试模型创建并移动到设备
    print("\n1. 测试模型创建...")
    # 实例化模型，并使用 .to(device) 将其移动到 GPU 或 CPU
    model = starnet_dual_pyramid_rcf(num_classes=9).to(device)
    print("   ✓ 模型创建成功并已移动到设备")
    
    # 2. 检查模型结构 (保持不变)
    print("\n2. 检查模型结构...")
    print(f"   - Local Pyramid downsamples: {len(model.local.downsamples)}")
    print(f"   - Local Pyramid blocks_list: {len(model.local.blocks_list)}")
    print(f"   - Global Pyramid stages: {len(model.global_pyr.stages)}")
    print(f"   - Adapters: {len(model.adapters)}")
    print(f"   - Fuse weights: {len(model.fuse_weights)}")
    print(f"   - Gamma weights: {len(model.gamma_weights)}")
    
    # 3. 测试前向传播
    print("\n3. 测试前向传播...")
    # 创建输入张量，并使用 .to(device) 将其移动到与模型相同的设备
    x = torch.randn(2, 3, 224, 224, device=device) # 直接在设备上创建张量
    
    # 确保模型处于评估模式（虽然测试时影响不大，但通常是好习惯）
    model.eval() 
    
    # 使用 torch.no_grad() 包裹，避免在测试时计算梯度，节省内存和时间
    with torch.no_grad():
        output = model(x)
        
    print(f"   ✓ 前向传播成功")
    # 检查输出形状，输出也应该在 device 上
    print(f"   - 输出形状: {output.shape}")
    print(f"   - 输出设备: {output.device}")
    
    # 4. 检查参数量 (保持不变)
    print("\n4. 检查参数量...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - 总参数量: {total_params / 1e6:.2f}M")
    print(f"   - 可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 5. 测试梯度反向传播
    print("\n5. 测试梯度反向传播...")
    model.train() # 切换回训练模式
    
    # 重新进行前向传播以计算梯度 (因为第3步使用了 torch.no_grad())
    x = torch.randn(2, 3, 224, 224, device=device)
    output = model(x)
    
    # 损失函数和标签也要移动到设备
    criterion = torch.nn.CrossEntropyLoss()
    labels = torch.randint(0, 9, (2,)).to(device)
    
    loss = criterion(output, labels)
    loss.backward()
    print(f"   ✓ 反向传播成功 (Loss: {loss.item():.4f})")
    
    print("\n" + "=" * 80)
    print("✅ 模型可以正常加载和使用！")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)