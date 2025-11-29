"""
计算图像数据集的均值和标准差（归一化参数）

使用方法：
    python mean_std.py --data_dir data/train
"""

from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
import argparse
from tqdm import tqdm
import os


def get_mean_and_std(dataset, batch_size=64, num_workers=4, device=None):
    """
    计算数据集的均值和标准差
    
    Args:
        dataset: ImageFolder数据集
        batch_size: 批次大小
        num_workers: 数据加载的worker数量
        device: 计算设备（'cpu'或'cuda'或torch.device），None时自动选择
    
    Returns:
        mean: 三个通道的均值 [R, G, B]
        std: 三个通道的标准差 [R, G, B]
    """
    # 自动选择设备
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"使用GPU计算: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("使用CPU计算")
    else:
        if isinstance(device, str):
            device = torch.device(device)
        print(f"使用设备: {device}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')  # GPU时启用pin_memory
    )
    
    # 初始化累加器（在指定设备上）
    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)
    total_samples = 0
    
    print("计算均值...")
    # 第一步：计算均值
    for images, _ in tqdm(dataloader, desc="Processing"):
        # images shape: [batch_size, 3, H, W]
        batch_samples = images.size(0)
        # 将图像移到指定设备
        images = images.to(device)
        # 将图像reshape为 [batch_size, 3, H*W]
        images = images.view(batch_samples, images.size(1), -1)
        # 计算每个通道的均值并累加
        mean += images.mean(2).sum(0)
        total_samples += batch_samples
    
    # 计算总体均值
    mean /= total_samples
    
    print("计算标准差...")
    # 第二步：计算标准差（基于均值）
    for images, _ in tqdm(dataloader, desc="Processing"):
        batch_samples = images.size(0)
        # 将图像移到指定设备
        images = images.to(device)
        images = images.view(batch_samples, images.size(1), -1)
        # 计算每个通道的方差并累加
        std += ((images - mean.unsqueeze(0).unsqueeze(2)) ** 2).mean(2).sum(0)
    
    # 计算总体标准差
    std = torch.sqrt(std / total_samples)
    
    # 移回CPU并转换为列表
    return mean.cpu().tolist(), std.cpu().tolist()


def get_mean_and_std_accurate(dataset, batch_size=64, num_workers=4, device=None):
    """
    更精确的计算方法：使用Welford's online algorithm
    适合大数据集，内存占用更小（单遍扫描）
    
    Args:
        dataset: ImageFolder数据集
        batch_size: 批次大小
        num_workers: 数据加载的worker数量
        device: 计算设备（'cpu'或'cuda'或torch.device），None时自动选择
    """
    # 自动选择设备
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"使用GPU计算: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("使用CPU计算")
    else:
        if isinstance(device, str):
            device = torch.device(device)
        print(f"使用设备: {device}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')  # GPU时启用pin_memory
    )
    
    # 使用Welford's online algorithm（单遍扫描，内存友好）
    mean = torch.zeros(3, device=device)
    M2 = torch.zeros(3, device=device)  # 用于计算方差
    count = 0
    
    print("使用精确算法计算均值和标准差（单遍扫描）...")
    for images, _ in tqdm(dataloader, desc="Processing"):
        batch_samples = images.size(0)
        # 将图像移到指定设备
        images = images.to(device)
        images = images.view(batch_samples, images.size(1), -1)
        
        # 对每个通道批量计算（GPU加速）
        for d in range(3):
            channel_data = images[:, d, :].flatten()
            
            # 批量更新（比逐元素更新快）
            for x in channel_data:
                count += 1
                delta = x - mean[d]
                mean[d] += delta / count
                delta2 = x - mean[d]
                M2[d] += delta * delta2
    
    # 计算标准差
    std = torch.sqrt(M2 / count)
    
    # 移回CPU并转换为列表
    return mean.cpu().tolist(), std.cpu().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算图像数据集的均值和标准差')
    parser.add_argument('--data_dir', type=str, default='data/train',
                       help='数据集目录路径（包含按类别组织的子文件夹）')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小（默认64）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载的worker数量（默认4）')
    parser.add_argument('--method', type=str, default='standard', choices=['standard', 'accurate'],
                       help='计算方法：standard（标准方法）或accurate（精确方法，适合大数据集）')
    parser.add_argument('--resize', type=int, default=224,
                       help='图像resize大小（默认224，应与训练时一致）')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备：cpu, cuda, cuda:0, cuda:1等（默认None=自动选择）')
    
    args = parser.parse_args()
    
    # 处理设备参数
    device = None
    if args.device is not None:
        if args.device.startswith('cuda'):
            if torch.cuda.is_available():
                if ':' in args.device:
                    # 指定GPU ID，如 cuda:1
                    device_id = int(args.device.split(':')[1])
                    if device_id < torch.cuda.device_count():
                        device = torch.device(f'cuda:{device_id}')
                        print(f"指定使用GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
                    else:
                        print(f"警告: GPU {device_id} 不存在，使用GPU 0")
                        device = torch.device('cuda:0')
                else:
                    # 只指定 cuda，使用GPU 0
                    device = torch.device('cuda:0')
            else:
                print("警告: CUDA不可用，使用CPU")
                device = torch.device('cpu')
        elif args.device == 'cpu':
            device = torch.device('cpu')
        else:
            print(f"警告: 未知设备 {args.device}，使用自动选择")
            device = None
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        exit(1)
    
    # 创建数据变换（只进行resize和ToTensor，不进行归一化）
    transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor()
    ])
    
    print(f"加载数据集: {args.data_dir}")
    dataset = ImageFolder(root=args.data_dir, transform=transform)
    print(f"数据集大小: {len(dataset)} 张图像")
    print(f"类别数量: {len(dataset.classes)}")
    print(f"类别列表: {dataset.classes}")
    print()
    
    # 计算均值和标准差
    if args.method == 'accurate':
        mean, std = get_mean_and_std_accurate(dataset, args.batch_size, args.num_workers, device)
    else:
        mean, std = get_mean_and_std(dataset, args.batch_size, args.num_workers, device)
    
    # 输出结果
    print("\n" + "="*60)
    print("归一化参数计算结果")
    print("="*60)
    print(f"\n均值 (mean):")
    print(f"  R通道: {mean[0]:.8f}")
    print(f"  G通道: {mean[1]:.8f}")
    print(f"  B通道: {mean[2]:.8f}")
    print(f"\n标准差 (std):")
    print(f"  R通道: {std[0]:.8f}")
    print(f"  G通道: {std[1]:.8f}")
    print(f"  B通道: {std[2]:.8f}")
    print("\n" + "="*60)
    print("在训练代码中使用:")
    print("="*60)
    print(f"transforms.Normalize(mean={mean}, std={std})")
    print("\n或者:")
    print(f"mean=[{mean[0]:.8f}, {mean[1]:.8f}, {mean[2]:.8f}]")
    print(f"std=[{std[0]:.8f}, {std[1]:.8f}, {std[2]:.8f}]")
    print("="*60)