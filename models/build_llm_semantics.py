"""
离线生成LLM语义矩阵和类别中心
用于LSAL (LLM-Semantic Adaptive Loss)训练

使用方法:
    python build_llm_semantics.py --classnames "Pneumonia" "Fracture" "Edema" --output_dir ./semantics
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import json

# 设置环境变量
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
os.environ.setdefault('HF_HUB_DOWNLOAD_MAX_RETRIES', '10')

# 添加open_clip路径
script_dir = Path(__file__).parent
open_clip_base_path = script_dir.parent / 'open_clip'
if open_clip_base_path.exists():
    sys.path.insert(0, str(open_clip_base_path.parent))
    print(f"Using open_clip path: {open_clip_base_path}")

try:
    from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
except ImportError:
    print("Error: open_clip not found. Please install or ensure open_clip is in the path.")
    print(f"  Tried path: {open_clip_base_path}")
    raise


def build_llm_semantics(classnames, biomedclip_model, tokenizer, device, tau=0.1, class_templates=None):
    """
    利用 BiomedCLIP 的 Text Encoder 和 LLM 的知识构建语义矩阵。
    
    Args:
        classnames: 类别名称列表，例如 ["Pneumonia", "Fracture", "Edema"]
        biomedclip_model: BiomedCLIP模型（已加载）
        tokenizer: BiomedCLIP的tokenizer
        device: 设备（'cuda'或'cpu'）
        tau: 温度系数，用于softmax平滑（默认0.1）
        class_templates: 每个类别的模板字典 {classname: [template1, template2, ...]}
                        如果为None或某个类别没有模板，将使用默认模板
    
    Returns:
        class_centers: [N_classes, Dim] 每个类别的语义中心
        soft_labels_matrix: [N_classes, N_classes] 软标签矩阵
    """
    biomedclip_model.eval()
    
    # 默认模板：模拟LLM的多角度描述
    default_templates = [
        "a histopathology slide of {}",
        "microscopic view of {}",
        "{} tissue structure",
        "pathological features of {}",
        "medical image showing {}",
        "clinical presentation of {}"
    ]
    
    all_class_embeddings = []
    
    print("Computing LLM Semantic Centers...")
    print(f"  Number of classes: {len(classnames)}")
    
    with torch.no_grad():
        for idx, name in enumerate(classnames):
            # 获取该类别的模板
            if class_templates and name in class_templates and class_templates[name]:
                # 使用自定义模板（已经是完整的描述，不需要format）
                prompts = class_templates[name]
                print(f"  Class {idx+1}/{len(classnames)}: {name} - Using {len(prompts)} custom templates")
            else:
                # 使用默认模板
                prompts = [t.format(name) for t in default_templates]
                if idx == 0:  # 只在第一个类别时打印
                    print(f"  Using default templates ({len(default_templates)} per class)")
            
            # Tokenize所有prompts并提取特征
            # 按照biomedcoop的方式处理tokenizer
            text_features_list = []
            for p in prompts:
                tokenized = tokenizer(p)
                if isinstance(tokenized, torch.Tensor):
                    tokenized = tokenized.to(device)
                elif isinstance(tokenized, dict):
                    tokenized = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in tokenized.items()}
                
                # 使用encode_text提取特征
                text_feat = biomedclip_model.encode_text(tokenized, normalize=True)
                text_features_list.append(text_feat)
            
            # Stack所有特征
            text_features = torch.cat(text_features_list, dim=0)  # [n_templates, dim]
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
            
            # 计算该类的"语义中心" (Mean Embedding)
            class_center = text_features.mean(dim=0)
            class_center = class_center / (class_center.norm(dim=-1, keepdim=True) + 1e-8)
            all_class_embeddings.append(class_center)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(classnames)} classes...")
    
    # [N_classes, Dim]
    class_centers = torch.stack(all_class_embeddings)
    
    print("Computing similarity matrix...")
    # 计算类间相似度矩阵
    # Sim[i, j] 越大，表示第 i 类和第 j 类在 BiomedCLIP 空间中越像
    similarity_matrix = class_centers @ class_centers.t()  # [N, N]
    
    print(f"  Similarity matrix shape: {similarity_matrix.shape}")
    print(f"  Similarity range: [{similarity_matrix.min().item():.4f}, {similarity_matrix.max().item():.4f}]")
    
    # 转化为 Soft Labels (带温度系数 tau)
    # 温度越高，标签越平滑（容忍度越高）；温度越低，越接近 One-hot
    soft_labels_matrix = torch.softmax(similarity_matrix / tau, dim=1)
    
    print(f"  Temperature (tau): {tau}")
    print(f"  Soft labels matrix shape: {soft_labels_matrix.shape}")
    
    # 打印一些统计信息
    diagonal_mean = soft_labels_matrix.diag().mean().item()
    off_diagonal_mean = (soft_labels_matrix.sum(dim=1) - soft_labels_matrix.diag()).mean().item() / (len(classnames) - 1)
    print(f"  Diagonal mean (self-similarity): {diagonal_mean:.4f}")
    print(f"  Off-diagonal mean (cross-similarity): {off_diagonal_mean:.4f}")
    
    return class_centers.cpu(), soft_labels_matrix.cpu()


def load_class_templates(classnames, templates_file=None, templates_dict=None):
    """
    加载类别模板描述
    
    Args:
        classnames: 类别名称列表
        templates_file: 模板文件路径（Python文件，包含 HIP_PROSTHESIS_TEMPLATES 字典）
        templates_dict: 直接提供的模板字典（优先级最高）
    
    Returns:
        templates_dict: {classname: [template1, template2, ...]}
    """
    # 如果直接提供了模板字典，直接使用
    if templates_dict is not None:
        return templates_dict
    
    # 如果提供了模板文件，尝试导入
    if templates_file:
        templates_file = Path(templates_file)
        if templates_file.exists():
            # 动态导入模板文件
            import importlib.util
            spec = importlib.util.spec_from_file_location("templates_module", templates_file)
            templates_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(templates_module)
            
            # 获取模板字典（假设名为 HIP_PROSTHESIS_TEMPLATES）
            if hasattr(templates_module, 'HIP_PROSTHESIS_TEMPLATES'):
                loaded_templates = templates_module.HIP_PROSTHESIS_TEMPLATES
                # 只返回需要的类别
                result = {}
                for name in classnames:
                    # 尝试匹配类别名称（支持大小写不敏感）
                    matched_key = None
                    for key in loaded_templates.keys():
                        if key.lower() == name.lower() or key.replace(" ", "_").lower() == name.replace(" ", "_").lower():
                            matched_key = key
                            break
                    
                    if matched_key:
                        result[name] = loaded_templates[matched_key]
                    else:
                        print(f"Warning: No templates found for class '{name}', will use default templates")
                        result[name] = None
                return result
            else:
                print(f"Warning: Template file {templates_file} does not contain HIP_PROSTHESIS_TEMPLATES")
    
    # 如果没有找到模板，返回None（将使用默认模板）
    return None


def main():
    parser = argparse.ArgumentParser(description='Build LLM semantics for LSAL training')
    parser.add_argument('--classnames', nargs='+', default=None,
                        help='List of class names, e.g., "Pneumonia" "Fracture" "Edema"')
    parser.add_argument('--classnames_file', type=str, default=None,
                        help='JSON file containing class names (alternative to --classnames)')
    parser.add_argument('--templates_file', type=str, default=None,
                        help='Python file containing class templates (e.g., hip_prosthesis_prompt_templates.py)')
    parser.add_argument('--output_dir', type=str, default='./semantics',
                        help='Output directory for saving semantics files')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='Temperature parameter for softmax (default: 0.1)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--model_name', type=str, 
                        default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                        help='BiomedCLIP model name')
    
    args = parser.parse_args()
    
    # 加载类别名称
    if args.classnames_file:
        with open(args.classnames_file, 'r', encoding='utf-8') as f:
            classnames_dict = json.load(f)
            classnames = list(classnames_dict.keys())
        print(f"Loaded {len(classnames)} classes from {args.classnames_file}")
    elif args.classnames:
        classnames = args.classnames
        print(f"Using {len(classnames)} classes from command line")
    else:
        raise ValueError("Either --classnames or --classnames_file must be provided")
    
    print(f"Classes: {classnames}")
    
    # 加载类别模板
    class_templates = None
    if args.templates_file:
        print(f"\nLoading class templates from: {args.templates_file}")
        class_templates = load_class_templates(classnames, templates_file=args.templates_file)
        if class_templates:
            # 统计模板数量
            total_templates = sum(len(templates) if templates else 0 for templates in class_templates.values())
            print(f"✓ Loaded templates for {len([t for t in class_templates.values() if t])} classes")
            print(f"  Total templates: {total_templates}")
            for name, templates in class_templates.items():
                if templates:
                    print(f"    - {name}: {len(templates)} templates")
                else:
                    print(f"    - {name}: Using default templates (no custom templates found)")
        else:
            print("  No templates loaded, will use default templates")
    else:
        # 尝试自动查找模板文件
        default_templates_path = Path(__file__).parent.parent / 'hip_prosthesis_prompt_templates.py'
        if default_templates_path.exists():
            print(f"\nAuto-detected templates file: {default_templates_path}")
            class_templates = load_class_templates(classnames, templates_file=default_templates_path)
            if class_templates:
                total_templates = sum(len(templates) if templates else 0 for templates in class_templates.values())
                print(f"✓ Auto-loaded templates: {total_templates} total templates")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型和tokenizer
    print(f"\nLoading BiomedCLIP model: {args.model_name}")
    print(f"Device: {args.device}")
    try:
        biomedclip_model, _ = create_model_from_pretrained(args.model_name)
        biomedclip_model = biomedclip_model.float().to(args.device)
        tokenizer = get_tokenizer(args.model_name)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise
    
    # 构建语义矩阵
    print("\n" + "="*80)
    print("Building LLM Semantics...")
    print("="*80)
    
    # 准备模板：如果有自定义模板，使用自定义模板；否则使用默认模板
    templates_to_use = None
    if class_templates:
        # 将字典格式转换为函数需要的格式
        # build_llm_semantics 需要每个类别一个模板列表
        templates_to_use = {}
        for name in classnames:
            if name in class_templates and class_templates[name]:
                templates_to_use[name] = class_templates[name]
            # 如果某个类别没有自定义模板，会在函数内部使用默认模板
    
    class_centers, soft_labels_matrix = build_llm_semantics(
        classnames, biomedclip_model, tokenizer, args.device, tau=args.tau, 
        class_templates=templates_to_use
    )
    
    # 保存结果
    print("\n" + "="*80)
    print("Saving results...")
    print("="*80)
    
    # 保存为PyTorch格式
    centers_path = output_dir / 'class_centers.pt'
    matrix_path = output_dir / 'soft_labels_matrix.pt'
    
    torch.save(class_centers, centers_path)
    torch.save(soft_labels_matrix, matrix_path)
    
    print(f"✓ Saved class centers to: {centers_path}")
    print(f"  Shape: {class_centers.shape}, Dtype: {class_centers.dtype}")
    print(f"✓ Saved soft labels matrix to: {matrix_path}")
    print(f"  Shape: {soft_labels_matrix.shape}, Dtype: {soft_labels_matrix.dtype}")
    
    # 保存类别名称映射（用于后续加载）
    classnames_dict = {idx: name for idx, name in enumerate(classnames)}
    classnames_path = output_dir / 'classnames.json'
    with open(classnames_path, 'w', encoding='utf-8') as f:
        json.dump(classnames_dict, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved classnames mapping to: {classnames_path}")
    
    # 保存配置信息
    config = {
        'num_classes': len(classnames),
        'tau': args.tau,
        'model_name': args.model_name,
        'classnames': classnames
    }
    config_path = output_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved config to: {config_path}")
    
    print("\n" + "="*80)
    print("Done! You can now use these files for LSAL training.")
    print("="*80)


if __name__ == '__main__':
    main()

