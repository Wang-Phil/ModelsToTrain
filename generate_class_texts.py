"""
自动生成类别文本描述文件
从数据目录结构生成类别文本描述的JSON文件
"""

import json
import argparse
from pathlib import Path


def generate_class_texts_from_template(data_dir, template="这是一张{class_name}的医学图像", output_file="class_texts.json"):
    """
    从模板生成类别文本描述
    
    Args:
        data_dir: 数据目录（按类别组织的文件夹）
        template: 文本模板，例如 "这是一张{class_name}的医学图像"
        output_file: 输出JSON文件路径
    """
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    class_texts = {}
    for cls in classes:
        class_texts[cls] = template.format(class_name=cls)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(class_texts, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 类别文本描述文件已生成: {output_file}")
    print(f"✓ 包含 {len(classes)} 个类别")
    print(f"\n类别列表:")
    for cls in classes:
        print(f"  {cls}: {class_texts[cls]}")
    
    return class_texts


def generate_class_texts_interactive(data_dir, output_file="class_texts.json"):
    """
    交互式生成类别文本描述（需要用户输入）
    
    Args:
        data_dir: 数据目录（按类别组织的文件夹）
        output_file: 输出JSON文件路径
    """
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"找到 {len(classes)} 个类别")
    print("请为每个类别输入文本描述（直接回车使用默认模板）：\n")
    
    class_texts = {}
    default_template = "这是一张{class_name}的医学图像"
    
    for cls in classes:
        default_text = default_template.format(class_name=cls)
        user_input = input(f"类别 '{cls}' 的文本描述 [{default_text}]: ").strip()
        
        if user_input:
            class_texts[cls] = user_input
        else:
            class_texts[cls] = default_text
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(class_texts, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 类别文本描述文件已生成: {output_file}")
    print(f"\n最终类别文本描述:")
    for cls, text in class_texts.items():
        print(f"  {cls}: {text}")
    
    return class_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成类别文本描述JSON文件')
    parser.add_argument('--data-dir', type=str, required=True, 
                       help='数据目录（按类别组织的文件夹）')
    parser.add_argument('--output', type=str, default='class_texts.json',
                       help='输出JSON文件路径（默认: class_texts.json）')
    parser.add_argument('--template', type=str, default=None,
                       help='文本模板，例如 "这是一张{class_name}的医学图像"')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式模式，为每个类别输入自定义描述')
    
    args = parser.parse_args()
    
    if args.interactive:
        generate_class_texts_interactive(args.data_dir, args.output)
    else:
        template = args.template if args.template else "这是一张{class_name}的医学图像"
        generate_class_texts_from_template(args.data_dir, template, args.output)

