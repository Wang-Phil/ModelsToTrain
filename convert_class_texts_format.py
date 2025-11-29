"""
将类别文本描述从列表格式转换为字典格式

输入格式（用户提供的格式）:
{
  "class_names": ["类别1", "类别2", ...],
  "class_texts": ["描述1", "描述2", ...]
}

输出格式（CLIP训练使用的格式）:
{
  "类别1": "描述1",
  "类别2": "描述2",
  ...
}
"""

import json
import argparse


def convert_class_texts_format(input_file, output_file):
    """
    转换类别文本描述格式
    
    Args:
        input_file: 输入的JSON文件路径（包含class_names和class_texts列表）
        output_file: 输出的JSON文件路径（类别名 -> 文本描述的字典）
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'class_names' not in data or 'class_texts' not in data:
        raise ValueError("输入文件必须包含 'class_names' 和 'class_texts' 字段")
    
    class_names = data['class_names']
    class_texts = data['class_texts']
    
    if len(class_names) != len(class_texts):
        raise ValueError(f"类别名称数量 ({len(class_names)}) 与文本描述数量 ({len(class_texts)}) 不匹配")
    
    # 转换为字典格式
    result = {}
    for class_name, class_text in zip(class_names, class_texts):
        result[class_name] = class_text
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 成功转换 {len(result)} 个类别的文本描述")
    print(f"✓ 输出文件: {output_file}")
    print(f"\n类别文本描述预览:")
    for i, (class_name, class_text) in enumerate(result.items(), 1):
        print(f"{i}. {class_name}: {class_text[:60]}..." if len(class_text) > 60 else f"{i}. {class_name}: {class_text}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='转换类别文本描述格式')
    parser.add_argument('--input', type=str, required=True,
                       help='输入的JSON文件路径（包含class_names和class_texts列表）')
    parser.add_argument('--output', type=str, required=True,
                       help='输出的JSON文件路径（类别名 -> 文本描述的字典）')
    
    args = parser.parse_args()
    convert_class_texts_format(args.input, args.output)

