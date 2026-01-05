"""
YOLO分割标注转换为YOLO边界框标注
将YOLO segment格式转换为YOLO bbox格式
"""
import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple


def read_classes_file(classes_file: str) -> List[str]:
    """读取类别文件"""
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def parse_segment_line(line: str) -> Tuple[int, List[float]]:
    """解析分割标注行，返回类别ID和坐标点列表"""
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    return class_id, coords


def segment_to_bbox(coords: List[float]) -> Tuple[float, float, float, float]:
    """
    将分割坐标转换为边界框
    coords: [x1, y1, x2, y2, ..., xn, yn]
    返回: (x_center, y_center, width, height) 归一化坐标
    """
    # 提取x和y坐标
    x_coords = coords[0::2]
    y_coords = coords[1::2]
    
    # 计算边界框
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # 转换为中心点格式
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height


def convert_file(input_file: str, output_file: str, allowed_classes: Set[int] = None, 
                 class_mapping: Dict[int, int] = None) -> None:
    """
    转换单个标注文件
    
    Args:
        input_file: 输入的YOLO分割标注文件路径
        output_file: 输出的YOLO边界框标注文件路径
        allowed_classes: 允许的类别ID集合（如果为None则转换所有类别）
        class_mapping: 类别ID映射字典（用于重映射）
    """
    if not os.path.exists(input_file):
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    bbox_lines = []
    for line in lines:
        if not line.strip():
            continue
        
        class_id, coords = parse_segment_line(line)
        
        # 如果指定了允许的类别，跳过不在列表中的类别
        if allowed_classes is not None and class_id not in allowed_classes:
            continue
        
        # 应用类别映射
        if class_mapping is not None:
            class_id = class_mapping.get(class_id, class_id)
        
        # 转换为边界框
        x_center, y_center, width, height = segment_to_bbox(coords)
        
        # 写入YOLO bbox格式
        bbox_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        bbox_lines.append(bbox_line)
    
    # 写入输出文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(bbox_lines)


def get_image_extensions() -> List[str]:
    """获取支持的图片扩展名"""
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP']


def find_image_for_label(label_path: str, input_dir: str) -> str:
    """查找标注文件对应的图片"""
    label_stem = Path(label_path).stem
    for ext in get_image_extensions():
        image_path = os.path.join(input_dir, label_stem + ext)
        if os.path.exists(image_path):
            return image_path
    return None


def collect_used_classes(input_dir: str, allowed_classes: Set[int] = None) -> Set[int]:
    """收集实际使用的类别ID"""
    used_classes = set()
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue
        
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    class_id, _ = parse_segment_line(line)
                    
                    # 如果指定了允许的类别，只收集允许的类别
                    if allowed_classes is None or class_id in allowed_classes:
                        used_classes.add(class_id)
        except Exception as e:
            print(f"警告: 读取文件 {filename} 时出错: {e}")
    
    return used_classes


def create_class_mapping(used_classes: Set[int]) -> Dict[int, int]:
    """创建类别ID重映射字典"""
    sorted_classes = sorted(used_classes)
    mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_classes)}
    return mapping


def save_remapped_classes(original_classes: List[str], class_mapping: Dict[int, int], 
                          output_path: str) -> None:
    """保存重映射后的类别文件"""
    # 创建新的类别列表
    remapped_classes = [''] * len(class_mapping)
    for old_id, new_id in class_mapping.items():
        if old_id < len(original_classes):
            remapped_classes[new_id] = original_classes[old_id]
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for class_name in remapped_classes:
            f.write(f"{class_name}\n")
    
    print(f"重映射后的类别文件已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='将YOLO分割标注转换为YOLO边界框标注'
    )
    parser.add_argument('--input', required=True, help='输入YOLO标注文件所在文件夹路径')
    parser.add_argument('--output', help='输出文件夹路径（默认为输入文件夹下的bbox文件夹）')
    parser.add_argument('--names', help='类别文件路径（classes.txt），只转换文件中的类别')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return
    
    # 设置输出目录
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(args.input, 'bbox')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取类别文件
    allowed_classes = None
    original_classes = None
    if args.names and os.path.exists(args.names):
        original_classes = read_classes_file(args.names)
        allowed_classes = set(range(len(original_classes)))
        print(f"已加载类别文件，共 {len(original_classes)} 个类别")
    
    # 收集实际使用的类别
    print("扫描标注文件，收集使用的类别...")
    used_classes = collect_used_classes(args.input, allowed_classes)
    print(f"实际使用的类别ID: {sorted(used_classes)}")
    
    # 创建类别映射
    class_mapping = None
    sorted_used = sorted(used_classes)
    if sorted_used != list(range(len(sorted_used))):
        # ID不连续，需要重映射
        class_mapping = create_class_mapping(used_classes)
        print(f"类别ID不连续，进行重映射: {class_mapping}")
        
        # 保存重映射后的类别文件
        if args.names and original_classes:
            names_dir = os.path.dirname(args.names)
            remapped_names_file = os.path.join(names_dir, 'classes_remapped.txt')
            save_remapped_classes(original_classes, class_mapping, remapped_names_file)
    
    # 转换标注文件
    print(f"\n开始转换标注文件...")
    converted_count = 0
    copied_images = 0
    
    for filename in os.listdir(args.input):
        if not filename.endswith('.txt'):
            continue
        
        input_file = os.path.join(args.input, filename)
        output_file = os.path.join(output_dir, filename)
        
        # 转换标注文件
        convert_file(input_file, output_file, allowed_classes, class_mapping)
        converted_count += 1
        
        # 查找并复制对应的图片
        image_path = find_image_for_label(input_file, args.input)
        if image_path:
            image_filename = os.path.basename(image_path)
            output_image_path = os.path.join(output_dir, image_filename)
            shutil.copy2(image_path, output_image_path)
            copied_images += 1
        
        if converted_count % 100 == 0:
            print(f"已转换 {converted_count} 个文件...")
    
    print(f"\n转换完成！")
    print(f"- 转换标注文件: {converted_count} 个")
    print(f"- 复制图片: {copied_images} 个")
    print(f"- 输出目录: {output_dir}")


if __name__ == '__main__':
    main()
