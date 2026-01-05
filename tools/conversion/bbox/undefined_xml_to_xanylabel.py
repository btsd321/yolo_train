# 本脚本功能是将xml文件中的标注信息提取出来然后转换为xanythinglabel格式的json标注文件
# 输入：图片和xml文件所在文件夹路径
# 输出：生成对应的xanythinglabel格式json标注文件，保存在输入文件夹中
# 支持两种输出模式：
#   1. segment格式（--format segment）：保留多边形点坐标
#   2. rectangle格式（--format bbox）：使用多边形外接矩形框
# 类别映射: 根据输入的classes.txt文件进行类别映射，未在文件中定义的类别将被跳过

import os
import xml.etree.ElementTree as ET
import json
import argparse
from pathlib import Path
from PIL import Image

# 用于跟踪已警告的未导出类别
_warned_skipped_classes = set()


def load_class_mapping(names_file_path):
    """
    从txt文件加载类别映射
    格式: 每行为 "id name"
    返回: {class_name_lower: class_name, ...} (保留原始大小写)
    """
    class_mapping = {}
    
    if not names_file_path or not os.path.exists(names_file_path):
        return None
    
    try:
        with open(names_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    class_name = parts[1]
                    class_mapping[class_name.lower()] = class_name  # 键用小写，值保留原始
        
        print(f"成功加载 {len(class_mapping)} 个类别映射")
        print(f"允许的类别: {', '.join(class_mapping.values())}")
        return class_mapping
    except Exception as e:
        print(f"警告: 加载类别映射文件时出错: {e}")
        return None


def parse_points(points_str):
    """
    解析多边形点坐标字符串
    例如: "(x1,y1);(x2,y2);(x3,y3);..."
    返回: [[x1, y1], [x2, y2], [x3, y3], ...]
    """
    points = []
    point_pairs = points_str.strip().strip(';').split(';')
    for pair in point_pairs:
        pair = pair.strip('()')
        if pair:
            x, y = pair.split(',')
            points.append([float(x), float(y)])
    return points


def get_bounding_box(points):
    """
    从多边形点获取外接矩形框
    返回: [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    # 返回矩形的4个顶点（左上、右上、右下、左下）
    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ]


def get_class_name(label_type, class_mapping=None):
    """
    根据labelType获取类别名称
    参数:
        label_type: 标签类型字符串 (例如: "1:delivery")
        class_mapping: 类别映射字典 {class_name_lower: class_name} 或 None
    返回: class_name 或 None
    """
    # 移除ID前缀（如果存在）
    if ':' in label_type:
        label_type = label_type.split(':', 1)[1]
    
    label_lower = label_type.lower()
    
    # 如果提供了类别映射，只使用映射中的类别
    if class_mapping is not None:
        if label_lower in class_mapping:
            return class_mapping[label_lower]
        # 类别不在映射中，记录警告
        if label_type not in _warned_skipped_classes:
            print(f"⚠ 警告: 类别 '{label_type}' 不在允许的类别列表中，将跳过")
            _warned_skipped_classes.add(label_type)
        return None
    
    # 如果没有提供类别映射，返回原始名称（移除ID后）
    return label_type


def get_image_size(xml_file_path, input_folder):
    """
    获取对应图片的尺寸
    尝试查找同名的jpg、png、jpeg等格式图片
    """
    base_name = os.path.splitext(os.path.basename(xml_file_path))[0]
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
    for ext in img_extensions:
        img_path = os.path.join(input_folder, base_name + ext)
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    return img.size, base_name + ext  # 返回 (width, height), filename
            except Exception as e:
                print(f"警告: 无法读取图片 {img_path}: {e}")
    
    print(f"警告: 找不到 {base_name} 对应的图片文件")
    return None, None


def parse_xml_to_xanylabel(xml_file_path, input_folder, output_format='segment', class_mapping=None):
    """
    解析单个XML文件并转换为XAnyLabel格式
    参数:
        output_format: 'segment' 或 'bbox'
        class_mapping: 类别映射字典 {class_name_lower: class_name} 或 None
    返回: XAnyLabel格式的字典
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # 获取图片尺寸和文件名
    img_size, img_filename = get_image_size(xml_file_path, input_folder)
    if img_size is None:
        img_width, img_height = 4024, 3036  # 默认尺寸
        img_filename = os.path.splitext(os.path.basename(xml_file_path))[0] + '.png'
    else:
        img_width, img_height = img_size
    
    # 创建XAnyLabel格式的基础结构
    xanylabel_data = {
        "version": "3.2.3",
        "flags": {},
        "shapes": [],
        "imagePath": img_filename,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width,
        "description": ""
    }
    
    # 处理矩形标注 (如果有rect元素)
    for rect in root.findall('rect'):
        label_type = rect.get('labelType', '')
        class_name = get_class_name(label_type, class_mapping)
        
        if class_name is None:
            continue
        
        x = float(rect.get('x'))
        y = float(rect.get('y'))
        w = float(rect.get('w'))
        h = float(rect.get('h'))
        
        # 矩形的4个顶点
        points = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]
        
        shape = {
            "label": class_name,
            "score": None,
            "points": points,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
            "kie_linking": []
        }
        
        xanylabel_data["shapes"].append(shape)
    
    # 处理多边形标注 (area元素)
    for area in root.findall('area'):
        label_type = area.get('labelType', '')
        class_name = get_class_name(label_type, class_mapping)
        
        if class_name is None:
            continue
        
        points_str = area.get('points', '')
        if not points_str:
            continue
        
        # 解析多边形点
        points = parse_points(points_str)
        if len(points) < 3:
            continue
        
        if output_format == 'bbox':
            # bbox格式 - 使用外接矩形框
            points = get_bounding_box(points)
            shape_type = "rectangle"
        else:
            # segment格式 - 保留多边形点
            shape_type = "polygon"
        
        shape = {
            "label": class_name,
            "score": None,
            "points": points,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": shape_type,
            "flags": {},
            "attributes": {},
            "kie_linking": []
        }
        
        xanylabel_data["shapes"].append(shape)
    
    return xanylabel_data


def convert_folder(input_folder, output_format='segment', class_mapping=None):
    """
    转换文件夹中所有的XML文件为XAnyLabel格式
    参数:
        output_format: 'segment' 或 'bbox'
        class_mapping: 类别映射字典 {class_name_lower: class_name} 或 None
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"错误: 文件夹不存在: {input_folder}")
        return
    
    xml_files = list(input_path.glob('*.xml'))
    if not xml_files:
        print(f"警告: 在 {input_folder} 中没有找到XML文件")
        return
    
    format_name = "矩形框格式" if output_format == 'bbox' else "多边形格式"
    print(f"找到 {len(xml_files)} 个XML文件")
    print(f"输出格式: {format_name}")
    converted_count = 0
    
    for xml_file in xml_files:
        try:
            # 解析XML并转换为XAnyLabel格式
            xanylabel_data = parse_xml_to_xanylabel(
                str(xml_file), input_folder, output_format, class_mapping
            )
            
            # 生成对应的json文件
            json_file = xml_file.with_suffix('.json')
            
            # 写入JSON文件
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(xanylabel_data, f, indent=2, ensure_ascii=False)
            
            if xanylabel_data["shapes"]:
                converted_count += 1
                print(f"✓ 已转换: {xml_file.name} -> {json_file.name} ({len(xanylabel_data['shapes'])} 个标注)")
            else:
                print(f"⚠ 跳过: {xml_file.name} (没有有效标注)")
                
        except Exception as e:
            print(f"✗ 错误: 处理 {xml_file.name} 时出错: {e}")
    
    print(f"\n转换完成! 成功转换 {converted_count}/{len(xml_files)} 个文件")
    
    # 显示跳过的类别总结
    if _warned_skipped_classes:
        print(f"\n跳过的类别总数: {len(_warned_skipped_classes)}")
        print(f"跳过的类别: {', '.join(sorted(_warned_skipped_classes))}")


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='将XML格式的标注文件转换为XAnyLabel格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成多边形格式（默认）
  python %(prog)s -i ./data/annotations
  
  # 生成矩形框格式
  python %(prog)s -i ./data/annotations --format bbox
  
  # 使用类别映射文件
  python %(prog)s -i ./data/annotations -n classes.txt --format segment
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=r"D:/Project/yolo_train/Data/waybill_perception",
        help='输入文件夹路径，包含XML文件和对应的图片文件'
    )
    
    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['segment', 'bbox'],
        default='bbox',
        help='输出格式: segment=多边形格式, bbox=矩形框格式 (默认: bbox)'
    )
    
    parser.add_argument(
        '-n', '--names',
        type=str,
        default=None,
        help='类别名称文件路径（可选），格式: 每行为 "id name"。如果指定，只导出该文件中定义的类别'
    )
    
    args = parser.parse_args()
    
    # 加载类别映射（如果提供）
    class_mapping = None
    if args.names:
        class_mapping = load_class_mapping(args.names)
        if not class_mapping:
            print("警告: 无法加载类别映射，将导出所有类别")
    else:
        print("未指定类别文件，将导出所有类别")
    
    # 执行转换
    print(f"输入文件夹: {args.input}")
    print(f"输出格式: {'矩形框格式' if args.format == 'bbox' else '多边形格式'}")
    print("-" * 60)
    
    convert_folder(args.input, args.format, class_mapping)
