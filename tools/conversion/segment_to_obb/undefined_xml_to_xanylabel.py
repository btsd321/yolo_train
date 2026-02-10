'''
将segment格式的XML标注转换为OBB格式的xanylabel标注
输入:
- (-i, --input) 含有segment格式XML标注文件和图片文件的文件夹路径
- (-n, --names) 类别名称文件路径（可选），格式: 每行为 "id name"。如果指定，只导出该文件中定义的类别
- (-o, --output) 输出文件夹路径，默认在输入文件夹下创建一个名为 "obb_format" 的子文件夹
输出:
- 生成OBB格式的xanylabel json文件，保存在输出文件夹中

参考文件：
- segment格式的XML文件：/home/lixinlong/Project/yolo_train/Data/DA6550741_1767605569480867885.xml
- OBB格式的xanylabel文件：/home/lixinlong/Project/yolo_train/DA6550741_1767605569480867885.json
'''

import os
import xml.etree.ElementTree as ET
import json
import argparse
import math
import numpy as np
import cv2
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


def polygon_to_obb(points):
    """
    将多边形点集转换为最小旋转矩形（OBB）
    参数:
        points: 多边形点列表 [[x1, y1], [x2, y2], ...]
    返回:
        obb_points: OBB的4个顶点 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        angle: 旋转角度（弧度）
    """
    # 将点转换为numpy数组
    points_array = np.array(points, dtype=np.float32)
    
    # 使用OpenCV计算最小旋转矩形
    rect = cv2.minAreaRect(points_array)
    
    # rect = ((center_x, center_y), (width, height), angle)
    # angle是矩形相对于水平方向的角度，范围[-90, 0)
    center, size, angle_deg = rect
    
    # 获取旋转矩形的4个顶点
    box = cv2.boxPoints(rect)
    
    # 将角度转换为弧度
    angle_rad = math.radians(angle_deg)
    
    # 确保顶点顺序一致（逆时针）
    obb_points = box.tolist()
    
    return obb_points, angle_rad


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


def parse_xml_to_obb(xml_file_path, input_folder, class_mapping=None):
    """
    解析单个XML文件并转换为OBB格式的XAnyLabel标注
    参数:
        xml_file_path: XML文件路径
        input_folder: 输入文件夹路径
        class_mapping: 类别映射字典 {class_name_lower: class_name} 或 None
    返回: XAnyLabel格式的字典（OBB格式）
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
        "version": "3.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": img_filename,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width,
        "description": ""
    }
    
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
        
        # 将多边形转换为OBB
        try:
            obb_points, angle = polygon_to_obb(points)
            
            shape = {
                "label": class_name,
                "score": None,
                "points": obb_points,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "rotation",  # OBB格式使用rotation类型
                "flags": {},
                "attributes": {},
                "kie_linking": [],
                "direction": angle  # 旋转角度
            }
            
            xanylabel_data["shapes"].append(shape)
            
        except Exception as e:
            print(f"警告: 转换多边形到OBB时出错: {e}")
            continue
    
    return xanylabel_data


def convert_folder(input_folder, output_folder=None, class_mapping=None):
    """
    转换文件夹中所有的XML文件为OBB格式的XAnyLabel文件
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径，如果为None则在输入文件夹下创建"obb_format"子文件夹
        class_mapping: 类别映射字典 {class_name_lower: class_name} 或 None
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"错误: 文件夹不存在: {input_folder}")
        return
    
    # 设置输出文件夹
    if output_folder is None:
        output_path = input_path / "obb_format"
    else:
        output_path = Path(output_folder)
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(input_path.glob('*.xml'))
    if not xml_files:
        print(f"警告: 在 {input_folder} 中没有找到XML文件")
        return
    
    print(f"找到 {len(xml_files)} 个XML文件")
    print(f"输出文件夹: {output_path}")
    converted_count = 0
    
    for xml_file in xml_files:
        try:
            # 解析XML并转换为OBB格式
            xanylabel_data = parse_xml_to_obb(
                str(xml_file), input_folder, class_mapping
            )
            
            # 生成对应的json文件
            json_file = output_path / (xml_file.stem + '.json')
            
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
        description='将segment格式的XML标注转换为OBB格式的XAnyLabel标注',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换XML文件为OBB格式（使用默认输出文件夹）
  python %(prog)s --input ./data/annotations
  
  # 指定输出文件夹
  python %(prog)s --input ./data/annotations --output ./data/obb_labels
  
  # 使用类别映射文件
  python %(prog)s --input ./data/annotations --names classes.txt --output ./data/obb_labels
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入文件夹路径，包含segment格式的XML标注文件和对应的图片文件'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出文件夹路径，默认在输入文件夹下创建 "obb_format" 子文件夹'
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
    if args.output:
        print(f"输出文件夹: {args.output}")
    else:
        print(f"输出文件夹: {args.input}/obb_format")
    print("-" * 60)
    
    convert_folder(args.input, args.output, class_mapping)