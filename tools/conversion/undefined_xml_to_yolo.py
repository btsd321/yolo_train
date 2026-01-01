# 本脚本功能是将xml文件中的标注信息提取出来然后转换为yolo格式的txt标注文件
# 输入：图片和xml文件所在文件夹路径
# 输出：生成对应的yolo格式txt标注文件，保存在输入文件夹中
# 支持两种输出模式：
#   1. YOLO分割格式（--format segment）：保留多边形点坐标
#   2. YOLO bbox格式（--format bbox）：使用多边形外接矩形框
# 类别映射：
#   0: delivery (软包裹)
#   1: box (硬纸盒)
#   2: ExpressBillSeg (面单)
#   3: BarCode (条形码)
#   4: 2DCode (二维码)

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import argparse

# 用于跟踪已警告的未导出类别
_warned_skipped_classes = set()


def load_class_mapping(names_file_path):
    """
    从txt文件加载类别映射
    格式: 每行为 "id name"
    返回: {class_name_lower: class_id, ...}
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
                    class_id = int(parts[0])
                    class_name = parts[1].lower()  # 转换为小写以便匹配
                    class_mapping[class_name] = class_id
        
        print(f"成功加载 {len(class_mapping)} 个类别映射")
        print(f"允许的类别: {', '.join([f'{name}({id})' for name, id in class_mapping.items()])}")
        return class_mapping
    except Exception as e:
        print(f"警告: 加载类别映射文件时出错: {e}")
        return None


def parse_points(points_str):
    """
    解析多边形点坐标字符串
    例如: "(x1,y1);(x2,y2);(x3,y3);..."
    返回: [(x1, y1), (x2, y2), (x3, y3), ...]
    """
    points = []
    point_pairs = points_str.strip().strip(';').split(';')
    for pair in point_pairs:
        pair = pair.strip('()')
        if pair:
            x, y = pair.split(',')
            points.append((float(x), float(y)))
    return points


def get_bounding_box(points):
    """
    从多边形点获取外接矩形框
    返回: (xmin, ymin, xmax, ymax)
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def convert_to_yolo_bbox(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    将像素坐标转换为YOLO bbox格式 (归一化的中心点坐标和宽高)
    返回: (x_center, y_center, width, height) 归一化后的值
    """
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height


def convert_polygon_to_yolo_segment(points, img_width, img_height):
    """
    将多边形点转换为YOLO分割格式 (归一化的点坐标)
    返回: [x1, y1, x2, y2, ..., xn, yn] 归一化后的值
    """
    normalized_points = []
    for x, y in points:
        normalized_points.append(x / img_width)
        normalized_points.append(y / img_height)
    return normalized_points


def get_class_id(label_type, class_mapping=None):
    """
    根据labelType获取类别ID
    参数:
        label_type: 标签类型字符串
        class_mapping: 类别映射字典 {class_name_lower: class_id} 或 None
    返回: (class_id, class_name) 或 (None, None)
    """
    label_lower = label_type.lower()
    
    # 如果提供了类别映射，只使用映射中的类别
    if class_mapping is not None:
        for class_name, class_id in class_mapping.items():
            if class_name in label_lower:
                return class_id, class_name
        # 类别不在映射中，记录警告
        if label_type not in _warned_skipped_classes:
            print(f"⚠ 警告: 类别 '{label_type}' 不在允许的类别列表中，将跳过")
            _warned_skipped_classes.add(label_type)
        return None, None
    
    # 如果没有提供类别映射，使用默认映射
    if 'delivery' in label_lower:
        return 0, 'delivery'  # 软包裹
    elif 'box' in label_lower:
        return 1, 'box'  # 硬纸盒
    elif 'expressbillseg' in label_lower:
        return 2, 'expressbillseg'  # 面单
    elif 'barcode' in label_lower:
        return 3, 'barcode'  # 条形码
    elif '2dcode' in label_lower or 'qrcode' in label_lower:
        return 4, '2dcode'  # 二维码
    else:
        return None, None


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
                    return img.size  # 返回 (width, height)
            except Exception as e:
                print(f"警告: 无法读取图片 {img_path}: {e}")
    
    # 如果找不到图片，返回默认尺寸（根据XML中的坐标推测）
    print(f"警告: 找不到 {base_name} 对应的图片文件，使用默认尺寸")
    return None


def check_and_remap_class_ids(input_folder, used_class_ids, original_class_mapping):
    """
    检查类别ID是否连续，如果不连续则询问用户是否重映射
    参数:
        input_folder: 输入文件夹路径
        used_class_ids: 使用的类别ID集合
        original_class_mapping: 原始类别映射 {class_name_lower: class_id}
    """
    if not used_class_ids:
        return
    
    sorted_ids = sorted(used_class_ids)
    min_id = min(sorted_ids)
    max_id = max(sorted_ids)
    expected_ids = set(range(min_id, max_id + 1))
    
    print(f"\n{'='*60}")
    print("类别ID检查")
    print(f"{'='*60}")
    print(f"使用的类别ID: {sorted_ids}")
    print(f"ID范围: {min_id} - {max_id}")
    
    # 检查是否从0开始
    if min_id != 0:
        print(f"⚠ 警告: 类别ID未从0开始 (最小ID是 {min_id})")
    
    # 检查是否连续
    missing_ids = expected_ids - used_class_ids
    if missing_ids:
        print(f"⚠ 警告: 类别ID不连续，缺少ID: {sorted(missing_ids)}")
    
    # 如果ID从0开始且连续，无需处理
    if min_id == 0 and not missing_ids:
        print("✓ 类别ID从0开始且连续，符合YOLO要求")
        return
    
    # 需要重映射
    print(f"\n{'='*60}")
    print("YOLO要求类别ID必须从0开始且连续!")
    print(f"{'='*60}")
    
    # 创建重映射方案
    id_remapping = {}
    new_id = 0
    for old_id in sorted_ids:
        id_remapping[old_id] = new_id
        new_id += 1
    
    print("\n建议的重映射方案:")
    for old_id in sorted(id_remapping.keys()):
        new_id = id_remapping[old_id]
        print(f"  {old_id} -> {new_id}")
    
    # 询问用户
    print(f"\n是否自动生成重映射文件并更新所有标注? (y/n): ", end='')
    response = input().strip().lower()
    
    if response == 'y':
        remap_annotations(input_folder, id_remapping, original_class_mapping)
    else:
        print("已取消重映射操作")
        print("⚠ 注意: 当前的类别ID不符合YOLO要求，可能导致训练失败!")


def remap_annotations(input_folder, id_remapping, original_class_mapping):
    """
    重映射所有标注文件的类别ID并生成新的类别映射文件
    参数:
        input_folder: 输入文件夹路径
        id_remapping: ID重映射字典 {old_id: new_id}
        original_class_mapping: 原始类别映射 {class_name_lower: old_id}
    """
    input_path = Path(input_folder)
    txt_files = list(input_path.glob('*.txt'))
    
    if not txt_files:
        print("错误: 没有找到需要重映射的txt文件")
        return
    
    print(f"\n开始重映射 {len(txt_files)} 个标注文件...")
    
    remapped_count = 0
    for txt_file in txt_files:
        try:
            # 读取原始标注
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 重映射每一行
            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if parts:
                    old_class_id = int(parts[0])
                    if old_class_id in id_remapping:
                        new_class_id = id_remapping[old_class_id]
                        parts[0] = str(new_class_id)
                        new_lines.append(' '.join(parts))
            
            # 写回文件
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            
            remapped_count += 1
            
        except Exception as e:
            print(f"✗ 错误: 重映射 {txt_file.name} 时出错: {e}")
    
    print(f"✓ 成功重映射 {remapped_count}/{len(txt_files)} 个文件")
    
    # 生成新的类别映射文件
    generate_remapped_classes_file(input_folder, id_remapping, original_class_mapping)


def generate_remapped_classes_file(input_folder, id_remapping, original_class_mapping):
    """
    生成重映射后的类别文件
    参数:
        input_folder: 输入文件夹路径
        id_remapping: ID重映射字典 {old_id: new_id}
        original_class_mapping: 原始类别映射 {class_name_lower: old_id}
    """
    output_file = Path(input_folder) / 'classes_remapped.txt'
    
    # 创建反向映射: old_id -> class_name
    id_to_name = {}
    if original_class_mapping:
        for class_name, old_id in original_class_mapping.items():
            id_to_name[old_id] = class_name
    else:
        # 如果没有原始映射，使用默认名称
        default_names = {
            0: 'delivery',
            1: 'box',
            2: 'expressbillseg',
            3: 'barcode',
            4: '2dcode'
        }
        for old_id in id_remapping.keys():
            id_to_name[old_id] = default_names.get(old_id, f'class_{old_id}')
    
    # 按新ID排序生成类别文件
    lines = []
    for old_id in sorted(id_remapping.keys()):
        new_id = id_remapping[old_id]
        class_name = id_to_name.get(old_id, f'class_{old_id}')
        lines.append(f"{new_id} {class_name}")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✓ 已生成重映射类别文件: {output_file}")
    print("\n新的类别映射:")
    for line in lines:
        print(f"  {line}")
    print(f"\n{'='*60}")
    print("重映射完成! 请使用新的类别文件进行训练。")
    print(f"{'='*60}")


def parse_xml_to_yolo(xml_file_path, input_folder, output_format='bbox', img_width=None, img_height=None, class_mapping=None):
    """
    解析单个XML文件并转换为YOLO格式
    参数:
        output_format: 'bbox' 或 'segment'
        class_mapping: 类别映射字典 {class_name_lower: class_id} 或 None
    返回: YOLO格式的标注行列表
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # 如果没有提供图片尺寸，尝试从图片文件获取
    if img_width is None or img_height is None:
        size = get_image_size(xml_file_path, input_folder)
        if size:
            img_width, img_height = size
        else:
            # 从XML中的坐标推测图片尺寸
            img_width, img_height = 1920, 1080  # 默认值
    
    yolo_annotations = []
    
    # 处理矩形标注 (如果有rect元素)
    for rect in root.findall('rect'):
        label_type = rect.get('labelType', '')
        class_id, class_name = get_class_id(label_type, class_mapping)
        
        if class_id is None:
            continue
        
        x = float(rect.get('x'))
        y = float(rect.get('y'))
        w = float(rect.get('w'))
        h = float(rect.get('h'))
        
        # 计算矩形框的边界
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        
        if output_format == 'bbox':
            # YOLO bbox格式
            x_center, y_center, width, height = convert_to_yolo_bbox(
                xmin, ymin, xmax, ymax, img_width, img_height
            )
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        else:
            # YOLO segment格式 - 矩形转为4个点的多边形
            points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            normalized = convert_polygon_to_yolo_segment(points, img_width, img_height)
            coords_str = ' '.join([f"{coord:.6f}" for coord in normalized])
            yolo_annotations.append(f"{class_id} {coords_str}")
    
    # 处理多边形标注 (area元素)
    for area in root.findall('area'):
        label_type = area.get('labelType', '')
        class_id, class_name = get_class_id(label_type, class_mapping)
        
        if class_id is None:
            continue
        
        points_str = area.get('points', '')
        if not points_str:
            continue
        
        # 解析多边形点
        points = parse_points(points_str)
        if len(points) < 3:
            continue
        
        if output_format == 'bbox':
            # YOLO bbox格式 - 使用外接矩形框
            xmin, ymin, xmax, ymax = get_bounding_box(points)
            x_center, y_center, width, height = convert_to_yolo_bbox(
                xmin, ymin, xmax, ymax, img_width, img_height
            )
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        else:
            # YOLO segment格式 - 保留多边形点
            normalized = convert_polygon_to_yolo_segment(points, img_width, img_height)
            coords_str = ' '.join([f"{coord:.6f}" for coord in normalized])
            yolo_annotations.append(f"{class_id} {coords_str}")
    
    return yolo_annotations


def convert_folder(input_folder, output_format='bbox', img_width=None, img_height=None, class_mapping=None):
    """
    转换文件夹中所有的XML文件为YOLO格式
    参数:
        output_format: 'bbox' 或 'segment'
        class_mapping: 类别映射字典 {class_name_lower: class_id} 或 None
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"错误: 文件夹不存在: {input_folder}")
        return
    
    xml_files = list(input_path.glob('*.xml'))
    if not xml_files:
        print(f"警告: 在 {input_folder} 中没有找到XML文件")
        return
    
    format_name = "YOLO分割格式" if output_format == 'segment' else "YOLO bbox格式"
    print(f"找到 {len(xml_files)} 个XML文件")
    print(f"输出格式: {format_name}")
    converted_count = 0
    used_class_ids = set()  # 收集所有使用的类别ID
    
    for xml_file in xml_files:
        try:
            # 解析XML并转换为YOLO格式
            yolo_annotations = parse_xml_to_yolo(
                str(xml_file), input_folder, output_format, img_width, img_height, class_mapping
            )
            
            # 收集使用的类别ID
            for annotation in yolo_annotations:
                class_id = int(annotation.split()[0])
                used_class_ids.add(class_id)
            
            # 生成对应的txt文件
            txt_file = xml_file.with_suffix('.txt')
            
            # 写入YOLO格式标注
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_annotations))
            
            if yolo_annotations:
                converted_count += 1
                print(f"✓ 已转换: {xml_file.name} -> {txt_file.name} ({len(yolo_annotations)} 个标注)")
            else:
                print(f"⚠ 跳过: {xml_file.name} (没有有效标注)")
                
        except Exception as e:
            print(f"✗ 错误: 处理 {xml_file.name} 时出错: {e}")
    
    print(f"\n转换完成! 成功转换 {converted_count}/{len(xml_files)} 个文件")
    
    # 检查类别ID的连续性
    if used_class_ids:
        check_and_remap_class_ids(input_folder, used_class_ids, class_mapping)


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='将XML格式的标注文件转换为YOLO格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成YOLO bbox数据集（默认）
  python %(prog)s -i ./data/annotations
  
  # 生成YOLO分割数据集
  python %(prog)s -i ./data/annotations --format segment
  
  # 指定图片尺寸
  python %(prog)s -i ./data/annotations -w 1920 -h 1080 --format segment
  
类别映射:
  0: delivery (软包裹)
  1: box (硬纸盒)
  2: ExpressBillSeg (面单)
  3: BarCode (条形码)
  4: 2DCode (二维码)
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
        choices=['bbox', 'segment'],
        default='bbox',
        help='输出格式: bbox=传统YOLO检测框格式, segment=YOLO分割格式 (默认: bbox)'
    )
    
    parser.add_argument(
        '-w', '--width',
        type=int,
        default=None,
        help='图片宽度（像素），如果不指定则自动从图片文件读取'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='图片高度（像素），如果不指定则自动从图片文件读取'
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
            print("警告: 无法加载类别映射，将使用默认类别")
    else:
        print("未指定类别文件，将使用默认类别映射")
    
    # 执行转换
    print(f"输入文件夹: {args.input}")
    print(f"输出格式: {'YOLO分割格式' if args.format == 'segment' else 'YOLO bbox格式'}")
    if args.width and args.height:
        print(f"指定图片尺寸: {args.width}x{args.height}")
    else:
        print("图片尺寸: 自动检测")
    print("-" * 60)
    
    convert_folder(args.input, args.format, args.width, args.height, class_mapping)
    
    # 显示跳过的类别总结
    if _warned_skipped_classes:
        print(f"\n跳过的类别总数: {len(_warned_skipped_classes)}")
        print(f"跳过的类别: {', '.join(sorted(_warned_skipped_classes))}")