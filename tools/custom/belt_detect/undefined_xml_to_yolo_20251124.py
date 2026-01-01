# 本脚本功能是将xml文件中的标注信息提取出来然后转换为yolo格式的txt标注文件
# 输入：图片和xml文件所在文件夹路径
# 输出：生成对应的yolo格式txt标注文件，保存在输入文件夹中
# 已知信息：要保存的信息有三种一种是包裹delivery，一种是面单ExpressBillSeg，还有一种是标牌数字number
# 已知信息：delivery和ExpressBillSeg是按照多边形标注的，number是按照矩形框标注的，转换时提取多边形的外接矩形框转换为yolo格式
# 输出时class_id：0表示number，1表示delivery，2表示ExpressBillSeg

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import argparse


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


def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    将像素坐标转换为YOLO格式 (归一化的中心点坐标和宽高)
    返回: (x_center, y_center, width, height) 归一化后的值
    """
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height


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


def parse_xml_to_yolo(xml_file_path, input_folder, img_width=None, img_height=None):
    """
    解析单个XML文件并转换为YOLO格式
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
    
    # 处理矩形标注 (number类别)
    for rect in root.findall('rect'):
        label_type = rect.get('labelType', '')
        if 'number' in label_type:
            class_id = 0  # number对应class_id=0
            
            x = float(rect.get('x'))
            y = float(rect.get('y'))
            w = float(rect.get('w'))
            h = float(rect.get('h'))
            
            # 计算矩形框的边界
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            
            # 转换为YOLO格式
            x_center, y_center, width, height = convert_to_yolo_format(
                xmin, ymin, xmax, ymax, img_width, img_height
            )
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 处理多边形标注 (delivery和ExpressBillSeg类别)
    for area in root.findall('area'):
        label_type = area.get('labelType', '')
        
        if 'delivery' in label_type:
            class_id = 1  # delivery对应class_id=1
        elif 'ExpressBillSeg' in label_type:
            class_id = 2  # ExpressBillSeg对应class_id=2
        else:
            continue
        
        points_str = area.get('points', '')
        if not points_str:
            continue
        
        # 解析多边形点
        points = parse_points(points_str)
        if len(points) < 3:
            continue
        
        # 获取外接矩形框
        xmin, ymin, xmax, ymax = get_bounding_box(points)
        
        # 转换为YOLO格式
        x_center, y_center, width, height = convert_to_yolo_format(
            xmin, ymin, xmax, ymax, img_width, img_height
        )
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


def convert_folder(input_folder, img_width=None, img_height=None):
    """
    转换文件夹中所有的XML文件为YOLO格式
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"错误: 文件夹不存在: {input_folder}")
        return
    
    xml_files = list(input_path.glob('*.xml'))
    if not xml_files:
        print(f"警告: 在 {input_folder} 中没有找到XML文件")
        return
    
    print(f"找到 {len(xml_files)} 个XML文件")
    converted_count = 0
    
    for xml_file in xml_files:
        try:
            # 解析XML并转换为YOLO格式
            yolo_annotations = parse_xml_to_yolo(str(xml_file), input_folder, img_width, img_height)
            
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


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='将XML格式的标注文件转换为YOLO格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python %(prog)s -i ./data/annotations
  python %(prog)s -i ./data/annotations -w 1920 -h 1080
  
类别映射:
  0: number (标牌数字)
  1: delivery (包裹)
  2: ExpressBillSeg (面单)
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=r"D:/Project/yolo_train/Data/waybill_perception/1124包裹标牌交付汇总",
        help='输入文件夹路径，包含XML文件和对应的图片文件 (默认: D:\\Project\\yolo_train\\Data\\waybill_perception\\1124包裹标牌交付汇总)'
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
    
    args = parser.parse_args()
    
    # 执行转换
    print(f"输入文件夹: {args.input}")
    if args.width and args.height:
        print(f"指定图片尺寸: {args.width}x{args.height}")
    else:
        print("图片尺寸: 自动检测")
    print("-" * 60)
    
    convert_folder(args.input, args.width, args.height)