# YOLO bbox可视化工具
# 功能：可视化YOLO格式的标注文件
# 输入：包含图片和txt标注文件的文件夹
# 操作：按'c'或空格键切换下一张，按'q'退出

import cv2
import os
import argparse
from pathlib import Path
import numpy as np


# 类别名称映射（根据实际情况修改）
CLASS_NAMES = {
    0: 'delivery',      # 软包裹
    1: 'box',           # 硬纸盒
    2: 'ExpressBillSeg',# 面单
    3: 'BarCode',       # 条形码
    4: '2DCode'         # 二维码
}

# 每个类别对应的颜色（BGR格式）
CLASS_COLORS = {
    0: (0, 255, 0),     # 绿色 - delivery
    1: (255, 0, 0),     # 蓝色 - box
    2: (0, 0, 255),     # 红色 - ExpressBillSeg
    3: (255, 255, 0),   # 青色 - BarCode
    4: (255, 0, 255)    # 洋红 - 2DCode
}


def parse_yolo_bbox(txt_file_path):
    """
    解析YOLO格式的bbox标注文件
    返回: [(class_id, x_center, y_center, width, height), ...]
    """
    annotations = []
    
    if not os.path.exists(txt_file_path):
        return annotations
    
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            annotations.append((class_id, x_center, y_center, width, height))
    
    return annotations


def draw_yolo_bbox(image, annotations):
    """
    在图片上绘制YOLO格式的bbox
    """
    img_height, img_width = image.shape[:2]
    
    for class_id, x_center, y_center, width, height in annotations:
        # 将归一化坐标转换为像素坐标
        x_center_px = int(x_center * img_width)
        y_center_px = int(y_center * img_height)
        width_px = int(width * img_width)
        height_px = int(height * img_height)
        
        # 计算左上角和右下角坐标
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        # 获取颜色和类别名称
        color = CLASS_COLORS.get(class_id, (0, 255, 255))
        class_name = CLASS_NAMES.get(class_id, f'Class_{class_id}')
        
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        label = f'{class_name} (ID:{class_id})'
        
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # 绘制标签背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            color,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            image,
            label,
            (x1 + 2, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return image


def find_image_annotation_pairs(folder_path):
    """
    查找文件夹中成对的图片和标注文件
    返回: [(image_path, txt_path), ...]
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        return []
    
    # 支持的图片格式（不区分大小写）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    pairs = []
    seen_files = set()  # 用于去重
    
    # 遍历文件夹中的所有文件
    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        
        # 检查是否是图片文件（不区分大小写）
        if file_path.suffix.lower() in image_extensions:
            # 查找对应的txt文件
            txt_file = file_path.with_suffix('.txt')
            if txt_file.exists():
                # 使用文件的stem（不含扩展名的文件名）作为唯一标识
                file_stem = file_path.stem.lower()
                if file_stem not in seen_files:
                    seen_files.add(file_stem)
                    pairs.append((str(file_path), str(txt_file)))
    
    # 按文件名排序
    pairs.sort(key=lambda x: x[0])
    
    return pairs


def visualize_yolo_dataset(folder_path, window_name='YOLO BBox Visualization'):
    """
    可视化YOLO数据集
    """
    # 查找所有成对的图片和标注
    pairs = find_image_annotation_pairs(folder_path)
    
    if not pairs:
        print(f"警告: 在 {folder_path} 中没有找到成对的图片和标注文件")
        return
    
    print(f"找到 {len(pairs)} 对图片和标注文件")
    print("操作说明:")
    print("  按 'c' 或 空格键: 切换到下一张图片")
    print("  按 'b': 返回上一张图片")
    print("  按 'q' 或 ESC: 退出")
    print("-" * 60)
    
    current_idx = 0
    
    while True:
        img_path, txt_path = pairs[current_idx]
        
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"错误: 无法读取图片 {img_path}")
            current_idx = (current_idx + 1) % len(pairs)
            continue
        
        # 解析标注
        annotations = parse_yolo_bbox(txt_path)
        
        # 在图片上绘制bbox
        vis_image = image.copy()
        vis_image = draw_yolo_bbox(vis_image, annotations)
        
        # 添加图片信息
        img_name = os.path.basename(img_path)
        info_text = f"[{current_idx + 1}/{len(pairs)}] {img_name} - {len(annotations)} objects"
        
        # 在图片顶部添加信息栏
        info_height = 40
        info_bar = np.zeros((info_height, vis_image.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            info_bar,
            info_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # 将信息栏和图片拼接
        vis_image = np.vstack([info_bar, vis_image])
        
        # 自动调整窗口大小以适应屏幕
        screen_height = 1080  # 假设屏幕高度
        if vis_image.shape[0] > screen_height:
            scale = screen_height / vis_image.shape[0]
            new_width = int(vis_image.shape[1] * scale)
            new_height = int(vis_image.shape[0] * scale)
            vis_image = cv2.resize(vis_image, (new_width, new_height))
        
        # 显示图片
        cv2.imshow(window_name, vis_image)
        
        # 打印当前图片信息
        print(f"[{current_idx + 1}/{len(pairs)}] {img_name} - {len(annotations)} 个目标")
        
        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' 或 ESC 退出
            print("退出可视化")
            break
        elif key == ord('c') or key == ord(' '):  # 'c' 或 空格 下一张
            current_idx = (current_idx + 1) % len(pairs)
        elif key == ord('b'):  # 'b' 上一张
            current_idx = (current_idx - 1) % len(pairs)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLO bbox可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python %(prog)s -i ./data/annotations
  python %(prog)s -i D:/Project/yolo_train/Data/waybill_perception

操作说明:
  按 'c' 或 空格键: 切换到下一张图片
  按 'b': 返回上一张图片
  按 'q' 或 ESC: 退出

类别映射:
  0: delivery (软包裹) - 绿色
  1: box (硬纸盒) - 蓝色
  2: ExpressBillSeg (面单) - 红色
  3: BarCode (条形码) - 青色
  4: 2DCode (二维码) - 洋红
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=r"D:/Project/yolo_train/Data/waybill_perception",
        help='输入文件夹路径，包含图片和对应的txt标注文件'
    )
    
    args = parser.parse_args()
    
    print(f"输入文件夹: {args.input}")
    print("-" * 60)
    
    visualize_yolo_dataset(args.input)
