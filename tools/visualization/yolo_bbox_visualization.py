# YOLO bbox可视化工具
# 功能：可视化YOLO格式的标注文件
# 输入：包含图片和txt标注文件的文件夹
# 操作：按'c'或空格键切换下一张，按'q'退出

import cv2
import os
import argparse
from pathlib import Path
import numpy as np


# 每个类别对应的颜色（BGR格式）- 20种颜色
CLASS_COLORS = {
    0: (0, 255, 0),       # 绿色
    1: (255, 0, 0),       # 蓝色
    2: (0, 0, 255),       # 红色
    3: (255, 255, 0),     # 青色
    4: (255, 0, 255),     # 洋红
    5: (0, 255, 255),     # 黄色
    6: (0, 165, 255),     # 橙色
    7: (128, 0, 128),     # 紫色
    8: (203, 192, 255),   # 粉红
    9: (42, 42, 165),     # 棕色
    10: (0, 128, 0),      # 深绿
    11: (139, 0, 0),      # 深蓝
    12: (0, 0, 139),      # 深红
    13: (255, 191, 0),    # 天蓝
    14: (0, 255, 191),    # 石灰绿
    15: (80, 127, 255),   # 珊瑚色
    16: (0, 215, 255),    # 金色
    17: (139, 139, 0),    # 深青
    18: (139, 0, 139),    # 深洋红
    19: (0, 140, 255)     # 深橙
}


def load_class_names(names_file_path):
    """
    从txt文件加载类别名称
    格式: 每行为 "id name"
    返回: {class_id: class_name, ...}
    """
    class_names = {}
    
    if not names_file_path or not os.path.exists(names_file_path):
        return class_names
    
    try:
        with open(names_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    class_id = int(parts[0])
                    class_name = parts[1]
                    class_names[class_id] = class_name
        
        print(f"成功加载 {len(class_names)} 个类别名称")
    except Exception as e:
        print(f"警告: 加载类别名称文件时出错: {e}")
    
    return class_names


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


def draw_yolo_bbox(image, annotations, class_names=None):
    """
    在图片上绘制YOLO格式的bbox
    参数:
        class_names: 类别名称字典 {class_id: class_name}
    """
    img_height, img_width = image.shape[:2]
    
    # 根据图像分辨率自适应调整参数
    # 以1920x1080为基准
    base_size = 1920
    scale_factor = min(img_width, img_height) / base_size
    
    # 自适应线条粗细（最小为1，最大为10）
    line_thickness = max(1, min(10, int(2 * scale_factor)))
    
    # 自适应字体大小（最小为0.3，最大为2.0）
    font_scale = max(0.3, min(2.0, 0.6 * scale_factor))
    
    # 自适应字体粗细（最小为1，最大为5）
    font_thickness = max(1, min(5, int(2 * scale_factor)))
    
    # 自适应边距
    padding = max(2, int(5 * scale_factor))
    
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
        if class_names and class_id in class_names:
            class_name = class_names[class_id]
            label = f'{class_name} (ID:{class_id})'
        else:
            label = f'ID:{class_id}'
        
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # 绘制标签背景
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - padding),
            (x1 + text_width + padding, y1),
            color,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            image,
            label,
            (x1 + padding // 2, y1 - baseline - padding // 2),
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


class ImageViewer:
    """支持鼠标缩放和拖拽的图像查看器"""
    def __init__(self, window_name):
        self.window_name = window_name
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        self.original_image = None
        self.display_image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调函数"""
        if event == cv2.EVENT_MOUSEWHEEL:
            # 鼠标滚轮缩放
            if flags > 0:  # 向上滚动，放大
                self.zoom_scale *= 1.1
            else:  # 向下滚动，缩小
                self.zoom_scale *= 0.9
            
            # 限制缩放范围
            self.zoom_scale = max(0.1, min(10.0, self.zoom_scale))
            self.update_display()
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # 开始拖拽
            self.dragging = True
            self.last_x = x
            self.last_y = y
            
        elif event == cv2.EVENT_LBUTTONUP:
            # 结束拖拽
            self.dragging = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # 拖拽移动
            if self.dragging:
                dx = x - self.last_x
                dy = y - self.last_y
                self.offset_x += dx
                self.offset_y += dy
                self.last_x = x
                self.last_y = y
                self.update_display()
    
    def set_image(self, image):
        """设置要显示的图像"""
        self.original_image = image.copy()
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update_display()
    
    def update_display(self):
        """更新显示的图像"""
        if self.original_image is None:
            return
        
        # 应用缩放
        if self.zoom_scale != 1.0:
            new_width = int(self.original_image.shape[1] * self.zoom_scale)
            new_height = int(self.original_image.shape[0] * self.zoom_scale)
            scaled_image = cv2.resize(self.original_image, (new_width, new_height))
        else:
            scaled_image = self.original_image.copy()
        
        # 创建显示画布（保持原始图像大小）
        canvas = np.zeros_like(self.original_image)
        
        # 计算粘贴位置
        h, w = scaled_image.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]
        
        # 应用偏移量
        x_start = self.offset_x
        y_start = self.offset_y
        
        # 计算源图像和目标画布的有效区域
        src_x1 = max(0, -x_start)
        src_y1 = max(0, -y_start)
        src_x2 = min(w, canvas_w - x_start)
        src_y2 = min(h, canvas_h - y_start)
        
        dst_x1 = max(0, x_start)
        dst_y1 = max(0, y_start)
        dst_x2 = min(canvas_w, x_start + w)
        dst_y2 = min(canvas_h, y_start + h)
        
        # 粘贴图像
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_image[src_y1:src_y2, src_x1:src_x2]
        
        self.display_image = canvas
        cv2.imshow(self.window_name, self.display_image)
    
    def reset_view(self):
        """重置视图"""
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update_display()


def visualize_yolo_dataset(folder_path, class_names=None, window_name='YOLO BBox Visualization'):
    """
    可视化YOLO数据集
    参数:
        class_names: 类别名称字典 {class_id: class_name}
    """
    # 查找所有成对的图片和标注
    pairs = find_image_annotation_pairs(folder_path)
    
    if not pairs:
        print(f"警告: 在 {folder_path} 中没有找到成对的图片和标注文件")
        return
    
    print(f"找到 {len(pairs)} 对图片和标注文件")
    print("操作说明:")
    print("  鼠标滚轮: 放大/缩小图像")
    print("  鼠标左键拖拽: 移动图像")
    print("  按 'r': 重置视图（恢复原始大小和位置）")
    print("  按 'c' 或 空格键: 切换到下一张图片")
    print("  按 'b': 返回上一张图片")
    print("  按 'q' 或 ESC: 退出")
    print("-" * 60)
    
    # 创建图像查看器
    viewer = ImageViewer(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, viewer.mouse_callback)
    
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
        vis_image = draw_yolo_bbox(vis_image, annotations, class_names)
        
        # 根据图像分辨率自适应调整信息栏参数
        base_size = 1920
        scale_factor = min(vis_image.shape[1], vis_image.shape[0]) / base_size
        info_height = max(30, int(40 * scale_factor))
        info_font_scale = max(0.4, min(1.5, 0.7 * scale_factor))
        info_font_thickness = max(1, min(4, int(2 * scale_factor)))
        info_padding = max(5, int(10 * scale_factor))
        
        # 添加图片信息
        img_name = os.path.basename(img_path)
        info_text = f"[{current_idx + 1}/{len(pairs)}] {img_name} - {len(annotations)} objects"
        
        # 在图片顶部添加信息栏
        info_bar = np.zeros((info_height, vis_image.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            info_bar,
            info_text,
            (info_padding, info_height - info_padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            info_font_scale,
            (255, 255, 255),
            info_font_thickness
        )
        
        # 将信息栏和图片拼接
        vis_image = np.vstack([info_bar, vis_image])
        
        # 自动调整窗口大小以适应屏幕
        screen_height = 1080  # 假设屏幕高度
        initial_scale = 1.0
        if vis_image.shape[0] > screen_height:
            initial_scale = screen_height / vis_image.shape[0]
            new_width = int(vis_image.shape[1] * initial_scale)
            new_height = int(vis_image.shape[0] * initial_scale)
            vis_image = cv2.resize(vis_image, (new_width, new_height))
        
        # 设置图像到查看器
        viewer.set_image(vis_image)
        
        # 打印当前图片信息
        print(f"[{current_idx + 1}/{len(pairs)}] {img_name} - {len(annotations)} 个目标")
        
        # 等待按键
        while True:
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' 或 ESC 退出
                print("退出可视化")
                cv2.destroyAllWindows()
                return
            elif key == ord('c') or key == ord(' '):  # 'c' 或 空格 下一张
                current_idx = (current_idx + 1) % len(pairs)
                break
            elif key == ord('b'):  # 'b' 上一张
                current_idx = (current_idx - 1) % len(pairs)
                break
            elif key == ord('r'):  # 'r' 重置视图
                viewer.reset_view()
                print("视图已重置")
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLO bbox可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 不使用类别名称（显示ID）
  python %(prog)s -i ./data/annotations
  
  # 使用类别名称文件
  python %(prog)s -i ./data/annotations -n classes.txt

操作说明:
  鼠标滚轮: 放大/缩小图像
  鼠标左键拖拽: 移动图像
  按 'r': 重置视图（恢复原始大小和位置）
  按 'c' 或 空格键: 切换到下一张图片
  按 'b': 返回上一张图片
  按 'q' 或 ESC: 退出

类别名称文件格式（可选）:
  每行格式: id name
  例如:
    0 delivery
    1 box
    2 ExpressBillSeg
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=r"D:/Project/yolo_train/Data/waybill_perception",
        help='输入文件夹路径，包含图片和对应的txt标注文件'
    )
    
    parser.add_argument(
        '-n', '--names',
        type=str,
        default=None,
        help='类别名称文件路径（可选），格式: 每行为 "id name"'
    )
    
    args = parser.parse_args()
    
    # 加载类别名称（如果提供）
    class_names = None
    if args.names:
        class_names = load_class_names(args.names)
        if class_names:
            print(f"类别名称: {len(class_names)} 个类别")
        else:
            print("未加载类别名称，将显示数字ID")
    else:
        print("未指定类别名称文件，将显示数字ID")
    
    print(f"输入文件夹: {args.input}")
    print("-" * 60)
    
    visualize_yolo_dataset(args.input, class_names)
