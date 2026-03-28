# YOLO OBB旋转框可视化工具
# 功能：可视化YOLO OBB格式的标注文件
# 输入：包含图片和txt标注文件的文件夹
# YOLO OBB格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标，4个角点顺序排列)
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

# 用于跟踪已警告的类别ID
_warned_class_ids = set()


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


def parse_yolo_obb(txt_file_path):
    """
    解析YOLO OBB格式的标注文件
    格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标)
    返回: [(class_id, [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]), ...]
    """
    annotations = []

    if not os.path.exists(txt_file_path):
        return annotations

    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 9:
            print(f"  警告: 第{line_num}行格式不正确（期望9个值，实际{len(parts)}个）: {line}")
            continue

        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:9]]
        points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
        annotations.append((class_id, points))

    return annotations


def draw_yolo_obb(image, annotations, class_names=None):
    """
    在图片上绘制YOLO OBB格式的旋转框
    参数:
        annotations: [(class_id, [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]), ...]
        class_names: 类别名称字典 {class_id: class_name}
    """
    img_height, img_width = image.shape[:2]

    # 根据图像分辨率自适应调整参数（以1920x1080为基准）
    base_size = 1920
    scale_factor = min(img_width, img_height) / base_size

    line_thickness = max(1, min(10, int(2 * scale_factor)))
    font_scale = max(0.3, min(2.0, 0.6 * scale_factor))
    font_thickness = max(1, min(5, int(2 * scale_factor)))
    padding = max(2, int(5 * scale_factor))
    # 第一个角点标记半径
    corner_radius = max(3, int(6 * scale_factor))

    for class_id, points in annotations:
        # 将归一化坐标转换为像素坐标
        pts_px = np.array(
            [[int(x * img_width), int(y * img_height)] for x, y in points],
            dtype=np.int32
        )

        # 获取颜色（超过20个类别时循环使用）
        if class_id >= 20:
            if class_id not in _warned_class_ids:
                print(f"  ⚠ 警告: 类别ID {class_id} 超过20，循环使用颜色（索引 {class_id % 20}）")
                _warned_class_ids.add(class_id)
            color = CLASS_COLORS.get(class_id % 20, (0, 255, 255))
        else:
            color = CLASS_COLORS.get(class_id, (0, 255, 255))

        # 绘制旋转框（四边形轮廓）
        cv2.polylines(image, [pts_px], isClosed=True, color=color, thickness=line_thickness)

        # 在第一个角点绘制实心圆标记方向
        cv2.circle(image, tuple(pts_px[0]), corner_radius, color, -1)

        # 获取类别名称
        if class_names and class_id in class_names:
            class_name = class_names[class_id]
            label = f'{class_name} (ID:{class_id})'
        else:
            label = f'ID:{class_id}'

        # 标签位置：取四个角点中 y 最小（最靠上）的角点
        top_idx = int(np.argmin(pts_px[:, 1]))
        label_x = pts_px[top_idx][0]
        label_y = pts_px[top_idx][1]

        # 计算文本尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # 绘制标签背景
        bg_x1 = label_x
        bg_y1 = label_y - text_height - baseline - padding
        bg_x2 = label_x + text_width + padding
        bg_y2 = label_y

        # 防止标签超出图像顶部
        if bg_y1 < 0:
            bg_y1 = label_y
            bg_y2 = label_y + text_height + baseline + padding

        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            image,
            label,
            (bg_x1 + padding // 2, bg_y2 - baseline - padding // 2),
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

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    pairs = []
    seen_files = set()

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in image_extensions:
            txt_file = file_path.with_suffix('.txt')
            if txt_file.exists():
                file_stem = file_path.stem.lower()
                if file_stem not in seen_files:
                    seen_files.add(file_stem)
                    pairs.append((str(file_path), str(txt_file)))

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

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调函数"""
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.zoom_scale *= 1.1
            else:
                self.zoom_scale *= 0.9
            self.zoom_scale = max(0.1, min(10.0, self.zoom_scale))
            self.update_display()

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_x = x
            self.last_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.offset_x += x - self.last_x
                self.offset_y += y - self.last_y
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

        if self.zoom_scale != 1.0:
            new_w = int(self.original_image.shape[1] * self.zoom_scale)
            new_h = int(self.original_image.shape[0] * self.zoom_scale)
            scaled = cv2.resize(self.original_image, (new_w, new_h))
        else:
            scaled = self.original_image.copy()

        canvas = np.zeros_like(self.original_image)
        h, w = scaled.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]

        src_x1 = max(0, -self.offset_x)
        src_y1 = max(0, -self.offset_y)
        src_x2 = min(w, canvas_w - self.offset_x)
        src_y2 = min(h, canvas_h - self.offset_y)

        dst_x1 = max(0, self.offset_x)
        dst_y1 = max(0, self.offset_y)
        dst_x2 = min(canvas_w, self.offset_x + w)
        dst_y2 = min(canvas_h, self.offset_y + h)

        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled[src_y1:src_y2, src_x1:src_x2]

        cv2.imshow(self.window_name, canvas)

    def reset_view(self):
        """重置视图"""
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update_display()


def visualize_yolo_obb_dataset(folder_path, class_names=None, window_name='YOLO OBB Visualization'):
    """
    可视化YOLO OBB数据集
    参数:
        class_names: 类别名称字典 {class_id: class_name}
    """
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
    print("  注: 旋转框第一个角点以实心圆标记")
    print("-" * 60)

    viewer = ImageViewer(window_name)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, viewer.mouse_callback)

    current_idx = 0

    while True:
        img_path, txt_path = pairs[current_idx]

        image = cv2.imread(img_path)
        if image is None:
            print(f"错误: 无法读取图片 {img_path}")
            current_idx = (current_idx + 1) % len(pairs)
            continue

        annotations = parse_yolo_obb(txt_path)

        vis_image = image.copy()
        vis_image = draw_yolo_obb(vis_image, annotations, class_names)

        # 信息栏自适应参数
        base_size = 1920
        scale_factor = min(vis_image.shape[1], vis_image.shape[0]) / base_size
        info_height = max(30, int(40 * scale_factor))
        info_font_scale = max(0.4, min(1.5, 0.7 * scale_factor))
        info_font_thickness = max(1, min(4, int(2 * scale_factor)))
        info_padding = max(5, int(10 * scale_factor))

        img_name = os.path.basename(img_path)
        info_text = f"[{current_idx + 1}/{len(pairs)}] {img_name} - {len(annotations)} objects (OBB)"

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

        vis_image = np.vstack([info_bar, vis_image])

        # 自动缩放以适应屏幕高度
        screen_height = 1080
        if vis_image.shape[0] > screen_height:
            scale = screen_height / vis_image.shape[0]
            vis_image = cv2.resize(
                vis_image,
                (int(vis_image.shape[1] * scale), int(vis_image.shape[0] * scale))
            )

        viewer.set_image(vis_image)

        print(f"[{current_idx + 1}/{len(pairs)}] {img_name} - {len(annotations)} 个旋转框")

        while True:
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q') or key == 27:  # 'q' 或 ESC 退出
                print("退出可视化")
                cv2.destroyAllWindows()
                return
            elif key == ord('c') or key == ord(' '):  # 下一张
                current_idx = (current_idx + 1) % len(pairs)
                break
            elif key == ord('b'):  # 上一张
                current_idx = (current_idx - 1) % len(pairs)
                break
            elif key == ord('r'):  # 重置视图
                viewer.reset_view()
                print("视图已重置")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLO OBB旋转框可视化工具',
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

YOLO OBB标注格式:
  每行: class_id x1 y1 x2 y2 x3 y3 x4 y4
  坐标为归一化值（0~1），4个角点按顺序排列
  第一个角点以实心圆标记，便于识别方向

类别名称文件格式（可选）:
  每行格式: id name
  例如:
    0 ship
    1 plane
    2 car
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default=r"D:/Project/yolo_train/Data",
        help='输入文件夹路径，包含图片和对应的txt标注文件'
    )

    parser.add_argument(
        '-n', '--names',
        type=str,
        default=None,
        help='类别名称文件路径（可选），格式: 每行为 "id name"'
    )

    args = parser.parse_args()

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

    visualize_yolo_obb_dataset(args.input, class_names)
