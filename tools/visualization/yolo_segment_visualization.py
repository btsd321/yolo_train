# YOLO segment可视化工具
# 功能：可视化YOLO分割格式的标注文件
# 输入：包含图片和txt分割标注文件的文件夹
# 显示：半透明掩码 + 边界线条 + 类别标签

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


def parse_yolo_segment(txt_file_path):
    """
    解析YOLO分割格式的标注文件
    格式: class_id x1 y1 x2 y2 x3 y3 ... xn yn (归一化坐标)
    返回: [(class_id, [(x1,y1), (x2,y2), ...]), ...]
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
        if len(parts) >= 7:  # 至少需要class_id + 3个点(6个坐标)
            class_id = int(parts[0])
            
            # 解析多边形点坐标
            coords = [float(x) for x in parts[1:]]
            
            # 确保坐标数量是偶数（x,y对）
            if len(coords) % 2 != 0:
                print(f"警告: 跳过无效的坐标数据（坐标数量不是偶数）")
                continue
            
            # 将坐标转换为点列表
            points = []
            for i in range(0, len(coords), 2):
                points.append((coords[i], coords[i+1]))
            
            annotations.append((class_id, points))
    
    return annotations


def draw_yolo_segment(image, annotations, class_names=None, alpha=0.4):
    """
    在图片上绘制YOLO分割格式的掩码
    参数:
        image: 输入图像
        annotations: [(class_id, [(x1,y1), (x2,y2), ...]), ...]
        class_names: 类别名称字典 {class_id: class_name}
        alpha: 掩码透明度 (0.0-1.0)
    """
    img_height, img_width = image.shape[:2]
    
    # 根据图像分辨率自适应调整参数
    base_size = 1920
    scale_factor = min(img_width, img_height) / base_size
    
    # 自适应线条粗细
    line_thickness = max(1, min(8, int(2 * scale_factor)))
    
    # 自适应字体参数
    font_scale = max(0.3, min(2.0, 0.6 * scale_factor))
    font_thickness = max(1, min(5, int(2 * scale_factor)))
    padding = max(2, int(5 * scale_factor))
    
    # 创建一个overlay图层用于绘制半透明掩码
    overlay = image.copy()
    
    for class_id, points in annotations:
        # 将归一化坐标转换为像素坐标
        pixel_points = []
        for x, y in points:
            px = int(x * img_width)
            py = int(y * img_height)
            pixel_points.append([px, py])
        
        pixel_points = np.array(pixel_points, dtype=np.int32)
        
        # 获取颜色（支持超过20个类别）
        if class_id >= 20:
            if class_id not in _warned_class_ids:
                print(f"⚠ 警告: 类别ID {class_id} 超过20，将循环使用颜色（使用颜色索引 {class_id % 20}）")
                _warned_class_ids.add(class_id)
            color = CLASS_COLORS.get(class_id % 20, (0, 255, 255))
        else:
            color = CLASS_COLORS.get(class_id, (0, 255, 255))
        
        # 绘制填充的多边形（半透明掩码）
        cv2.fillPoly(overlay, [pixel_points], color)
        
        # 绘制多边形边界（实线）
        cv2.polylines(image, [pixel_points], isClosed=True, color=color, thickness=line_thickness)
        
        # 计算多边形的中心点或边界框来放置标签
        x_coords = [p[0] for p in pixel_points]
        y_coords = [p[1] for p in pixel_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 标签放在边界框左上角
        label_x = x_min
        label_y = y_min
        
        # 获取类别名称
        if class_names and class_id in class_names:
            class_name = class_names[class_id]
            label = f'{class_name} (ID:{class_id})'
        else:
            label = f'ID:{class_id}'
        
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # 绘制标签背景
        cv2.rectangle(
            image,
            (label_x, label_y - text_height - baseline - padding),
            (label_x + text_width + padding, label_y),
            color,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            image,
            label,
            (label_x + padding // 2, label_y - baseline - padding // 2),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    # 将overlay与原图混合，实现半透明效果
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    return image


print("✓ 步骤2完成: 半透明掩码绘制函数")
print("已有知识:")
print("  - draw_yolo_segment(): 绘制分割掩码")
print("    · 使用cv2.fillPoly()填充多边形")
print("    · 使用cv2.polylines()绘制边界")
print("    · 使用cv2.addWeighted()实现半透明效果")
print("    · alpha参数控制透明度(默认0.4)")
print("    · 自适应线条粗细和字体大小")


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


print("✓ 步骤3完成: 文件查找和图像查看器")
print("已有知识:")
print("  - find_image_annotation_pairs(): 查找图片和标注对")
print("  - ImageViewer类: 支持鼠标缩放和拖拽")
print("    · 鼠标滚轮: 缩放")
print("    · 鼠标拖拽: 平移")
print("    · reset_view(): 重置视图")


def visualize_yolo_segment_dataset(folder_path, class_names=None, alpha=0.4, window_name='YOLO Segment Visualization'):
    """
    可视化YOLO分割数据集
    参数:
        folder_path: 包含图片和标注文件的文件夹路径
        class_names: 类别名称字典 {class_id: class_name}
        alpha: 掩码透明度 (0.0-1.0)
        window_name: 窗口名称
    """
    # 查找所有成对的图片和标注
    pairs = find_image_annotation_pairs(folder_path)
    
    if not pairs:
        print(f"警告: 在 {folder_path} 中没有找到成对的图片和标注文件")
        return
    
    print(f"找到 {len(pairs)} 对图片和标注文件")
    print(f"掩码透明度: {alpha}")
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
        annotations = parse_yolo_segment(txt_path)
        
        # 在图片上绘制分割掩码
        vis_image = image.copy()
        vis_image = draw_yolo_segment(vis_image, annotations, class_names, alpha)
        
        # 根据图像分辨率自适应调整信息栏参数
        base_size = 1920
        scale_factor = min(vis_image.shape[1], vis_image.shape[0]) / base_size
        info_height = max(30, int(40 * scale_factor))
        info_font_scale = max(0.4, min(1.5, 0.7 * scale_factor))
        info_font_thickness = max(1, min(4, int(2 * scale_factor)))
        info_padding = max(5, int(10 * scale_factor))
        
        # 添加图片信息
        img_name = os.path.basename(img_path)
        info_text = f"[{current_idx + 1}/{len(pairs)}] {img_name} - {len(annotations)} objects (alpha={alpha})"
        
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
        description='YOLO分割可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 不使用类别名称（显示ID）
  python %(prog)s -i ./data/annotations
  
  # 使用类别名称文件
  python %(prog)s -i ./data/annotations -n classes.txt
  
  # 调整掩码透明度
  python %(prog)s -i ./data/annotations -a 0.6

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
        help='输入文件夹路径，包含图片和对应的txt分割标注文件'
    )
    
    parser.add_argument(
        '-n', '--names',
        type=str,
        default=None,
        help='类别名称文件路径（可选），格式: 每行为 "id name"'
    )
    
    parser.add_argument(
        '-a', '--alpha',
        type=float,
        default=0.4,
        help='掩码透明度 (0.0-1.0)，默认0.4'
    )
    
    args = parser.parse_args()
    
    # 验证alpha参数
    if not 0.0 <= args.alpha <= 1.0:
        print("错误: alpha参数必须在0.0到1.0之间")
        exit(1)
    
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
    
    visualize_yolo_segment_dataset(args.input, class_names, args.alpha)


print("\n" + "=" * 60)
print("✓ 步骤4完成: 主可视化函数和命令行接口")
print("=" * 60)
print("\n📚 完整知识总结:")
print("\n1. 数据结构:")
print("   - YOLO分割格式: class_id x1 y1 x2 y2 ... xn yn (归一化)")
print("   - 解析结果: [(class_id, [(x1,y1), (x2,y2), ...]), ...]")
print("\n2. 核心函数:")
print("   - parse_yolo_segment(): 解析分割标注")
print("   - draw_yolo_segment(): 绘制半透明掩码")
print("   - visualize_yolo_segment_dataset(): 主可视化函数")
print("\n3. 可视化技术:")
print("   - cv2.fillPoly(): 填充多边形掩码")
print("   - cv2.polylines(): 绘制多边形边界")
print("   - cv2.addWeighted(): 混合实现半透明")
print("   - alpha参数控制透明度")
print("\n4. 交互功能:")
print("   - 鼠标滚轮缩放")
print("   - 鼠标拖拽平移")
print("   - 键盘切换图片")
print("   - 自适应分辨率")
print("\n5. 参数:")
print("   - -i/--input: 输入文件夹")
print("   - -n/--names: 类别名称文件")
print("   - -a/--alpha: 掩码透明度 (0.0-1.0)")
print("\n✅ YOLO分割可视化工具开发完成!")
print("=" * 60)
