'''
本文件读取传送带检测的标定结果，并修正检测到的标牌位置，具体步骤如下：
1. 输入传送带检测图片和标注json所在文件夹路径
2. 遍历文件夹，读取匹配的图片
3. 使用预训练的yolo模型检测标牌矩形框位置（对象类别 ID (0: 数字标牌, 1: 包裹)）
4. 将标牌矩形框扩大一定比例（如10%，可调），截取扩大后的标牌图像进行图像处理
5. 图像处理步骤：1. 转换成灰度图、二值化（阈值可调），二值化后偏白的部分为标牌，计算标牌的外接矩形（要求外接矩形为yolo box的样式即box横竖要和图像的x轴、y轴平行）
6. 将修正后的标牌位置保存到输出文件夹路径下，格式与输入文件夹中图片同名json一致，名称也一致同时把图片也复制一份到输出文件夹
6. 备注：所有参数要有默认值尽量可调，使用argparse库管理输入
'''

import json
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def load_json_annotation(json_path: Path) -> Optional[Dict]:
    """
    加载X-AnyLabeling格式的JSON标注文件
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        标注数据字典或None
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON {json_path}: {e}")
        return None


def save_json_annotation(json_data: Dict, output_path: Path):
    """
    保存X-AnyLabeling格式的JSON标注文件
    
    Args:
        json_data: 标注数据字典
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving JSON {output_path}: {e}")


def calculate_iou(box1: Tuple[float, float, float, float], 
                 box2: Tuple[float, float, float, float]) -> float:
    """
    计算两个矩形框的IoU
    
    Args:
        box1: 第一个框 (x, y, w, h)
        box2: 第二个框 (x, y, w, h)
    
    Returns:
        IoU值 (0-1之间)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 计算交集
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算并集
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def expand_box(x, y, w, h, expand_ratio: float, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    扩大矩形框
    
    Args:
        x, y, w, h: 原始矩形框（左上角坐标和宽高）
        expand_ratio: 扩大比例（如0.1表示扩大10%）
        img_width, img_height: 图像尺寸
    
    Returns:
        扩大后的矩形框 (x, y, w, h)
    """
    # 计算扩大量
    expand_w = w * expand_ratio
    expand_h = h * expand_ratio
    
    # 扩大后的坐标
    new_x = max(0, x - expand_w / 2)
    new_y = max(0, y - expand_h / 2)
    new_w = min(img_width - new_x, w + expand_w)
    new_h = min(img_height - new_y, h + expand_h)
    
    return int(new_x), int(new_y), int(new_w), int(new_h)


def find_sign_contour(roi_image: np.ndarray, detection_box: Tuple[int, int, int, int],
                     debug: bool = False, image_name: str = "",
                     original_box: Optional[Tuple[int, int, int, int]] = None,
                     morph_kernel_size: int = 5,
                     morph_iterations: int = 1,
                     thresh_offset: int = 30,
                     min_sign_brightness: int = 140,
                     light_mode: str = 'uniform') -> Optional[Tuple[int, int, int, int, dict, bool]]:
    """
    在ROI图像中查找标牌的外接矩形
    
    Args:
        roi_image: ROI图像（BGR格式）
        detection_box: 检测框在ROI中的相对坐标 (x, y, w, h)，用于计算自动阈值
        debug: 是否显示调试信息
        image_name: 图像名称（用于调试窗口标题）
        original_box: 原始标注框在ROI中的相对坐标 (x, y, w, h)，用于调试显示
        morph_kernel_size: 形态学操作核大小（默认5）
        morph_iterations: 形态学操作迭代次数（默认1）
        thresh_offset: 阈值偏移量，标牌像素值-offset作为二值化阈值（默认30）
        min_sign_brightness: 标牌最小亮度阈值，低于此值认为是背景（默认140）
        light_mode: 光照模式 uniform / vertical / horizontal
    
    Returns:
        (x, y, w, h, debug_info, accepted)
        若失败返回None
    """
    # 转换为灰度图
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # 从检测框区域计算自动阈值
    det_x, det_y, det_w, det_h = detection_box
    # 确保坐标在有效范围内
    det_x = max(0, det_x)
    det_y = max(0, det_y)
    det_w = min(det_w, gray.shape[1] - det_x)
    det_h = min(det_h, gray.shape[0] - det_y)
    
    debug_info = {
        'light_mode': light_mode,
        'thresh_offset': thresh_offset,
        'min_sign_brightness': min_sign_brightness,
        'peaks': [],
        'thresholds': [],
        'sign_pixels': []
    }

    combined_binary = None
    sign_pixel_value = None
    binary_thresh = None

    def compute_half(region: np.ndarray, label: str) -> Optional[Tuple[np.ndarray, int, int]]:
        hist_local = cv2.calcHist([region], [0], None, [256], [0, 256]).flatten()
        checked = []
        local_sign = None
        for _ in range(10):
            peak = int(np.argmax(hist_local))
            checked.append(peak)
            if peak >= min_sign_brightness:
                local_sign = peak
                break
            hist_local[peak] = 0
        debug_info['peaks'].append({label: checked})
        if local_sign is None:
            return None
        local_thresh = max(local_sign - thresh_offset, 0)
        debug_info['sign_pixels'].append({label: local_sign})
        debug_info['thresholds'].append({label: local_thresh})
        _, local_binary = cv2.threshold(region, local_thresh, 255, cv2.THRESH_BINARY)
        return local_binary, local_sign, local_thresh

    if det_w > 0 and det_h > 0:
        detection_region = gray[det_y:det_y+det_h, det_x:det_x+det_w]
        if light_mode == 'uniform':
            result = compute_half(detection_region, 'uniform')
            if result is None:
                print(f"[ERROR] {image_name}: Cannot find bright peak in uniform mode")
                return None
            combined_binary, sign_pixel_value, binary_thresh = result
        elif light_mode == 'vertical':
            h_half = detection_region.shape[0] // 2
            top = detection_region[:h_half, :]
            bottom = detection_region[h_half:, :]
            res_top = compute_half(top, 'top')
            res_bottom = compute_half(bottom, 'bottom')
            if res_top is None and res_bottom is None:
                print(f"[ERROR] {image_name}: No bright peaks (vertical mode)")
                return None
            # 初始化空白二值图
            combined_binary = np.zeros_like(detection_region)
            if res_top is not None:
                combined_binary[:h_half, :] = res_top[0]
            if res_bottom is not None:
                combined_binary[h_half:, :] = res_bottom[0]
            # 记录用于最终显示的主像素与阈值（取亮度更高的那一个）
            candidates = [r for r in [res_top, res_bottom] if r is not None]
            chosen = max(candidates, key=lambda x: x[1])
            sign_pixel_value, binary_thresh = chosen[1], chosen[2]
        elif light_mode == 'horizontal':
            w_half = detection_region.shape[1] // 2
            left = detection_region[:, :w_half]
            right = detection_region[:, w_half:]
            res_left = compute_half(left, 'left')
            res_right = compute_half(right, 'right')
            if res_left is None and res_right is None:
                print(f"[ERROR] {image_name}: No bright peaks (horizontal mode)")
                return None
            combined_binary = np.zeros_like(detection_region)
            if res_left is not None:
                combined_binary[:, :w_half] = res_left[0]
            if res_right is not None:
                combined_binary[:, w_half:] = res_right[0]
            candidates = [r for r in [res_left, res_right] if r is not None]
            chosen = max(candidates, key=lambda x: x[1])
            sign_pixel_value, binary_thresh = chosen[1], chosen[2]
        else:
            print(f"[ERROR] {image_name}: Unknown light_mode {light_mode}")
            return None
    else:
        print(f"[ERROR] {image_name}: Invalid detection box region size")
        return None

    if morph_kernel_size > 0 and morph_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        combined_binary = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)

    # 生成与 ROI 同尺寸的 binary 图: 将 combined_binary 放回原检测框位置
    full_binary = np.zeros(gray.shape, dtype=np.uint8)
    full_binary[det_y:det_y+det_h, det_x:det_x+det_w] = combined_binary

    contours, _ = cv2.findContours(full_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if debug:
            print(f"[DEBUG] {image_name}: No contours found after light_mode processing")
        return None

    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # 二值化（偏白的部分为标牌）
    _, binary = cv2.threshold(gray, binary_thresh, 255, cv2.THRESH_BINARY)
    
    # 形态学操作：先闭运算（去除内部小孔），再开运算（去除外部尖刺）
    if morph_kernel_size > 0 and morph_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
        # 闭运算：填充内部小孔
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        # 开运算：去除外部尖刺和小噪点
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    
    # 调试模式：显示三个图像并排，并获取用户输入
    accepted = True
    if debug:
        # 在二值化图像上绘制实际检测到的轮廓（用于对比）
        binary_with_contour = cv2.cvtColor(full_binary, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(binary_with_contour, [max_contour], -1, (0, 165, 255), 2)  # 橙色轮廓
        cv2.rectangle(binary_with_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色外接矩形
        # 外扩比例，用于更好地显示矩形框
        padding = 5
        
        # 第一个图像：原始标注框（如果有）
        original_img = roi_image.copy()
        if original_box is not None:
            orig_x, orig_y, orig_w, orig_h = original_box
            cv2.rectangle(original_img, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), 
                         (255, 0, 0), 2)  # 蓝色框表示原始标注
            cv2.putText(original_img, "Original Annotation", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            info_text_orig = f"Box: ({orig_x}, {orig_y}, {orig_w}, {orig_h})"
            cv2.putText(original_img, info_text_orig, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            cv2.putText(original_img, "No Original Annotation", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 第二个图像：彩色ROI + 修正后的矩形框
        color_img = roi_image.copy()
        cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色框表示修正后
        cv2.putText(color_img, "Color ROI + Fixed Box", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_text = f"Box: ({x}, {y}, {w}, {h})"
        cv2.putText(color_img, info_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        thresh_text = f"Thresh:{binary_thresh} SignPix:{sign_pixel_value} Mode:{light_mode}"
        cv2.putText(color_img, thresh_text, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 第三个图像：二值化图像 + 修正后的矩形框 + 轮廓
        cv2.rectangle(binary_with_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(binary_with_contour, "Binary + Contour(Orange) + Box(Green)", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(binary_with_contour, info_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加边距（外扩显示）
        original_padded = cv2.copyMakeBorder(original_img, padding, padding, padding, padding,
                                            cv2.BORDER_CONSTANT, value=(50, 50, 50))
        color_padded = cv2.copyMakeBorder(color_img, padding, padding, padding, padding,
                                         cv2.BORDER_CONSTANT, value=(50, 50, 50))
        binary_padded = cv2.copyMakeBorder(binary_with_contour, padding, padding, padding, padding,
                                          cv2.BORDER_CONSTANT, value=(50, 50, 50))
        
        # 确保三个图像高度一致
        h_max = max(original_padded.shape[0], color_padded.shape[0], binary_padded.shape[0])
        for img_padded in [original_padded, color_padded, binary_padded]:
            if img_padded.shape[0] < h_max:
                diff = h_max - img_padded.shape[0]
                img_padded = cv2.copyMakeBorder(img_padded, 0, diff, 0, 0,
                                               cv2.BORDER_CONSTANT, value=(50, 50, 50))
        
        # 重新赋值以应用高度调整
        imgs = [original_padded, color_padded, binary_padded]
        for i, img in enumerate(imgs):
            if img.shape[0] < h_max:
                diff = h_max - img.shape[0]
                imgs[i] = cv2.copyMakeBorder(img, 0, diff, 0, 0,
                                            cv2.BORDER_CONSTANT, value=(50, 50, 50))
        original_padded, color_padded, binary_padded = imgs
        
        # 添加分隔线
        separator = np.ones((h_max, 3, 3), dtype=np.uint8) * 100
        
        # 水平拼接三个图像
        combined = np.hstack([original_padded, separator, color_padded, separator, binary_padded])
        
        # 将拼接后的图像放大3倍以便查看
        scale_factor = 3
        new_width = int(combined.shape[1] * scale_factor)
        new_height = int(combined.shape[0] * scale_factor)
        combined_resized = cv2.resize(combined, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 显示图像
        window_name = f"Debug: {image_name} - Press 'y' to accept or 'n' to reject"
        cv2.imshow(window_name, combined_resized)
        print(f"[DEBUG] {image_name}: Showing panels (3x). Press 'y' to accept, 'n' to reject.")
        key = cv2.waitKey(0) & 0xFF
        accepted = (key == ord('y'))
        if key == ord('n'):
            print(f"[DEBUG] {image_name}: Rejected by user (pressed 'n')")
        elif key != ord('y'):
            print(f"[DEBUG] {image_name}: Invalid key pressed, treating as reject")
            accepted = False
        cv2.destroyAllWindows()
    debug_info['accepted'] = accepted
    debug_info['final_sign_pixel'] = sign_pixel_value
    debug_info['final_threshold'] = binary_thresh
    debug_info['box'] = (x, y, w, h)
    return x, y, w, h, debug_info, accepted


def points_to_xywh(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    将4个顶点坐标转换为 x, y, w, h
    
    Args:
        points: 4个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    Returns:
        (x, y, w, h) 左上角坐标和宽高
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x = min(x_coords)
    y = min(y_coords)
    w = max(x_coords) - x
    h = max(y_coords) - y
    
    return x, y, w, h


def xywh_to_points(x: float, y: float, w: float, h: float) -> List[List[float]]:
    """
    将 x, y, w, h 转换为4个顶点坐标
    
    Args:
        x, y, w, h: 左上角坐标和宽高
    
    Returns:
        4个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    return [
        [x, y],           # 左上
        [x + w, y],       # 右上
        [x + w, y + h],   # 右下
        [x, y + h]        # 左下
    ]


def process_image(image_path: Path, json_path: Path, output_dir: Path,
                 model: Optional[YOLO], expand_ratio: float = 0.1,
                 thresh_offset: int = 30, target_label: str = "number",
                 use_yolo: bool = True, device: str = 'cpu', debug: bool = False,
                 iou_threshold: float = 0.5, morph_kernel_size: int = 5,
                 morph_iterations: int = 1, min_sign_brightness: int = 140,
                 light_mode: str = 'uniform', error_log: Optional[Path] = None,
                 accept_iou_threshold: float = 0.88, judge_iou_threshold: float = 0.7) -> bool:
    """
    处理单张图像和对应的JSON标注
    
    Args:
        image_path: 图像文件路径
        json_path: JSON标注文件路径
        output_dir: 输出目录
        model: YOLO模型（如果use_yolo=True）
        expand_ratio: 矩形框扩大比例
        thresh_offset: 阈值偏移量，标牌像素值-offset作为二值化阈值（默认30）
        target_label: 要处理的标签类型
        use_yolo: 是否使用YOLO模型检测（False则使用JSON中的标注）
        device: 设备类型 ('cuda' 或 'cpu')
        debug: 是否启用调试模式
        iou_threshold: IoU阈值，用于匹配原始标注（默认0.5）
        light_mode: 光照模式 uniform/vertical/horizontal
        error_log: 错误日志路径（仅在debug下拒绝时写入）
        accept_iou_threshold: 接受阈值，YOLO模式下IoU大于此值自动接受（默认0.88）
        judge_iou_threshold: 判断阈值，YOLO模式下IoU小于此值直接拒绝（默认0.7）
    
    Returns:
        是否处理成功
    """
    try:
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Cannot read image {image_path}")
            return False
        
        img_height, img_width = image.shape[:2]
        
        # 读取JSON标注
        json_data = load_json_annotation(json_path)
        if json_data is None:
            return False
        
        # 从输入JSON中提取原始标注框（用于IoU匹配）
        original_boxes = []
        shapes = json_data.get("shapes", [])
        for shape in shapes:
            if shape.get("label") == target_label:
                points = shape.get("points", [])
                if len(points) == 4:
                    x, y, w, h = points_to_xywh(points)
                    original_boxes.append((x, y, w, h))
        
        # 获取标牌位置
        boxes_to_process = []
        
        if use_yolo and model is not None:
            # 使用YOLO模型检测（使用GPU加速）
            results = model(image, verbose=False, device=device)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    # 0: 数字标牌, 1: 包裹
                    if cls == 0:  # 只处理数字标牌
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x, y = x1, y1
                        w, h = x2 - x1, y2 - y1
                        boxes_to_process.append((x, y, w, h, None))
        else:
            # 使用JSON中的标注
            shapes = json_data.get("shapes", [])
            for idx, shape in enumerate(shapes):
                if shape.get("label") == target_label:
                    points = shape.get("points", [])
                    if len(points) == 4:
                        x, y, w, h = points_to_xywh(points)
                        boxes_to_process.append((x, y, w, h, idx))
        
        if not boxes_to_process:
            print(f"Warning: No {target_label} boxes found in {image_path.name}")
            # 仍然复制图像和JSON
            shutil.copy(image_path, output_dir / image_path.name)
            shutil.copy(json_path, output_dir / json_path.name)
            return True
        
        # 处理每个标牌框
        modified = False
        for box_idx, (x, y, w, h, shape_idx) in enumerate(boxes_to_process):
            # 扩大矩形框
            exp_x, exp_y, exp_w, exp_h = expand_box(x, y, w, h, expand_ratio, img_width, img_height)
            
            # 截取ROI
            roi = image[exp_y:exp_y+exp_h, exp_x:exp_x+exp_w]
            
            if roi.size == 0:
                continue
            
            # 通过IoU匹配找到对应的原始标注框
            original_box_in_roi = None
            current_box = (x, y, w, h)
            max_iou = 0
            best_match_box = None
            
            if original_boxes:
                # 计算当前框与所有原始框的IoU
                for orig_box in original_boxes:
                    iou = calculate_iou(current_box, orig_box)
                    if iou > max_iou:
                        max_iou = iou
                        best_match_box = orig_box
                
                # 如果IoU超过阈值，则认为匹配成功
                if max_iou >= iou_threshold and best_match_box is not None:
                    orig_x, orig_y, orig_w, orig_h = best_match_box
                    # 计算原始框在ROI中的相对坐标
                    orig_x_in_roi = int(orig_x - exp_x)
                    orig_y_in_roi = int(orig_y - exp_y)
                    orig_w_in_roi = int(orig_w)
                    orig_h_in_roi = int(orig_h)
                    original_box_in_roi = (orig_x_in_roi, orig_y_in_roi, orig_w_in_roi, orig_h_in_roi)
                    
                    if debug:
                        print(f"[DEBUG] Matched original box with IoU={max_iou:.3f} for box {box_idx+1}")
            
            # YOLO模式下的IoU验证
            if use_yolo and model is not None:
                if not original_boxes:
                    # 没有原始标注，拒绝
                    if error_log:
                        header_needed = not error_log.exists()
                        with open(error_log, 'a', encoding='utf-8') as f:
                            if header_needed:
                                f.write('image_name,image_path,json_path,mode,accepted,reason,light_mode,orig_box,yolo_box,detection_box,max_iou,thresh_offset,min_sign_brightness,expand_ratio,iou_threshold,morph_kernel_size,morph_iterations\n')
                            det_box_str = str(current_box)
                            f.write(f"{image_path.name},{image_path},{json_path},yolo,0,no_original_annotation,{light_mode},None,{det_box_str},{det_box_str},{max_iou:.4f},{thresh_offset},{min_sign_brightness},{expand_ratio},{iou_threshold},{morph_kernel_size},{morph_iterations}\n")
                    print(f"[REJECT] {image_path.name} box{box_idx+1}: No original annotation found")
                    continue
                elif max_iou < judge_iou_threshold:
                    # IoU低于判断阈值，直接拒绝
                    if error_log:
                        header_needed = not error_log.exists()
                        with open(error_log, 'a', encoding='utf-8') as f:
                            if header_needed:
                                f.write('image_name,image_path,json_path,mode,accepted,reason,light_mode,orig_box,yolo_box,detection_box,max_iou,thresh_offset,min_sign_brightness,expand_ratio,iou_threshold,morph_kernel_size,morph_iterations\n')
                            orig_box_str = str(best_match_box) if best_match_box else 'None'
                            det_box_str = str(current_box)
                            f.write(f"{image_path.name},{image_path},{json_path},yolo,0,low_iou,{light_mode},{orig_box_str},{det_box_str},{det_box_str},{max_iou:.4f},{thresh_offset},{min_sign_brightness},{expand_ratio},{iou_threshold},{morph_kernel_size},{morph_iterations}\n")
                    print(f"[REJECT] {image_path.name} box{box_idx+1}: IoU={max_iou:.3f} < {judge_iou_threshold} (judge threshold)")
                    continue
            
            # 构建调试用的图像名称
            debug_name = f"{image_path.stem}_box{box_idx+1}"
            
            # 计算当前检测框在ROI中的相对坐标
            detection_box_in_roi = (int(x - exp_x), int(y - exp_y), int(w), int(h))
            
            # 决定是否需要显示调试窗口
            # debug模式：IoU > judge_threshold 就显示
            # 非debug模式：judge_threshold < IoU < accept_threshold 才显示
            need_window = False
            if use_yolo and model is not None:
                if debug:
                    need_window = (max_iou >= judge_iou_threshold)
                else:
                    need_window = (judge_iou_threshold <= max_iou < accept_iou_threshold)
            else:
                need_window = debug  # JSON模式下保持原逻辑
            
            # 查找标牌外接矩形
            contour_result = find_sign_contour(roi, detection_box_in_roi, need_window, debug_name, original_box_in_roi,
                                              morph_kernel_size, morph_iterations, thresh_offset, min_sign_brightness,
                                              light_mode)
            if contour_result is not None:
                sign_x, sign_y, sign_w, sign_h, dbg_info, accepted = contour_result
                if debug and not accepted:
                    # 用户拒绝，记录失败日志，不修改JSON
                    if error_log:
                        header_needed = not error_log.exists()
                        with open(error_log, 'a', encoding='utf-8') as f:
                            if header_needed:
                                f.write('image_name,image_path,json_path,mode,accepted,reason,light_mode,orig_box,yolo_box,detection_box,max_iou,proposed_box,final_sign_pixel,final_threshold,thresh_offset,min_sign_brightness,expand_ratio,iou_threshold,morph_kernel_size,morph_iterations\n')
                            mode_str = 'yolo' if (use_yolo and model is not None) else 'json'
                            orig_box_str = str(best_match_box) if best_match_box else 'None'
                            det_box_str = str(current_box)
                            proposed_box_str = str((sign_x, sign_y, sign_w, sign_h))
                            f.write(f"{image_path.name},{image_path},{json_path},{mode_str},0,user_reject,{light_mode},{orig_box_str},{det_box_str},{det_box_str},{max_iou:.4f},{proposed_box_str},{dbg_info.get('final_sign_pixel')},{dbg_info.get('final_threshold')},{thresh_offset},{min_sign_brightness},{expand_ratio},{iou_threshold},{morph_kernel_size},{morph_iterations}\n")
                    print(f"[REJECT] {image_path.name} box{box_idx+1}: User rejected")
                    continue  # 不应用修改
                # accepted 或 非 debug 模式应用修改
                final_x = exp_x + sign_x
                final_y = exp_y + sign_y
                final_w = sign_w
                final_h = sign_h
                
                # 更新JSON中的坐标
                if shape_idx is not None:
                    # 使用JSON标注的情况：直接修改现有shape
                    new_points = xywh_to_points(final_x, final_y, final_w, final_h)
                    json_data["shapes"][shape_idx]["points"] = new_points
                    modified = True
                else:
                    # 使用YOLO检测的情况：需要找到匹配的原始标注并修改它
                    if best_match_box is not None:
                        # 找到与best_match_box匹配的shape索引
                        matched_shape_idx = None
                        for idx, shape in enumerate(json_data.get("shapes", [])):
                            if shape.get("label") == target_label:
                                points = shape.get("points", [])
                                if len(points) == 4:
                                    shape_x, shape_y, shape_w, shape_h = points_to_xywh(points)
                                    if (abs(shape_x - best_match_box[0]) < 1 and 
                                        abs(shape_y - best_match_box[1]) < 1 and
                                        abs(shape_w - best_match_box[2]) < 1 and
                                        abs(shape_h - best_match_box[3]) < 1):
                                        matched_shape_idx = idx
                                        break
                        
                        if matched_shape_idx is not None:
                            # 修改匹配到的原始标注
                            new_points = xywh_to_points(final_x, final_y, final_w, final_h)
                            json_data["shapes"][matched_shape_idx]["points"] = new_points
                            if "description" not in json_data["shapes"][matched_shape_idx]:
                                json_data["shapes"][matched_shape_idx]["description"] = ""
                            json_data["shapes"][matched_shape_idx]["description"] += " [Fixed by belt_detect_sign_fix]"
                            modified = True
                        else:
                            # 如果找不到匹配的shape（理论上不应该发生），添加新的
                            print(f"[WARNING] {image_path.name} box{box_idx+1}: Could not find matching shape, adding new one")
                            new_shape = {
                                "label": target_label,
                                "score": None,
                                "points": xywh_to_points(final_x, final_y, final_w, final_h),
                                "group_id": None,
                                "description": "Fixed by belt_detect_sign_fix",
                                "difficult": False,
                                "shape_type": "rectangle",
                                "flags": {},
                                "attributes": {},
                                "kie_linking": []
                            }
                            if "shapes" not in json_data:
                                json_data["shapes"] = []
                            json_data["shapes"].append(new_shape)
                            modified = True
                    else:
                        # 没有匹配的原始框（理论上不应该到这里，因为前面已经验证过）
                        print(f"[WARNING] {image_path.name} box{box_idx+1}: No original box to update")
                        continue
        
        # 保存结果
        output_image_path = output_dir / image_path.name
        output_json_path = output_dir / json_path.name
        
        shutil.copy(image_path, output_image_path)
        save_json_annotation(json_data, output_json_path)
        
        status = "Modified" if modified else "Copied"
        print(f"✓ {status}: {image_path.name}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {image_path.name}: {e}")
        return False


def process_directory(input_dir: Path, output_dir: Path, model_path: Optional[Path] = None,
                     expand_ratio: float = 0.1, thresh_offset: int = 30,
                     target_label: str = "number", use_yolo: bool = False, debug: bool = False,
                     iou_threshold: float = 0.5, morph_kernel_size: int = 5,
                     morph_iterations: int = 1, min_sign_brightness: int = 140,
                     light_mode: str = 'uniform', error_log: Optional[Path] = None,
                     accept_iou_threshold: float = 0.88, judge_iou_threshold: float = 0.7):
    """
    批量处理目录中的图像
    
    Args:
        input_dir: 输入目录（包含图像和JSON文件）
        output_dir: 输出目录
        model_path: YOLO模型路径
        expand_ratio: 矩形框扩大比例
        thresh_offset: 阈值偏移量，标牌像素值-offset作为二值化阈值（默认30）
        target_label: 要处理的标签类型
        use_yolo: 是否使用YOLO模型
        debug: 是否启用调试模式
        iou_threshold: IoU阈值，用于匹配原始标注（默认0.5）
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if debug:
        print("[DEBUG MODE ENABLED] Will show binary images and bounding boxes for each detection")
    
    # 加载YOLO模型
    model = None
    if use_yolo:
        if model_path and model_path.exists():
            print(f"Loading YOLO model from {model_path}")
            model = YOLO(str(model_path))
            model.to(device)  # 将模型移到指定设备
        else:
            print(f"Error: YOLO model not found at {model_path}")
            return
    
    # 查找所有图像文件
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_exts:
        image_files.extend(input_dir.glob(f"*{ext}"))
    
    if not image_files:
        print(f"Warning: No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    success_count = 0
    error_count = 0
    
    for image_path in image_files:
        # 查找对应的JSON文件
        json_path = input_dir / f"{image_path.stem}.json"
        
        if not json_path.exists():
            print(f"Warning: JSON file not found for {image_path.name}, skipping...")
            continue
        
        # 处理图像
        if process_image(image_path, json_path, output_dir, model,
                        expand_ratio, thresh_offset, target_label, use_yolo, device, debug,
                        iou_threshold, morph_kernel_size, morph_iterations, min_sign_brightness,
                        light_mode, error_log, accept_iou_threshold, judge_iou_threshold):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Success: {success_count}, Error: {error_count}")


def main():
    parser = argparse.ArgumentParser(
        description="传送带标牌检测修正工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用JSON标注进行修正（不使用YOLO）
  python belt_detect_sign_fix.py -i Data/belt_detect/origin_imags -o output/fixed
  
  # 使用YOLO模型进行检测和修正
  python belt_detect_sign_fix.py -i Data/images -o output/fixed --use-yolo --model-path models/best.pt
  
  # 自定义参数
  python belt_detect_sign_fix.py -i Data/images -o output/fixed -e 0.15 -t 180 --target-label number
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='输入目录路径（包含图像和JSON标注文件）'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='输出目录路径（保存修正后的图像和JSON）'
    )
    
    parser.add_argument(
        '-m', '--model-path',
        type=str,
        default=None,
        help='YOLO模型路径（仅在使用--use-yolo时需要）'
    )
    
    parser.add_argument(
        '-e', '--expand-ratio',
        type=float,
        default=0.1,
        help='矩形框扩大比例（默认: 0.1，即10%%）'
    )
    
    parser.add_argument(
        '-t', '--thresh-offset',
        type=int,
        default=30,
        help='阈值偏移量，标牌像素值-offset作为二值化阈值（默认: 30）'
    )
    
    parser.add_argument(
        '--target-label',
        type=str,
        default='number',
        help='要处理的标签类型（默认: number）'
    )
    
    parser.add_argument(
        '--use-yolo',
        action='store_true',
        help='使用YOLO模型进行检测（需要指定--model-path）'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式（显示二值化图像和检测框）'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU阈值，用于匹配原始标注框（默认: 0.5）'
    )
    
    parser.add_argument(
        '--morph-kernel-size',
        type=int,
        default=5,
        help='形态学操作核大小，用于平滑轮廓去除尖刺（默认: 5，设为0禁用）'
    )
    
    parser.add_argument(
        '--morph-iterations',
        type=int,
        default=1,
        help='形态学操作迭代次数（默认: 1）'
    )
    
    parser.add_argument(
        '--min-sign-brightness',
        type=int,
        default=140,
        help='标牌最小亮度阈值，低于此值认为是背景（默认: 140）'
    )

    parser.add_argument(
        '--light-mode',
        type=str,
        choices=['uniform', 'vertical', 'horizontal'],
        default='uniform',
        help='光照模式：uniform(均匀)/vertical(上下)/horizontal(左右)，用于分块自适应阈值'
    )

    parser.add_argument(
        '--error-log',
        type=str,
        default=None,
        help='调试拒绝时的错误日志文件路径（默认: 输出目录下error_log.csv）'
    )
    
    parser.add_argument(
        '--accept-iou-threshold',
        type=float,
        default=0.88,
        help='接受阈值，YOLO模式下IoU大于此值自动接受（默认: 0.88）'
    )
    
    parser.add_argument(
        '--judge-iou-threshold',
        type=float,
        default=0.7,
        help='判断阈值，YOLO模式下IoU小于此值直接拒绝（默认: 0.7）'
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path) if args.model_path else None
    
    # 检查输入目录
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return
    
    # 检查YOLO模型
    if args.use_yolo and not model_path:
        print("Error: --model-path is required when using --use-yolo")
        return
    
    # 处理目录
    # 计算错误日志路径
    error_log_path = Path(args.error_log) if args.error_log else (output_dir / 'error_log.csv')

    process_directory(
        input_dir,
        output_dir,
        model_path,
        args.expand_ratio,
        args.thresh_offset,
        args.target_label,
        args.use_yolo,
        args.debug,
        args.iou_threshold,
        args.morph_kernel_size,
        args.morph_iterations,
        args.min_sign_brightness,
        args.light_mode,
        error_log_path if args.debug else None,
        args.accept_iou_threshold,
        args.judge_iou_threshold
    )


if __name__ == "__main__":
    main()
