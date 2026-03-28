"""
面单数据集裁剪工具

功能：
1. 读取xanylabel格式的标注文件和对应的图片
2. 找到面单(ExpressBillSeg)的边界框
3. 随机扩展边界框（默认0.5-1.5倍）
4. 截取扩展后的区域作为新图片
5. 更新所有标注（面单、条形码、二维码）的坐标以适应新图片

用途：
解决高分辨率图片中目标占比过小导致YOLO训练效果差的问题
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import cv2
import numpy as np


def imread_unicode(img_path: Path) -> Optional[np.ndarray]:
    """
    读取支持中文路径的图片

    Args:
        img_path: 图片路径

    Returns:
        图片数组，失败返回None
    """
    try:
        with open(img_path, 'rb') as f:
            img_data = f.read()
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(img_path: Path, img: np.ndarray) -> bool:
    """
    保存支持中文路径的图片

    Args:
        img_path: 图片路径
        img: 图片数组

    Returns:
        是否保存成功
    """
    try:
        ext = img_path.suffix.lower()
        success, encoded_img = cv2.imencode(ext, img)
        if success:
            with open(img_path, 'wb') as f:
                f.write(encoded_img.tobytes())
            return True
        return False
    except Exception:
        return False


def parse_xanylabel_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    从xanylabel的4个顶点坐标解析出边界框

    Args:
        points: 4个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    return x_min, y_min, x_max, y_max


def expand_bbox(
    x_min: float, y_min: float, x_max: float, y_max: float,
    img_width: int, img_height: int,
    expand_min: float, expand_max: float
) -> Tuple[int, int, int, int]:
    """
    扩展边界框

    Args:
        x_min, y_min, x_max, y_max: 原始边界框坐标
        img_width, img_height: 图片尺寸
        expand_min: 扩展倍数最小值
        expand_max: 扩展倍数最大值

    Returns:
        (new_x_min, new_y_min, new_x_max, new_y_max) 扩展后的边界框（整数）
    """
    # 计算原始宽高
    width = x_max - x_min
    height = y_max - y_min

    # 随机生成四个方向的扩展倍数（独立随机，增加数据多样性）
    expand_ratio_left = random.uniform(expand_min / 2, expand_max / 2)
    expand_ratio_right = random.uniform(expand_min / 2, expand_max / 2)
    expand_ratio_top = random.uniform(expand_min / 2, expand_max / 2)
    expand_ratio_bottom = random.uniform(expand_min / 2, expand_max / 2)

    # 计算四个方向的扩展量
    expand_left = width * expand_ratio_left
    expand_right = width * expand_ratio_right
    expand_top = height * expand_ratio_top
    expand_bottom = height * expand_ratio_bottom

    # 计算新的边界框（四个方向独立扩展）
    new_x_min = x_min - expand_left
    new_y_min = y_min - expand_top
    new_x_max = x_max + expand_right
    new_y_max = y_max + expand_bottom

    # 确保不超过图片边界
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(img_width, new_x_max)
    new_y_max = min(img_height, new_y_max)

    # 转换为整数
    return int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)


def is_shape_center_in_crop(
    shape: Dict,
    crop_x_min: int,
    crop_y_min: int,
    crop_x_max: int,
    crop_y_max: int
) -> bool:
    """
    判断标注的中心点是否在裁剪区域内

    Args:
        shape: xanylabel格式的shape对象
        crop_x_min, crop_y_min, crop_x_max, crop_y_max: 裁剪区域坐标

    Returns:
        中心点是否在裁剪区域内
    """
    points = shape.get("points", [])
    if len(points) != 4:
        return False

    # 计算标注的中心点
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)

    # 判断中心点是否在裁剪区域内
    return (crop_x_min <= center_x <= crop_x_max and
            crop_y_min <= center_y <= crop_y_max)


def update_shape_coordinates(
    shape: Dict,
    crop_x_min: int,
    crop_y_min: int,
    crop_x_max: int,
    crop_y_max: int
) -> Optional[Dict]:
    """
    更新标注坐标以适应裁剪后的图片

    Args:
        shape: xanylabel格式的shape对象
        crop_x_min, crop_y_min: 裁剪区域的左上角坐标
        crop_x_max, crop_y_max: 裁剪区域的右下角坐标

    Returns:
        更新后的shape对象，如果标注中心不在裁剪区域内则返回None
    """
    # 检查标注中心是否在裁剪区域内
    if not is_shape_center_in_crop(shape, crop_x_min, crop_y_min, crop_x_max, crop_y_max):
        return None

    points = shape.get("points", [])
    if len(points) != 4:
        return None

    # 更新所有顶点坐标
    new_points = []
    for point in points:
        new_x = point[0] - crop_x_min
        new_y = point[1] - crop_y_min
        new_points.append([new_x, new_y])

    updated_shape = shape.copy()
    updated_shape["points"] = new_points

    return updated_shape


def process_single_crop(
    img: np.ndarray,
    shapes: List[Dict],
    annotation_data: Dict,
    target_bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    expand_min: float,
    expand_max: float,
    output_dir: Path,
    base_name: str,
    img_suffix: str
) -> Tuple[bool, List[Dict]]:
    """
    处理单个截图（面单或游离的二维码/条形码）

    Args:
        img: 原始图片
        shapes: 所有标注
        annotation_data: 原始标注数据
        target_bbox: 目标边界框
        img_width, img_height: 图片尺寸
        expand_min, expand_max: 扩展范围
        output_dir: 输出目录
        base_name: 输出文件基础名
        img_suffix: 图片扩展名

    Returns:
        (是否成功, 已包含的标注列表)
    """
    # 扩展边界框
    x_min, y_min, x_max, y_max = target_bbox
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = expand_bbox(
        x_min, y_min, x_max, y_max,
        img_width, img_height,
        expand_min, expand_max
    )

    # 裁剪图片
    cropped_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # 更新所有标注的坐标（只保留中心在裁剪区域内的）
    updated_shapes = []
    included_shapes = []
    for shape in shapes:
        updated_shape = update_shape_coordinates(
            shape, crop_x_min, crop_y_min, crop_x_max, crop_y_max
        )
        if updated_shape is not None:
            updated_shapes.append(updated_shape)
            included_shapes.append(shape)

    # 更新标注数据
    updated_annotation = annotation_data.copy()
    updated_annotation["shapes"] = updated_shapes
    updated_annotation["imageHeight"] = crop_y_max - crop_y_min
    updated_annotation["imageWidth"] = crop_x_max - crop_x_min

    # 生成输出文件名
    output_img_name = f"{base_name}{img_suffix}"
    output_json_name = f"{base_name}.json"

    # 更新JSON中的imagePath字段
    updated_annotation["imagePath"] = output_img_name

    # 保存裁剪后的图片
    output_img_path = output_dir / output_img_name
    if not imwrite_unicode(output_img_path, cropped_img):
        print(f"✗ 无法保存图片: {output_img_name}")
        return False, []

    # 保存更新后的标注文件
    output_json_path = output_dir / output_json_name
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(updated_annotation, f, indent=4, ensure_ascii=False)

    return True, included_shapes


def process_single_image(
    json_path: Path,
    img_path: Path,
    output_dir: Path,
    expand_min: float,
    expand_max: float
) -> Tuple[int, int]:
    """
    处理单张图片和对应的标注文件（处理所有面单和游离的二维码/条形码）

    Args:
        json_path: JSON标注文件路径
        img_path: 图片文件路径
        output_dir: 输出目录
        expand_min: 扩展倍数最小值
        expand_max: 扩展倍数最大值

    Returns:
        (成功数量, 失败数量)
    """
    try:
        # 读取JSON标注
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)

        # 读取图片（支持中文路径）
        img = imread_unicode(img_path)
        if img is None:
            print(f"✗ 无法读取图片: {img_path.name}")
            return 0, 1

        img_height, img_width = img.shape[:2]

        # 查找所有面单(ExpressBillSeg)的边界框
        waybill_shapes = []
        shapes = annotation_data.get("shapes", [])

        for shape in shapes:
            label = shape.get("label", "")
            if label == "ExpressBillSeg":
                points = shape.get("points", [])
                if len(points) == 4:
                    bbox = parse_xanylabel_bbox(points)
                    waybill_shapes.append((shape, bbox))

        if not waybill_shapes:
            print(f"✗ 未找到面单标注: {json_path.name}")
            return 0, 1

        # 记录所有已处理的标注
        processed_shapes = []

        # 处理每个面单
        success_count = 0
        for idx, (_, waybill_bbox) in enumerate(waybill_shapes, start=1):
            # 生成输出文件名
            if len(waybill_shapes) > 1:
                base_name = f"{img_path.stem}_w{idx}"
            else:
                base_name = img_path.stem

            # 处理单个截图
            success, included = process_single_crop(
                img, shapes, annotation_data, waybill_bbox,
                img_width, img_height, expand_min, expand_max,
                output_dir, base_name, img_path.suffix
            )

            if success:
                success_count += 1
                processed_shapes.extend(included)

        # 查找游离的二维码和条形码（中心不在任何面单截图中的）
        orphan_codes = []
        for shape in shapes:
            label = shape.get("label", "")
            if label in ["BarCode", "2DCode"]:
                # 检查是否已被处理
                if shape not in processed_shapes:
                    points = shape.get("points", [])
                    if len(points) == 4:
                        bbox = parse_xanylabel_bbox(points)
                        orphan_codes.append((shape, bbox, label))

        # 处理游离的二维码和条形码
        if orphan_codes:
            print(f"  发现 {len(orphan_codes)} 个游离的二维码/条形码")
            for idx, (_, code_bbox, label) in enumerate(orphan_codes, start=1):
                # 生成输出文件名
                label_abbr = "bc" if label == "BarCode" else "qr"
                base_name = f"{img_path.stem}_{label_abbr}{idx}"

                # 处理单个截图
                success, _ = process_single_crop(
                    img, shapes, annotation_data, code_bbox,
                    img_width, img_height, expand_min, expand_max,
                    output_dir, base_name, img_path.suffix
                )

                if success:
                    success_count += 1

        if success_count > 0:
            print(f"✓ 处理成功: {img_path.name} (生成 {success_count} 个截图)")

        return success_count, 0

    except Exception as e:
        print(f"✗ 处理失败 {img_path.name}: {str(e)}")
        return 0, 1


def process_directory(
    input_dir: Path,
    output_dir: Path,
    expand_min: float,
    expand_max: float,
    num_threads: int = 8
):
    """
    处理目录中的所有图片和标注文件（多线程）

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        expand_min: 扩展倍数最小值
        expand_max: 扩展倍数最大值
        num_threads: 线程数量
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有JSON文件
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"警告: 在 {input_dir} 中未找到JSON文件")
        return

    print(f"找到 {len(json_files)} 个JSON文件")
    print(f"扩展范围: [{expand_min}, {expand_max})")
    print(f"线程数量: {num_threads}")

    # 准备任务列表（JSON文件和对应的图片文件）
    tasks = []
    for json_file in json_files:
        # 查找对应的图片文件（支持png, jpg, jpeg）
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            potential_img = input_dir / (json_file.stem + ext)
            if potential_img.exists():
                img_path = potential_img
                break

        if img_path is None:
            print(f"✗ 未找到对应图片: {json_file.stem}")
            continue

        tasks.append((json_file, img_path))

    if not tasks:
        print("没有可处理的任务")
        return

    print(f"准备处理 {len(tasks)} 个任务\n")

    # 使用线程锁保护计数器
    lock = Lock()
    success_count = 0
    error_count = 0

    def process_task(task):
        """处理单个任务的包装函数"""
        nonlocal success_count, error_count
        json_file, img_path = task
        succ, fail = process_single_image(json_file, img_path, output_dir, expand_min, expand_max)

        with lock:
            success_count += succ
            error_count += fail

        return succ > 0

    # 使用线程池处理任务
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        futures = [executor.submit(process_task, task) for task in tasks]

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"✗ 任务执行异常: {str(e)}")
                with lock:
                    error_count += 1

    print(f"\n处理完成!")
    print(f"成功生成截图: {success_count}, 失败: {error_count}")


def main():
    parser = argparse.ArgumentParser(
        description="面单数据集裁剪工具 - 扩展并裁剪面单区域以提升YOLO训练效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数（8线程）
  python crop_waybill_dataset.py -i D:\\Data\\waybill_perception\\0104扫码相机标注交付

  # 指定输出目录和线程数
  python crop_waybill_dataset.py -i D:\\Data\\waybill_perception\\0104扫码相机标注交付 -o D:\\Data\\waybill_perception\\cropped --threads 16

  # 自定义扩展范围和线程数
  python crop_waybill_dataset.py -i ./input -o ./output --expand-min 0.3 --expand-max 2.0 --threads 4
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入文件夹路径（包含xanylabel格式的JSON文件和对应的PNG图片）'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出文件夹路径（默认：输入文件夹/intercept）'
    )

    parser.add_argument(
        '--expand-min',
        type=float,
        default=0.5,
        help='扩展范围最小值（默认：0.5）'
    )

    parser.add_argument(
        '--expand-max',
        type=float,
        default=1.5,
        help='扩展范围最大值（默认：1.5）'
    )

    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='线程数量（默认：8）'
    )

    args = parser.parse_args()

    # 转换为Path对象
    input_dir = Path(args.input)

    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    if not input_dir.is_dir():
        print(f"错误: 输入路径不是目录: {input_dir}")
        return

    # 设置输出目录
    if args.output is None:
        output_dir = input_dir / "intercept"
    else:
        output_dir = Path(args.output)

    # 验证扩展范围参数
    if args.expand_min < 0 or args.expand_max < 0:
        print(f"错误: 扩展范围必须为正数")
        return

    if args.expand_min >= args.expand_max:
        print(f"错误: expand_min 必须小于 expand_max")
        return

    # 验证线程数量参数
    if args.threads < 1:
        print(f"错误: 线程数量必须大于0")
        return

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 处理数据集
    process_directory(input_dir, output_dir, args.expand_min, args.expand_max, args.threads)


if __name__ == "__main__":
    main()

