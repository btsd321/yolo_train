"""
将undefined_json格式的标注文件转换为X-AnyLabeling格式

undefined_json格式特点：
- 是一个列表，包含多个标注对象
- 每个对象有 x, y, w, h (左上角坐标和宽高)
- type字段包含标签信息

X-AnyLabeling格式特点：
- 包含version, flags, shapes等字段
- shapes是列表，每个shape有label, points等
- points是矩形的4个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image


def get_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    """
    获取图像的宽高
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        (width, height) 或 None（如果无法读取）
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # 返回 (width, height)
    except Exception as e:
        print(f"Warning: Cannot read image {image_path}: {e}")
        return None


def find_image_file(json_file: Path, image_dir: Path, possible_exts: List[str]) -> Optional[Path]:
    """
    查找与JSON文件同名的图像文件
    
    Args:
        json_file: JSON文件路径
        image_dir: 图像文件所在目录
        possible_exts: 可能的图像扩展名列表
    
    Returns:
        图像文件路径或None
    """
    stem = json_file.stem
    
    for ext in possible_exts:
        image_path = image_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path
    
    return None


def convert_undefined_to_xanylabel(undefined_data: List[Dict], image_name: str, 
                                   image_width: int = 1920, image_height: int = 1080) -> Dict:
    """
    将undefined格式转换为X-AnyLabeling格式
    
    Args:
        undefined_data: undefined格式的标注数据（列表）
        image_name: 图像文件名
        image_width: 图像宽度（默认1920）
        image_height: 图像高度（默认1080）
    
    Returns:
        X-AnyLabeling格式的字典
    """
    shapes = []
    
    for item in undefined_data:
        # 提取标签名称
        # type字段可能是 {"1": "number"} 或 {"1": "parcel"} 等
        label = None
        if "type" in item and isinstance(item["type"], dict):
            # 取第一个值作为标签
            label = list(item["type"].values())[0]
        elif "labelType" in item:
            # labelType格式如 "1:number"，取冒号后的部分
            label = item["labelType"].split(":")[-1]
        
        if not label:
            print(f"Warning: No label found for item {item.get('id', 'unknown')}, skipping...")
            continue
        
        # 从 x, y, w, h 转换为4个顶点坐标
        x = item.get("x", 0)
        y = item.get("y", 0)
        w = item.get("w", 0)
        h = item.get("h", 0)
        
        # 矩形的4个顶点：左上、右上、右下、左下
        points = [
            [x, y],           # 左上
            [x + w, y],       # 右上
            [x + w, y + h],   # 右下
            [x, y + h]        # 左下
        ]
        
        shape = {
            "label": label,
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
        
        shapes.append(shape)
    
    # 构建X-AnyLabeling格式
    xanylabel_data = {
        "version": "3.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    
    return xanylabel_data


def process_directory(input_dir: Path, output_dir: Path, 
                      image_dir: Optional[Path] = None,
                      image_width: int = 1920, image_height: int = 1080,
                      image_ext: str = ".jpg"):
    """
    处理目录中的所有JSON文件
    
    Args:
        input_dir: 输入目录路径（包含undefined格式的JSON文件）
        output_dir: 输出目录路径
        image_dir: 图像文件所在目录（如果提供，将自动检测图像尺寸）
        image_width: 图像宽度（默认值，当无法检测时使用）
        image_height: 图像高度（默认值，当无法检测时使用）
        image_ext: 图像文件扩展名
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有JSON文件
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    # 可能的图像扩展名
    possible_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    success_count = 0
    error_count = 0
    
    for json_file in json_files:
        try:
            # 读取undefined格式的JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                undefined_data = json.load(f)
            
            # 确保是列表格式
            if not isinstance(undefined_data, list):
                print(f"Error: {json_file.name} is not a list format, skipping...")
                error_count += 1
                continue
            
            # 生成对应的图像文件名（默认）
            image_name = json_file.stem + image_ext
            
            # 尝试获取实际图像尺寸
            actual_width = image_width
            actual_height = image_height
            
            if image_dir and image_dir.exists():
                # 查找同名图像文件
                image_file = find_image_file(json_file, image_dir, possible_exts)
                
                if image_file:
                    # 获取图像尺寸
                    size = get_image_size(image_file)
                    if size:
                        actual_width, actual_height = size
                        image_name = image_file.name
                        print(f"  Found image: {image_file.name} ({actual_width}x{actual_height})")
                    else:
                        print(f"  Warning: Cannot read image size, using default ({actual_width}x{actual_height})")
                else:
                    print(f"  Warning: No matching image found for {json_file.name}, using default size")
            
            # 转换格式
            xanylabel_data = convert_undefined_to_xanylabel(
                undefined_data, 
                image_name,
                actual_width,
                actual_height
            )
            
            # 写入输出文件（如果已存在则覆盖）
            output_file = output_dir / json_file.name
            
            # 检查文件是否已存在
            file_status = "Overwritten" if output_file.exists() else "Created"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(xanylabel_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ {file_status}: {json_file.name} -> {output_file.name}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {str(e)}")
            error_count += 1
    
    print(f"\nConversion complete!")
    print(f"Success: {success_count}, Error: {error_count}")



def main():
    parser = argparse.ArgumentParser(
        description="将undefined_json格式的标注文件转换为X-AnyLabeling格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python undefined_json_to_xanylabel.py -i ./input_jsons -o ./output_jsons
  python undefined_json_to_xanylabel.py -i ./input_jsons -o ./output_jsons -d ./images
  python undefined_json_to_xanylabel.py -i ./input_jsons -o ./output_jsons -d ./images -w 1920 -ht 1080
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='输入目录路径（包含undefined格式的JSON文件）'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='输出目录路径（转换后的X-AnyLabeling格式JSON文件）'
    )
    
    parser.add_argument(
        '-d', '--image-dir',
        type=str,
        default=None,
        help='图像文件所在目录（如果提供，将自动检测图像尺寸）'
    )
    
    parser.add_argument(
        '-w', '--image-width',
        type=int,
        default=1920,
        help='图像宽度（默认: 1920，当无法检测图像时使用）'
    )
    
    parser.add_argument(
        '-ht', '--image-height',
        type=int,
        default=1080,
        help='图像高度（默认: 1080，当无法检测图像时使用）'
    )
    
    parser.add_argument(
        '-e', '--image-ext',
        type=str,
        default='.jpg',
        help='图像文件扩展名（默认: .jpg，当无法检测图像时使用）'
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return
    
    # 检查图像目录
    if image_dir and not image_dir.exists():
        print(f"Warning: Image directory does not exist: {image_dir}")
        print(f"Will use default image size: {args.image_width}x{args.image_height}")
        image_dir = None
    
    # 处理转换
    process_directory(
        input_dir,
        output_dir,
        image_dir,
        args.image_width,
        args.image_height,
        args.image_ext
    )



if __name__ == "__main__":
    main()
