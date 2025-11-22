"""
将X-AnyLabeling格式的标注文件转换为undefined_json格式

X-AnyLabeling格式特点：
- 包含version, flags, shapes等字段
- shapes是列表，每个shape有label, points等
- points是矩形的4个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

undefined_json格式特点：
- 是一个列表，包含多个标注对象
- 每个对象有 x, y, w, h (左上角坐标和宽高)
- type字段包含标签信息
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def convert_xanylabel_to_undefined(xanylabel_data: Dict) -> List[Dict]:
    """
    将X-AnyLabeling格式转换为undefined格式
    
    Args:
        xanylabel_data: X-AnyLabeling格式的标注数据（字典）
    
    Returns:
        undefined格式的列表
    """
    undefined_data = []
    
    shapes = xanylabel_data.get("shapes", [])
    
    for idx, shape in enumerate(shapes, start=1):
        # 获取标签
        label = shape.get("label", "unknown")
        
        # 获取矩形的4个顶点坐标
        points = shape.get("points", [])
        
        if len(points) != 4:
            print(f"Warning: Shape {idx} does not have 4 points, skipping...")
            continue
        
        # 从4个顶点计算 x, y, w, h
        # points格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # 通常是: 左上、右上、右下、左下
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x = min(x_coords)  # 左上角x
        y = min(y_coords)  # 左上角y
        w = max(x_coords) - x  # 宽度
        h = max(y_coords) - y  # 高度
        
        # 构建undefined格式的对象
        undefined_item = {
            "alias": str(idx),
            "cid": idx,
            "id": f"r-{idx}",
            "region": "rect",
            "children": [],
            "text": {
                "text": ""
            },
            "type": {
                "1": label
            },
            "pId": -1,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "imgAttr": {},
            "color": "",
            "propIdInfo": {
                "referId": 1,
                "key": "",
                "value": ""
            },
            "isNewResult": True,
            "markRotationAngle": "0",
            "labelType": f"1:{label}"
        }
        
        undefined_data.append(undefined_item)
    
    return undefined_data


def process_directory(input_dir: Path, output_dir: Path):
    """
    处理目录中的所有JSON文件
    
    Args:
        input_dir: 输入目录路径（包含X-AnyLabeling格式的JSON文件）
        output_dir: 输出目录路径
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有JSON文件
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    success_count = 0
    error_count = 0
    
    for json_file in json_files:
        try:
            # 读取X-AnyLabeling格式的JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                xanylabel_data = json.load(f)
            
            # 确保是字典格式
            if not isinstance(xanylabel_data, dict):
                print(f"Error: {json_file.name} is not a dict format, skipping...")
                error_count += 1
                continue
            
            # 转换格式
            undefined_data = convert_xanylabel_to_undefined(xanylabel_data)
            
            # 写入输出文件（如果已存在则覆盖）
            output_file = output_dir / json_file.name
            
            # 检查文件是否已存在
            file_status = "Overwritten" if output_file.exists() else "Created"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(undefined_data, f, indent=4, ensure_ascii=False)
            
            print(f"✓ {file_status}: {json_file.name} -> {output_file.name} ({len(undefined_data)} annotations)")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {str(e)}")
            error_count += 1
    
    print(f"\nConversion complete!")
    print(f"Success: {success_count}, Error: {error_count}")


def main():
    parser = argparse.ArgumentParser(
        description="将X-AnyLabeling格式的标注文件转换为undefined_json格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python xanylabel_to_undefined_json.py -i ./xanylabel_jsons -o ./undefined_jsons
  python xanylabel_to_undefined_json.py -i Data/belt_detect/origin_imags -o Data/belt_detect/json_converted
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='输入目录路径（包含X-AnyLabeling格式的JSON文件）'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='输出目录路径（转换后的undefined格式JSON文件）'
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return
    
    # 处理转换
    process_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()
