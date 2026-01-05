import argparse
import os
import shutil
from pathlib import Path


def is_image_file(filename):
    """检查文件是否为图片格式"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp', '.ico'}
    return Path(filename).suffix.lower() in image_extensions


def is_xml_file(filename):
    """检查文件是否为XML格式"""
    return Path(filename).suffix.lower() == '.xml'


def is_txt_file(filename):
    """检查文件是否为TXT格式"""
    return Path(filename).suffix.lower() == '.txt'


def move_files(input_folder, output_folder):
    """
    将input文件夹及其子文件夹中的所有图片和XML文件移动到output文件夹
    
    Args:
        input_folder: 源文件夹路径
        output_folder: 目标文件夹路径
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 检查input文件夹是否存在
    if not input_path.exists():
        print(f"错误: 输入文件夹 '{input_folder}' 不存在")
        return
    
    # 创建output文件夹（如果不存在）
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    moved_count = 0
    skipped_count = 0
    
    # 递归遍历input文件夹及其子文件夹
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            # 检查是否为图片、XML或TXT文件
            if is_image_file(filename) or is_xml_file(filename) or is_txt_file(filename):
                source_file = Path(root) / filename
                dest_file = output_path / filename
                
                # 如果目标文件已存在，添加编号避免覆盖
                if dest_file.exists():
                    base_name = dest_file.stem
                    extension = dest_file.suffix
                    counter = 1
                    while dest_file.exists():
                        dest_file = output_path / f"{base_name}_{counter}{extension}"
                        counter += 1
                    print(f"警告: 文件已存在，重命名为 '{dest_file.name}'")
                
                try:
                    # 移动文件
                    shutil.move(str(source_file), str(dest_file))
                    print(f"已移动: {source_file} -> {dest_file}")
                    moved_count += 1
                except Exception as e:
                    print(f"错误: 无法移动文件 '{source_file}': {e}")
                    skipped_count += 1
    
    # 输出统计信息
    print(f"\n完成! 共移动 {moved_count} 个文件")
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 个文件（出现错误）")


def main():
    parser = argparse.ArgumentParser(
        description='将input文件夹及其子文件夹中的所有图片、XML和TXT文件剪切到output文件夹'
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入文件夹路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出文件夹路径'
    )
    
    args = parser.parse_args()
    
    print(f"输入文件夹: {args.input}")
    print(f"输出文件夹: {args.output}")
    print("开始移动文件...\n")
    
    move_files(args.input, args.output)


if __name__ == '__main__':
    main()
