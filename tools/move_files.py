import argparse
import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm


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


# 线程锁，用于保护文件重命名操作
file_lock = threading.Lock()


def move_single_file(source_file, output_path):
    """
    移动单个文件到目标文件夹
    
    Args:
        source_file: 源文件路径（Path对象）
        output_path: 目标文件夹路径（Path对象）
    
    Returns:
        tuple: (success: bool, message: str)
    """
    dest_file = output_path / source_file.name
    
    # 使用锁保护文件重命名检查和移动操作
    with file_lock:
        # 如果目标文件已存在，添加编号避免覆盖
        if dest_file.exists():
            base_name = dest_file.stem
            extension = dest_file.suffix
            counter = 1
            while dest_file.exists():
                dest_file = output_path / f"{base_name}_{counter}{extension}"
                counter += 1
            warning_msg = f"警告: 文件已存在，重命名为 '{dest_file.name}'"
            print(warning_msg)
        
        try:
            # 移动文件
            shutil.move(str(source_file), str(dest_file))
            return (True, f"已移动: {source_file} -> {dest_file}")
        except Exception as e:
            return (False, f"错误: 无法移动文件 '{source_file}': {e}")


def move_files(input_folder, output_folder, num_threads=4):
    """
    将input文件夹及其子文件夹中的所有图片、XML和TXT文件移动到output文件夹
    
    Args:
        input_folder: 源文件夹路径
        output_folder: 目标文件夹路径
        num_threads: 线程数（默认4）
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 检查input文件夹是否存在
    if not input_path.exists():
        print(f"错误: 输入文件夹 '{input_folder}' 不存在")
        return
    
    # 创建output文件夹（如果不存在）
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 收集所有需要移动的文件
    files_to_move = []
    print("正在扫描文件...")
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            # 检查是否为图片、XML或TXT文件
            if is_image_file(filename) or is_xml_file(filename) or is_txt_file(filename):
                source_file = Path(root) / filename
                files_to_move.append(source_file)
    
    if not files_to_move:
        print("没有找到需要移动的文件")
        return
    
    print(f"找到 {len(files_to_move)} 个文件，使用 {num_threads} 个线程进行移动...\n")
    
    # 统计信息
    moved_count = 0
    skipped_count = 0
    
    # 使用线程池移动文件
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(move_single_file, file, output_path): file 
                         for file in files_to_move}
        
        # 使用tqdm显示进度
        with tqdm(total=len(files_to_move), desc="移动文件", unit="个") as pbar:
            # 处理完成的任务
            for future in as_completed(future_to_file):
                success, message = future.result()
                if success:
                    moved_count += 1
                else:
                    skipped_count += 1
                    # 只在出错时打印详细信息
                    tqdm.write(message)
                pbar.update(1)
    
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
    
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='线程数，用于并行移动文件（默认: 4）'
    )
    
    args = parser.parse_args()
    
    print(f"输入文件夹: {args.input}")
    print(f"输出文件夹: {args.output}")
    print(f"线程数: {args.threads}")
    print("开始移动文件...\n")
    
    move_files(args.input, args.output, args.threads)


if __name__ == '__main__':
    main()
