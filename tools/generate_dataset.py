# 本脚本的功能是输入图片和yolo标注文件所在文件夹路径，按照指定的比例划分训练集、验证集和测试集，输出对应的txt文件
# 输入由argparse模块处理
# 输出文件夹格式:
# dataset/
#   ├── images/
#   │   ├── train
#   │   ├── val
#   │   └── test
#   └── labels/
#       ├── train
#       ├── val
#       └── test

import os
import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict


def get_image_label_pairs(input_folder):
    """
    获取输入文件夹中所有的图片和标注文件对
    返回: [(image_path, label_path), ...]
    """
    input_path = Path(input_folder)
    
    # 支持的图片格式
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    
    # 查找所有图片文件
    image_files = []
    for ext in img_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
    
    # 构建图片-标注对
    pairs = []
    missing_labels = []
    
    for img_file in image_files:
        # 查找对应的txt标注文件
        label_file = img_file.with_suffix('.txt')
        
        if label_file.exists():
            pairs.append((img_file, label_file))
        else:
            missing_labels.append(img_file.name)
    
    if missing_labels:
        print(f"警告: {len(missing_labels)} 个图片文件没有对应的标注文件")
        if len(missing_labels) <= 10:
            for name in missing_labels:
                print(f"  - {name}")
        else:
            for name in missing_labels[:10]:
                print(f"  - {name}")
            print(f"  ... 还有 {len(missing_labels) - 10} 个")
    
    return pairs


def split_dataset(pairs, train_ratio, val_ratio, test_ratio, seed=42):
    """
    按照指定比例划分数据集
    返回: (train_pairs, val_pairs, test_pairs)
    """
    # 设置随机种子以保证可复现
    random.seed(seed)
    
    # 随机打乱
    pairs_copy = pairs.copy()
    random.shuffle(pairs_copy)
    
    total = len(pairs_copy)
    
    # 按比例计算每个集合的样本数
    # 注意：剩余样本优先分配给训练集
    if test_ratio == 0:
        # 如果test比例为0，剩余样本分配给训练集
        val_count = int(total * val_ratio)
        test_count = 0
        train_count = total - val_count
    elif val_ratio == 0:
        # 如果val比例为0，剩余样本分配给训练集
        test_count = int(total * test_ratio)
        val_count = 0
        train_count = total - test_count
    else:
        # 正常情况：剩余样本分配给训练集
        val_count = int(total * val_ratio)
        test_count = int(total * test_ratio)
        train_count = total - val_count - test_count
    
    train_pairs = pairs_copy[:train_count]
    val_pairs = pairs_copy[train_count:train_count + val_count]
    test_pairs = pairs_copy[train_count + val_count:]
    
    return train_pairs, val_pairs, test_pairs


def create_dataset_structure(output_folder):
    """
    创建数据集文件夹结构
    """
    output_path = Path(output_folder)
    
    # 创建目录结构
    dirs = [
        output_path / 'images' / 'train',
        output_path / 'images' / 'val',
        output_path / 'images' / 'test',
        output_path / 'labels' / 'train',
        output_path / 'labels' / 'val',
        output_path / 'labels' / 'test',
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return output_path


def copy_files(pairs, output_folder, split_name):
    """
    复制文件到对应的数据集文件夹
    """
    output_path = Path(output_folder)
    img_dir = output_path / 'images' / split_name
    label_dir = output_path / 'labels' / split_name
    
    for img_file, label_file in pairs:
        # 复制图片
        shutil.copy2(img_file, img_dir / img_file.name)
        # 复制标注
        shutil.copy2(label_file, label_dir / label_file.name)


def generate_file_list(pairs, output_folder, split_name):
    """
    生成包含文件路径的txt列表（可选功能）
    """
    output_path = Path(output_folder)
    list_file = output_path / f'{split_name}.txt'
    
    with open(list_file, 'w', encoding='utf-8') as f:
        for img_file, _ in pairs:
            # 写入相对路径
            rel_path = f'./images/{split_name}/{img_file.name}'
            f.write(f'{rel_path}\n')
    
    return list_file


def analyze_dataset(pairs):
    """
    分析数据集的类别分布
    """
    class_counts = defaultdict(int)
    total_objects = 0
    
    for _, label_file in pairs:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
                    total_objects += 1
    
    return class_counts, total_objects


def main():
    parser = argparse.ArgumentParser(
        description='将YOLO格式数据集按比例划分为训练集、验证集和测试集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python %(prog)s -i ./data/annotations -o ./dataset
  python %(prog)s -i ./data/annotations -o ./dataset --train 0.7 --val 0.2 --test 0.1
  python %(prog)s -i ./data/annotations -o ./dataset --seed 123
  
输出结构:
  dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── train.txt
    ├── val.txt
    └── test.txt
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入文件夹路径，包含图片文件和对应的YOLO格式txt标注文件'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出数据集文件夹路径'
    )
    
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='训练集比例 (默认: 0.7)'
    )
    
    parser.add_argument(
        '--val',
        type=float,
        default=0.2,
        help='验证集比例 (默认: 0.2)'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=0.1,
        help='测试集比例 (默认: 0.1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子，用于数据集划分的可复现性 (默认: 42)'
    )
    
    parser.add_argument(
        '--no-copy',
        action='store_true',
        help='不复制文件，仅生成文件列表txt'
    )
    
    args = parser.parse_args()
    
    # 验证比例总和
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.01:
        print(f"错误: 训练集、验证集和测试集比例之和必须为1.0，当前为 {total_ratio}")
        return
    
    # 检查输入文件夹
    if not os.path.exists(args.input):
        print(f"错误: 输入文件夹不存在: {args.input}")
        return
    
    print("=" * 60)
    print("YOLO数据集划分工具")
    print("=" * 60)
    print(f"输入文件夹: {args.input}")
    print(f"输出文件夹: {args.output}")
    print(f"划分比例: 训练集={args.train:.1%}, 验证集={args.val:.1%}, 测试集={args.test:.1%}")
    print(f"随机种子: {args.seed}")
    print("-" * 60)
    
    # 获取图片-标注对
    print("正在扫描输入文件夹...")
    pairs = get_image_label_pairs(args.input)
    
    if not pairs:
        print("错误: 没有找到有效的图片-标注文件对")
        return
    
    print(f"找到 {len(pairs)} 对图片和标注文件")
    
    # 划分数据集
    print("\n正在划分数据集...")
    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs, args.train, args.val, args.test, args.seed
    )
    
    print(f"  训练集: {len(train_pairs)} 个样本")
    print(f"  验证集: {len(val_pairs)} 个样本")
    print(f"  测试集: {len(test_pairs)} 个样本")
    
    # 分析数据集分布
    print("\n数据集统计信息:")
    for split_name, split_pairs in [('训练集', train_pairs), ('验证集', val_pairs), ('测试集', test_pairs)]:
        if split_pairs:
            class_counts, total_objects = analyze_dataset(split_pairs)
            print(f"\n  {split_name}:")
            print(f"    样本数: {len(split_pairs)}")
            print(f"    目标数: {total_objects}")
            print(f"    类别分布:")
            for class_id in sorted(class_counts.keys()):
                count = class_counts[class_id]
                percentage = count / total_objects * 100
                print(f"      类别 {class_id}: {count} ({percentage:.1f}%)")
    
    # 创建输出文件夹结构
    print(f"\n正在创建输出文件夹结构: {args.output}")
    output_path = create_dataset_structure(args.output)
    
    # 复制文件
    if not args.no_copy:
        print("\n正在复制文件...")
        if train_pairs:
            print(f"  复制训练集文件...")
            copy_files(train_pairs, args.output, 'train')
        if val_pairs:
            print(f"  复制验证集文件...")
            copy_files(val_pairs, args.output, 'val')
        if test_pairs:
            print(f"  复制测试集文件...")
            copy_files(test_pairs, args.output, 'test')
    
    # 生成文件列表
    print("\n正在生成文件列表...")
    if train_pairs:
        train_list = generate_file_list(train_pairs, args.output, 'train')
        print(f"  生成: {train_list}")
    if val_pairs:
        val_list = generate_file_list(val_pairs, args.output, 'val')
        print(f"  生成: {val_list}")
    if test_pairs:
        test_list = generate_file_list(test_pairs, args.output, 'test')
        print(f"  生成: {test_list}")
    
    print("\n" + "=" * 60)
    print("✓ 数据集划分完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
