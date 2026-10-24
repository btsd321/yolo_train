# 本脚本的功能是输入图片和标注文件所在文件夹路径，按照指定的比例划分训练集、验证集和测试集
# 支持两种任务模式：
#   1. detect（默认）：输入图片+YOLO格式txt标注文件，输出 images/ + labels/ 结构
#   2. classify：输入按类别子目录组织的图片文件夹，输出 {split}/{class_name}/ 结构
#
# detect 模式输出文件夹格式:
# dataset/
#   ├── images/
#   │   ├── train
#   │   ├── val
#   │   └── test
#   └── labels/
#       ├── train
#       ├── val
#       └── test
#   ├── train.txt
#   ├── val.txt
#   └── test.txt
#
# classify 模式输出文件夹格式（YOLO 图像分类标准格式）:
# dataset/
#   ├── train/
#   │   ├── class_a/
#   │   │   ├── img1.jpg
#   │   │   └── img2.jpg
#   │   └── class_b/
#   │       └── img3.jpg
#   ├── val/
#   │   ├── class_a/
#   │   └── class_b/
#   └── test/
#       ├── class_a/
#       └── class_b/

import os
import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict


# 支持的图片格式
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP', '.webp', '.WEBP'}


# ============================================================================
# 检测模式函数 (detect)
# ============================================================================

def get_image_label_pairs(input_folder):
    """
    获取输入文件夹中所有的图片和标注文件对
    返回: [(image_path, label_path), ...]
    """
    input_path = Path(input_folder)
    
    # 查找所有图片文件（大小写不敏感去重）
    image_files = set()
    for ext in IMG_EXTENSIONS:
        image_files.update(input_path.glob(f'*{ext}'))
    image_files = sorted(image_files)
    
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
    创建检测数据集文件夹结构
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
    分析检测数据集的类别分布
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


# ============================================================================
# 分类模式函数 (classify)
# ============================================================================

def get_classify_images(input_folder):
    """
    扫描输入文件夹中的类别子目录，收集每个类别的图片文件
    输入文件夹结构应为:
      input_folder/
        ├── class_a/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── class_b/
            └── img3.jpg
    
    返回: {class_name: [image_path, ...], ...}
    """
    input_path = Path(input_folder)
    class_images = {}
    
    # 遍历输入文件夹的子目录，每个子目录是一个类别
    for item in sorted(input_path.iterdir()):
        if not item.is_dir():
            continue
        
        class_name = item.name
        # 查找该类别目录下的所有图片（大小写不敏感去重）
        images = set()
        for ext in IMG_EXTENSIONS:
            images.update(item.glob(f'*{ext}'))
        
        if images:
            class_images[class_name] = sorted(images)
            print(f"  类别 '{class_name}': {len(images)} 张图片")
        else:
            print(f"  警告: 类别目录 '{class_name}' 中没有图片文件")
    
    return class_images


def split_classify_dataset(class_images, train_ratio, val_ratio, test_ratio, seed=42):
    """
    对分类数据集按类别进行分层抽样划分
    每个类别独立按比例划分，保持各类别在 train/val/test 中的比例一致
    
    返回: {split_name: {class_name: [image_path, ...], ...}, ...}
    """
    random.seed(seed)
    
    splits = {'train': {}, 'val': {}, 'test': {}}
    
    for class_name, images in class_images.items():
        images_copy = images.copy()
        random.shuffle(images_copy)
        
        total = len(images_copy)
        
        # 按比例计算每个集合的样本数
        if test_ratio == 0:
            val_count = int(total * val_ratio)
            test_count = 0
            train_count = total - val_count
        elif val_ratio == 0:
            test_count = int(total * test_ratio)
            val_count = 0
            train_count = total - test_count
        else:
            val_count = int(total * val_ratio)
            test_count = int(total * test_ratio)
            train_count = total - val_count - test_count
        
        splits['train'][class_name] = images_copy[:train_count]
        splits['val'][class_name] = images_copy[train_count:train_count + val_count]
        splits['test'][class_name] = images_copy[train_count + val_count:]
    
    return splits


def create_classify_structure(output_folder, class_names):
    """
    创建分类数据集文件夹结构
    结构: output/{split}/{class_name}/
    """
    output_path = Path(output_folder)
    
    dirs = []
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            dirs.append(output_path / split / class_name)
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return output_path


def copy_classify_files(splits, output_folder):
    """
    复制图片到分类数据集对应的 {split}/{class_name}/ 目录
    """
    output_path = Path(output_folder)
    
    for split_name, class_dict in splits.items():
        for class_name, images in class_dict.items():
            dest_dir = output_path / split_name / class_name
            for img_file in images:
                shutil.copy2(img_file, dest_dir / img_file.name)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='将数据集按比例划分为训练集、验证集和测试集，支持检测(detect)和分类(classify)两种模式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 检测模式（默认）
  python %(prog)s -i ./data/annotations -o ./dataset
  python %(prog)s -i ./data/annotations -o ./dataset --train 0.7 --val 0.2 --test 0.1

  # 分类模式
  python %(prog)s -i ./data/raw -o ./dataset --task classify --train 0.8 --val 0.2
  python %(prog)s -i ./data/raw -o ./dataset --task classify --train 0.7 --val 0.15 --test 0.15 --seed 123

检测模式输出结构:
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

分类模式输出结构 (YOLO 图像分类标准格式):
  dataset/
    ├── train/
    │   ├── class_a/
    │   └── class_b/
    ├── val/
    │   ├── class_a/
    │   └── class_b/
    └── test/
        ├── class_a/
        └── class_b/
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入文件夹路径（detect模式: 图片+YOLO标注; classify模式: 按类别子目录组织的图片）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=False,
        default=None,
        help='输出数据集文件夹路径（默认: 在输入文件夹内创建dataset文件夹）'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['detect', 'classify'],
        default='detect',
        help='任务类型: detect=目标检测(图片+标注), classify=图像分类(按类别子目录) (默认: detect)'
    )
    
    parser.add_argument(
        '--train',
        type=float,
        default=0.8,
        help='训练集比例 (默认: 0.8)'
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
        default=0.0,
        help='测试集比例 (默认: 0.0)'
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
        help='不复制文件，仅生成文件列表txt（仅detect模式有效）'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，默认在输入文件夹内创建dataset文件夹
    if args.output is None:
        args.output = os.path.join(args.input, 'dataset')
    
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
    print(f"YOLO数据集划分工具 ({args.task} 模式)")
    print("=" * 60)
    print(f"输入文件夹: {args.input}")
    print(f"输出文件夹: {args.output}")
    print(f"划分比例: 训练集={args.train:.1%}, 验证集={args.val:.1%}, 测试集={args.test:.1%}")
    print(f"随机种子: {args.seed}")
    print("-" * 60)
    
    if args.task == 'classify':
        run_classify(args)
    else:
        run_detect(args)


def run_detect(args):
    """检测模式主流程"""
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
    create_dataset_structure(args.output)
    
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
    print("[OK] 数据集划分完成!")
    print("=" * 60)


def run_classify(args):
    """分类模式主流程"""
    # 扫描类别子目录
    print("正在扫描类别子目录...")
    class_images = get_classify_images(args.input)
    
    if not class_images:
        print("错误: 输入文件夹中没有找到包含图片的类别子目录")
        print("提示: classify 模式要求输入文件夹下每个子目录代表一个类别，如:")
        print("  input/")
        print("    ├── class_a/")
        print("    │   ├── img1.jpg")
        print("    │   └── img2.jpg")
        print("    └── class_b/")
        print("        └── img3.jpg")
        return
    
    total_images = sum(len(imgs) for imgs in class_images.values())
    print(f"\n共 {len(class_images)} 个类别, {total_images} 张图片")
    
    # 划分数据集（分层抽样）
    print("\n正在划分数据集...")
    splits = split_classify_dataset(class_images, args.train, args.val, args.test, args.seed)
    
    # 打印划分统计
    print(f"\n划分结果:")
    for split_name in ['train', 'val', 'test']:
        split_total = sum(len(imgs) for imgs in splits[split_name].values())
        print(f"  {split_name}: {split_total} 张图片")
        for class_name in sorted(splits[split_name].keys()):
            count = len(splits[split_name][class_name])
            print(f"    {class_name}: {count}")
    
    # 创建输出文件夹结构
    print(f"\n正在创建输出文件夹结构: {args.output}")
    class_names = sorted(class_images.keys())
    create_classify_structure(args.output, class_names)
    
    # 复制文件
    print("\n正在复制文件...")
    copy_classify_files(splits, args.output)
    print("  复制完成")
    
    print("\n" + "=" * 60)
    print("[OK] 分类数据集划分完成!")
    print(f"  输出目录: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
