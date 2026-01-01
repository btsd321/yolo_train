# YOLO12 模型训练脚本
# 本脚本用于训练 YOLO12 目标检测模型
# 支持单GPU和多GPU训练，支持断点续训
# 使用 argparse 管理命令行参数

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLO12模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从头开始训练
  python %(prog)s --data ./dataset/coco.yaml --model yolo12n.pt --epochs 100
  
  # 使用预训练权重训练
  python %(prog)s --data ./dataset/coco.yaml --model yolo12m.pt --epochs 200 --batch 16
  
  # 断点续训
  python %(prog)s --resume ./runs/detect/train/weights/last.pt
  
  # 多GPU训练
  python %(prog)s --data ./dataset/coco.yaml --model yolo12l.pt --device 0,1
        """
    )
    
    # 数据集和模型参数
    parser.add_argument(
        '--data',
        type=str,
        default='',
        help='数据集配置文件路径 (YAML格式，包含train/val/test路径和类别信息)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolo12n.pt',
        help='模型配置或预训练权重路径 (默认: yolo12n.pt)'
    )
    
    # 训练参数
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='训练轮数 (默认: 100)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='批次大小 (默认: 16)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='输入图像尺寸 (默认: 640)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='训练设备，如 0 或 0,1,2,3 或 cpu (默认: 自动选择)'
    )
    
    # 优化器和学习率参数
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
        help='优化器类型 (默认: auto)'
    )
    
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='初始学习率 (默认: 0.01)'
    )
    
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='最终学习率因子 (默认: 0.01)'
    )
    
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.937,
        help='SGD momentum/Adam beta1 (默认: 0.937)'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='权重衰减系数 (默认: 0.0005)'
    )
    
    # 数据增强参数
    parser.add_argument(
        '--hsv-h',
        type=float,
        default=0.015,
        help='HSV色调增强范围 (默认: 0.015)'
    )
    
    parser.add_argument(
        '--hsv-s',
        type=float,
        default=0.7,
        help='HSV饱和度增强范围 (默认: 0.7)'
    )
    
    parser.add_argument(
        '--hsv-v',
        type=float,
        default=0.4,
        help='HSV明度增强范围 (默认: 0.4)'
    )
    
    parser.add_argument(
        '--degrees',
        type=float,
        default=0.0,
        help='旋转角度范围 (默认: 0.0)'
    )
    
    parser.add_argument(
        '--translate',
        type=float,
        default=0.1,
        help='平移范围 (默认: 0.1)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='缩放范围 (默认: 0.5)'
    )
    
    parser.add_argument(
        '--shear',
        type=float,
        default=0.0,
        help='剪切角度范围 (默认: 0.0)'
    )
    
    parser.add_argument(
        '--perspective',
        type=float,
        default=0.0,
        help='透视变换范围 (默认: 0.0)'
    )
    
    parser.add_argument(
        '--flipud',
        type=float,
        default=0.0,
        help='上下翻转概率 (默认: 0.0)'
    )
    
    parser.add_argument(
        '--fliplr',
        type=float,
        default=0.5,
        help='左右翻转概率 (默认: 0.5)'
    )
    
    parser.add_argument(
        '--mosaic',
        type=float,
        default=1.0,
        help='Mosaic增强概率 (默认: 1.0)'
    )
    
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.0,
        help='MixUp增强概率 (默认: 0.0)'
    )
    
    parser.add_argument(
        '--copy-paste',
        type=float,
        default=0.0,
        help='Copy-Paste增强概率 (默认: 0.0)'
    )
    
    # 保存和验证参数
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='项目保存路径 (默认: runs/detect)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='train',
        help='实验名称 (默认: train)'
    )
    
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='是否覆盖已存在的实验目录'
    )
    
    parser.add_argument(
        '--save-period',
        type=int,
        default=-1,
        help='每隔N个epoch保存一次模型，-1表示只保存最后一次 (默认: -1)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='早停耐心值，多少个epoch无改善后停止 (默认: 50)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='数据加载线程数 (默认: 8)'
    )
    
    # 其他参数
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='断点续训权重路径 (如果指定则忽略其他参数)'
    )
    
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='是否使用预训练权重'
    )
    
    parser.add_argument(
        '--amp',
        action='store_true',
        default=True,
        help='使用自动混合精度训练 (默认: True)'
    )
    
    parser.add_argument(
        '--cache',
        type=str,
        default='',
        choices=['', 'ram', 'disk'],
        help='是否缓存图像到内存/磁盘以加速训练 (默认: 不缓存)'
    )
    
    parser.add_argument(
        '--close-mosaic',
        type=int,
        default=10,
        help='在最后N个epoch关闭Mosaic增强 (默认: 10)'
    )
    
    parser.add_argument(
        '--val',
        action='store_true',
        default=True,
        help='训练时是否进行验证 (默认: True)'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        default=True,
        help='是否保存训练图表 (默认: True)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='是否显示详细信息 (默认: True)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='随机种子 (默认: 0)'
    )
    
    parser.add_argument(
        '--rect',
        action='store_true',
        help='使用矩形训练（减少填充）'
    )
    
    parser.add_argument(
        '--cos-lr',
        action='store_true',
        help='使用余弦学习率调度器'
    )
    
    parser.add_argument(
        '--fraction',
        type=float,
        default=1.0,
        help='使用数据集的比例 (默认: 1.0，即全部数据)'
    )
    
    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()
    
    print("=" * 80)
    print("YOLO12 模型训练")
    print("=" * 80)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"✓ CUDA可用，GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ CUDA不可用，将使用CPU训练")
    
    # 断点续训
    if args.resume:
        print(f"\n从检查点恢复训练: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
        print("\n" + "=" * 80)
        print("✓ 训练完成!")
        print("=" * 80)
        return results
    
    # 检查数据集配置文件
    if not args.data:
        print("错误: 必须指定数据集配置文件 (--data)")
        return
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 数据集配置文件不存在: {args.data}")
        return
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    print(f"预训练: {args.pretrained}")
    model = YOLO(args.model)
    
    # 训练配置
    print("\n训练配置:")
    print(f"  数据集: {args.data}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch}")
    print(f"  图像尺寸: {args.imgsz}")
    print(f"  设备: {args.device if args.device else '自动选择'}")
    print(f"  优化器: {args.optimizer}")
    print(f"  初始学习率: {args.lr0}")
    print(f"  最终学习率因子: {args.lrf}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  工作线程: {args.workers}")
    print(f"  保存路径: {args.project}/{args.name}")
    print(f"  混合精度: {args.amp}")
    print(f"  图像缓存: {args.cache if args.cache else '不缓存'}")
    print(f"  数据增强:")
    print(f"    - Mosaic: {args.mosaic}")
    print(f"    - MixUp: {args.mixup}")
    print(f"    - Copy-Paste: {args.copy_paste}")
    print(f"    - 左右翻转: {args.fliplr}")
    print(f"    - 上下翻转: {args.flipud}")
    print(f"    - HSV (H/S/V): {args.hsv_h}/{args.hsv_s}/{args.hsv_v}")
    print(f"    - 平移: {args.translate}")
    print(f"    - 缩放: {args.scale}")
    print("-" * 80)
    
    # 开始训练
    print("\n开始训练...\n")
    try:
        results = model.train(
            # 数据和模型
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            
            # 设备和性能
            device=args.device,
            workers=args.workers,
            amp=args.amp,
            cache=args.cache if args.cache else False,
            
            # 优化器参数
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            cos_lr=args.cos_lr,
            
            # 数据增强
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            
            # 保存和验证
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            save_period=args.save_period,
            patience=args.patience,
            val=args.val,
            plots=args.plots,
            
            # 其他
            pretrained=args.pretrained,
            verbose=args.verbose,
            seed=args.seed,
            rect=args.rect,
            close_mosaic=args.close_mosaic,
            fraction=args.fraction,
        )
        
        print("\n" + "=" * 80)
        print("✓ 训练完成!")
        print("=" * 80)
        print(f"\n最佳模型保存位置: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
        print(f"最后模型保存位置: {Path(args.project) / args.name / 'weights' / 'last.pt'}")
        
        # 显示最终指标
        if hasattr(results, 'results_dict'):
            print("\n最终指标:")
            metrics = results.results_dict
            if 'metrics/mAP50(B)' in metrics:
                print(f"  mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"  mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        return None
    except Exception as e:
        print(f"\n\n训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
