@echo off
REM ============================================================================
REM YOLO12 自定义训练脚本
REM ============================================================================
REM 使用说明:
REM   此脚本允许你自定义所有训练参数
REM   编辑下方的参数配置部分，然后双击运行
REM ============================================================================

echo ============================================================================
echo YOLO12 自定义训练
echo ============================================================================
echo.

REM ============================================================================
REM 参数配置 - 在此处修改你的训练参数
REM ============================================================================

REM 数据集配置
set DATA_YAML=output\waybill_perception\dataset\coco.yaml

REM 模型配置
REM 可选: yolo12n.pt, yolo12s.pt, yolo12m.pt, yolo12l.pt, yolo12x.pt
set MODEL=yolo12m.pt
set PRETRAINED=--pretrained

REM 训练参数
set EPOCHS=100
set BATCH=16
set IMGSZ=640
set DEVICE=0

REM 优化器参数
set OPTIMIZER=auto
set LR0=0.01
set LRF=0.01
set MOMENTUM=0.937
set WEIGHT_DECAY=0.0005
set COS_LR=

REM 数据增强
set MOSAIC=1.0
set MIXUP=0.0
set COPY_PASTE=0.0
set FLIPLR=0.5
set FLIPUD=0.0
set HSV_H=0.015
set HSV_S=0.7
set HSV_V=0.4
set DEGREES=0.0
set TRANSLATE=0.1
set SCALE=0.5
set SHEAR=0.0
set PERSPECTIVE=0.0

REM 保存和验证
set PROJECT=runs\waybill_perception
set NAME=yolo12m_custom
set SAVE_PERIOD=-1
set PATIENCE=50
set WORKERS=8

REM 其他选项
set AMP=--amp
set CACHE=
set RECT=
set CLOSE_MOSAIC=10
set FRACTION=1.0
set SEED=0

REM ============================================================================
REM 以下部分通常不需要修改
REM ============================================================================

REM 激活conda环境
echo 正在激活 conda 环境 dl_env...
call conda activate dl_env
if errorlevel 1 (
    echo 错误: 无法激活 conda 环境 dl_env
    pause
    exit /b 1
)
echo ✓ Conda 环境已激活
echo.

REM 切换到项目根目录
cd /d "%~dp0.."

REM 检查数据集配置文件
if not exist "%DATA_YAML%" (
    echo 错误: 数据集配置文件不存在: %DATA_YAML%
    pause
    exit /b 1
)

echo 训练配置:
echo   数据集: %DATA_YAML%
echo   模型: %MODEL% %PRETRAINED%
echo   训练轮数: %EPOCHS%
echo   批次大小: %BATCH%
echo   图像尺寸: %IMGSZ%
echo   设备: %DEVICE%
echo   优化器: %OPTIMIZER% (lr0=%LR0%, lrf=%LRF%, wd=%WEIGHT_DECAY%)
echo   数据增强: mosaic=%MOSAIC%, mixup=%MIXUP%, fliplr=%FLIPLR%
echo   保存路径: %PROJECT%\%NAME%
echo.
echo ----------------------------------------------------------------------------
echo 开始训练...
echo ----------------------------------------------------------------------------
echo.

REM 构建训练命令
set TRAIN_CMD=python tools\train.py ^
    --data "%DATA_YAML%" ^
    --model "%MODEL%" ^
    --epochs %EPOCHS% ^
    --batch %BATCH% ^
    --imgsz %IMGSZ% ^
    --device %DEVICE% ^
    --optimizer %OPTIMIZER% ^
    --lr0 %LR0% ^
    --lrf %LRF% ^
    --momentum %MOMENTUM% ^
    --weight-decay %WEIGHT_DECAY% ^
    --project "%PROJECT%" ^
    --name "%NAME%" ^
    --workers %WORKERS% ^
    --patience %PATIENCE% ^
    --save-period %SAVE_PERIOD% ^
    --mosaic %MOSAIC% ^
    --mixup %MIXUP% ^
    --copy-paste %COPY_PASTE% ^
    --fliplr %FLIPLR% ^
    --flipud %FLIPUD% ^
    --hsv-h %HSV_H% ^
    --hsv-s %HSV_S% ^
    --hsv-v %HSV_V% ^
    --degrees %DEGREES% ^
    --translate %TRANSLATE% ^
    --scale %SCALE% ^
    --shear %SHEAR% ^
    --perspective %PERSPECTIVE% ^
    --close-mosaic %CLOSE_MOSAIC% ^
    --fraction %FRACTION% ^
    --seed %SEED% ^
    --plots ^
    --verbose

REM 添加可选参数
if not "%PRETRAINED%"=="" set TRAIN_CMD=%TRAIN_CMD% %PRETRAINED%
if not "%AMP%"=="" set TRAIN_CMD=%TRAIN_CMD% %AMP%
if not "%CACHE%"=="" set TRAIN_CMD=%TRAIN_CMD% --cache %CACHE%
if not "%RECT%"=="" set TRAIN_CMD=%TRAIN_CMD% %RECT%
if not "%COS_LR%"=="" set TRAIN_CMD=%TRAIN_CMD% %COS_LR%

REM 执行训练
%TRAIN_CMD%

if errorlevel 1 (
    echo.
    echo ============================================================================
    echo ✗ 训练失败
    echo ============================================================================
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo ✓ 训练完成
echo ============================================================================
echo 最佳模型: %PROJECT%\%NAME%\weights\best.pt
echo 最后模型: %PROJECT%\%NAME%\weights\last.pt
echo ============================================================================
pause
