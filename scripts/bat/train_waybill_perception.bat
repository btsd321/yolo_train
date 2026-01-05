@echo off
REM ============================================================================
REM YOLO12 训练脚本 - 面单感知数据集
REM ============================================================================
REM 使用说明:
REM   直接双击运行使用默认参数
REM   或在命令行中使用自定义参数，例如:
REM   train_waybill_perception.bat --epochs 200 --batch 32
REM ============================================================================

echo ============================================================================
echo YOLO12 模型训练 - 面单感知数据集
echo ============================================================================
echo.

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

REM 检查数据集配置文件是否存在
set DATA_YAML=output\waybill_perception\dataset\coco.yaml
if not exist "%DATA_YAML%" (
    echo 错误: 数据集配置文件不存在: %DATA_YAML%
    echo 请先运行数据集划分脚本生成数据集
    pause
    exit /b 1
)

REM 默认参数
set MODEL=yolo12m.pt
set EPOCHS=300
set BATCH=16
set IMGSZ=640
set DEVICE=0
set PROJECT=runs\waybill_perception
set NAME=yolo12m_train

REM 解析命令行参数（如果有）
:parse_args
if "%~1"=="" goto start_train
if /i "%~1"=="--model" (
    set MODEL=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--batch" (
    set BATCH=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--device" (
    set DEVICE=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--name" (
    set NAME=%~2
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:start_train
echo 训练配置:
echo   数据集: %DATA_YAML%
echo   模型: %MODEL%
echo   训练轮数: %EPOCHS%
echo   批次大小: %BATCH%
echo   图像尺寸: %IMGSZ%
echo   设备: %DEVICE%
echo   保存路径: %PROJECT%\%NAME%
echo.
echo ----------------------------------------------------------------------------
echo 开始训练...
echo ----------------------------------------------------------------------------
echo.

REM 执行训练
python tools\train.py ^
    --data "%DATA_YAML%" ^
    --model "%MODEL%" ^
    --epochs %EPOCHS% ^
    --batch %BATCH% ^
    --imgsz %IMGSZ% ^
    --device %DEVICE% ^
    --optimizer auto ^
    --lr0 0.01 ^
    --lrf 0.01 ^
    --weight-decay 0.0005 ^
    --project "%PROJECT%" ^
    --name "%NAME%" ^
    --workers 8 ^
    --patience 50 ^
    --save-period -1 ^
    --mosaic 1.0 ^
    --mixup 0.0 ^
    --fliplr 0.5 ^
    --hsv-h 0.015 ^
    --hsv-s 0.7 ^
    --hsv-v 0.4 ^
    --pretrained ^
    --amp ^
    --plots ^
    --verbose

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
