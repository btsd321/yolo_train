@echo off
REM ============================================================================
REM YOLO12 断点续训脚本
REM ============================================================================
REM 使用说明:
REM   方式1: 直接双击运行，会提示输入权重文件路径
REM   方式2: 在命令行中指定权重文件路径
REM          resume_train.bat runs\waybill_perception\yolo12m_train\weights\last.pt
REM ============================================================================

echo ============================================================================
echo YOLO12 模型断点续训
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

REM 获取权重文件路径
set WEIGHT_PATH=%~1

REM 如果没有指定权重路径，提示用户输入
if "%WEIGHT_PATH%"=="" (
    echo 请输入权重文件路径 ^(相对于项目根目录^):
    echo 例如: runs\waybill_perception\yolo12m_train\weights\last.pt
    echo.
    set /p WEIGHT_PATH="权重路径: "
)

REM 检查权重文件是否存在
if not exist "%WEIGHT_PATH%" (
    echo.
    echo 错误: 权重文件不存在: %WEIGHT_PATH%
    echo.
    echo 常见权重文件位置:
    dir /b /s runs\*\weights\last.pt 2>nul
    pause
    exit /b 1
)

echo.
echo 从以下权重恢复训练:
echo   %WEIGHT_PATH%
echo.
echo ----------------------------------------------------------------------------
echo 开始续训...
echo ----------------------------------------------------------------------------
echo.

REM 执行断点续训
python tools\train.py --resume "%WEIGHT_PATH%"

if errorlevel 1 (
    echo.
    echo ============================================================================
    echo ✗ 续训失败
    echo ============================================================================
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo ✓ 续训完成
echo ============================================================================
pause
