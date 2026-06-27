# ============================================================================
# YOLO26 End-to-End Training Script - Linden Perception Dataset
# Optimized: cls/box weight adjustment to reduce false positives, MixUp for generalization, class weight balancing
# ============================================================================
param(
    # 预训练模型路径或模型配置文件 .yaml [默认: 见下]
    [string]$Model      = "D:\Project\yolo_train\Data\dataset\linden_perception_dataset_hefei_20260624\parcel_seg40_640x_hefei.pt",

    # 训练轮数 [默认: 100]
    [int]$Epochs         = 300,

    # 批次大小（每GPU） [默认: 16]
    [int]$Batch          = 8,

    # 输入图像尺寸（像素，正方形） [默认: 640]
    [int]$Imgsz          = 640,

    # 训练设备：0/1 单卡，"0,1" 多卡，"cpu" [默认: 自动选择]
    [string]$Device      = "0",

    # 输出根目录（位于 runs/ 下） [默认: linden_perception]
    [string]$Project     = "linden_perception",

    # 实验名称，结果保存在 Project/Name 下 [默认: yolo26m_train_20260627]
    [string]$Name        = "yolo26m_train_20260627",

    # 分类loss权重，越大越能减少误判 [默认: 0.5]
    [float]$Cls          = 1.0,

    # 框回归loss权重，越小则模型越专注分类 [默认: 7.5]
    [float]$Box          = 5.0,

    # 类别频率权重指数：0.0=禁用，1.0=完全逆频率加权 [默认: 0.0]
    [float]$ClsPw        = 1.0,

    # MixUp增强概率：0.0=禁用 [默认: 0.0]
    [float]$Mixup        = 0.0,

    # 权重衰减L2正则化，越大抗过拟合越强 [默认: 0.0005]
    [float]$WeightDecay  = 0.0005,

    # 优化器：SGD/Adam/AdamW/NAdam/RAdam/RMSProp/auto [默认: auto]
    [string]$Optimizer   = "auto",

    # 初始学习率 [默认: 0.01]
    [float]$Lr0          = 0.01,

    # 最终学习率因子：final_lr = lr0 * lrf [默认: 0.01]
    [float]$Lrf          = 0.01,

    # 数据加载线程数 [默认: 8]
    [int]$Workers        = 8,

    # 早停耐心值：连续N个epoch无改善则停止 [默认: 100]
    [int]$Patience       = 100,

    # 每N个epoch保存一次模型，-1=仅保存last和best [默认: -1]
    [int]$SavePeriod     = -1,

    # Mosaic增强概率，1.0=训练期间始终开启 [默认: 1.0]
    [float]$Mosaic       = 1.0,

    # 水平翻转概率 [默认: 0.5]
    [float]$FlipLr       = 0.5,

    # HSV色调增强范围（百分比） [默认: 0.015]
    [float]$HsvH         = 0.015,

    # HSV饱和度增强范围（百分比） [默认: 0.7]
    [float]$HsvS         = 0.7,

    # HSV明度增强范围（百分比） [默认: 0.4]
    [float]$HsvV         = 0.4
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================================"
Write-Host "YOLO26 End-to-End Model Training - Linden Perception Dataset"
Write-Host "============================================================================"
Write-Host ""

# Switch to project root directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..\..")

# Activate virtual environment
$ProjectRoot = (Get-Location).Path
Write-Host "Activating .venv virtual environment..."
$VenvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
    Write-Host "ERROR: Virtual environment not found: $VenvPath" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
. (Join-Path $VenvPath "Scripts\Activate.ps1")
Write-Host "OK .venv virtual environment activated"
Write-Host ""

# Set ultralytics config directory
$YoloConfigDir = Join-Path $ProjectRoot ".ultralytics"
if (-not (Test-Path $YoloConfigDir)) {
    New-Item -ItemType Directory -Path $YoloConfigDir | Out-Null
}
Write-Host "YOLO_CONFIG_DIR: $YoloConfigDir"

# Check dataset config file exists
$DataYaml = "D:\Project\yolo_train\tools\custom\linden_perception\config\dataset.yaml"
if (-not (Test-Path $DataYaml)) {
    Write-Host "ERROR: Dataset config file not found: $DataYaml" -ForegroundColor Red
    Write-Host "Please run the dataset split script first"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Training Configuration:"
Write-Host "  Dataset:        $DataYaml"
Write-Host "  Model:          $Model"
Write-Host "  Epochs:         $Epochs"
Write-Host "  Batch Size:     $Batch"
Write-Host "  Image Size:     $Imgsz"
Write-Host "  Device:         $Device"
Write-Host "  Cls Weight:     $Cls"
Write-Host "  Box Weight:     $Box"
Write-Host "  Class PW Index: $ClsPw"
Write-Host "  MixUp:          $Mixup"
Write-Host "  Weight Decay:   $WeightDecay"
Write-Host "  Save Path:      $Project\$Name"
Write-Host ""
Write-Host "----------------------------------------------------------------------------"
Write-Host "Starting training..."
Write-Host "----------------------------------------------------------------------------"
Write-Host ""

# Use venv python.exe directly (Start-Process does not inherit venv PATH)
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

# Execute training
$PythonArgs = @(
    "tools\train.py",
    "--data", $DataYaml,
    "--model", $Model,
    "--epochs", $Epochs,
    "--batch", $Batch,
    "--imgsz", $Imgsz,
    "--device", $Device,
    "--cls", $Cls,
    "--box", $Box,
    "--cls-pw", $ClsPw,
    "--mixup", $Mixup,
    "--weight-decay", $WeightDecay,
    "--optimizer", $Optimizer,
    "--lr0", $Lr0,
    "--lrf", $Lrf,
    "--project", $Project,
    "--name", $Name,
    "--workers", $Workers,
    "--patience", $Patience,
    "--save-period", $SavePeriod,
    "--mosaic", $Mosaic,
    "--fliplr", $FlipLr,
    "--hsv-h", $HsvH,
    "--hsv-s", $HsvS,
    "--hsv-v", $HsvV,
    "--pretrained",
    "--amp",
    "--plots",
    "--verbose"
)

$Process = Start-Process -FilePath $PythonExe -ArgumentList $PythonArgs -NoNewWindow -Wait -PassThru

if ($Process.ExitCode -ne 0) {
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Red
    Write-Host "X Training Failed" -ForegroundColor Red
    Write-Host "============================================================================" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "OK Training Completed" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "Best Model:  $Project\$Name\weights\best.pt"
Write-Host "Last Model:  $Project\$Name\weights\last.pt"
Write-Host "============================================================================" -ForegroundColor Green
Read-Host "Press Enter to exit"
