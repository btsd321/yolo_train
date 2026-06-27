# ============================================================================
# YOLO26 End-to-End Training Script - Linden Perception Dataset
# Optimized: cls/box weight adjustment to reduce false positives, MixUp for generalization, class weight balancing
# ============================================================================
param(
    [string]$Model      = "D:\Project\yolo_train\Data\dataset\linden_perception_dataset_hefei_20260624\parcel_seg40_640x_hefei.pt",
    [int]$Epochs         = 300,
    [int]$Batch          = 16,
    [int]$Imgsz          = 640,
    [string]$Device      = "0",
    [string]$Project     = "linden_perception",
    [string]$Name        = "yolo26m_train_20260627",
    [float]$Cls          = 1.0,
    [float]$Box          = 5.0,
    [float]$ClsPw        = 1.0,
    [float]$Mixup        = 0.0,
    [float]$WeightDecay  = 0.001,
    [string]$Optimizer   = "auto",
    [float]$Lr0          = 0.01,
    [float]$Lrf          = 0.01,
    [int]$Workers        = 8,
    [int]$Patience       = 50,
    [int]$SavePeriod     = -1,
    [float]$Mosaic       = 1.0,
    [float]$FlipLr       = 0.5,
    [float]$HsvH         = 0.015,
    [float]$HsvS         = 0.7,
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
