# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO training repository based on ultralytics YOLO framework (Python 3.10.12). Used for object detection model training, annotation conversion, and dataset preparation.

## Key Dependencies

- ultralytics YOLO (custom fork): Located in `thirdparty/ultralytics` as a git submodule from https://github.com/btsd321/ultralytics.git
- Python 3.10.12
- Conda environment: `dl_env`

## Common Commands

### Training

```bash
# Basic training with template script
python tools/train.py --data <dataset.yaml> --model yolo12n.pt --epochs 100

# Resume training from checkpoint
python tools/train.py --resume ./runs/detect/train/weights/last.pt

# Multi-GPU training
python tools/train.py --data <dataset.yaml> --model yolo12l.pt --device 0,1

# Using batch scripts (Windows)
scripts/bat/train_custom.bat          # Customizable training parameters
scripts/bat/train_waybill_perception.bat  # Project-specific training
scripts/bat/resume_train.bat          # Resume from checkpoint
```

### Dataset Generation

```bash
# Convert images + YOLO labels to YOLO dataset structure
python tools/generate_dataset.py -i <input_folder> -o <output_folder>

# Custom split ratios
python tools/generate_dataset.py -i ./data/annotations -o ./dataset --train 0.7 --val 0.2 --test 0.1
```

### Visualization

```bash
# Visualize YOLO bounding boxes
python tools/visualization/yolo/yolo_bbox_visualization.py

# Visualize YOLO segmentation masks
python tools/visualization/yolo/yolo_segment_visualization.py
```

## Architecture

### Directory Structure

- `Data/`: Datasets and annotation files (not in repo)
- `scripts/`: Training and utility scripts
  - `bat/`: Windows batch scripts for training workflows
  - `sh/`: Linux shell scripts (e.g., compression utilities)
- `tools/`: Core utilities and scripts
  - `train.py`: Template training script (reference implementation)
  - `generate_dataset.py`: Convert image+label folders to YOLO dataset format
  - `conversion/`: Annotation format conversion tools
    - `bbox/`: Bounding box conversions (XML, JSON, XanyLabel, YOLO)
    - `segment/`: Polygon/segmentation conversions
    - `segment_to_bbox/`: Convert segmentation to bounding boxes
    - `segment_to_obb/`: Convert segmentation to oriented bounding boxes
  - `custom/`: Project-specific implementations (non-generic code)
    - Each subdirectory is an independent project with its own training scripts
    - Class definition files typically in `doc/` subdirectories
  - `visualization/`: Visualization tools for different annotation formats
- `thirdparty/ultralytics`: Custom ultralytics fork (git submodule)

### Training Workflow

1. **Template vs Custom**: `tools/train.py` is a reference implementation. Actual projects use custom training scripts in `tools/custom/<project_name>/train/`
2. **Custom Projects**: Located in `tools/custom/`, each has independent functionality and class definitions
3. **Class Files**: Stored in project-specific `doc/` folders, format: `{class_id} {class_name}` per line (class_id may not be continuous)

## Annotation Formats

Supports multiple annotation formats:
- YOLO (txt): Native format
- XML: Various XML schemas
- JSON: Custom JSON formats
- XanyLabel: JSON-based labeling format

### Class ID Handling

Class IDs in input files may not be continuous. Conversion tools must:
1. Check for non-continuous class IDs
2. Remap to continuous IDs starting from 0 when outputting YOLO format
3. Maintain mapping documentation

## Code Conventions

### Argument Parsing

All scripts use argparse with consistent parameter names:
- `--names`: Class definition file path
- `--input` or `-i`: Input file/folder path
- `--output` or `-o`: Output file/folder path (auto-create if missing)
- `--data`: Dataset YAML configuration (for training)

### Code Style

- Keep functions modular with single responsibilities
- Auto-create output directories when they don't exist
- Include comments for complex logic and key steps
- Update documentation when code changes

## Dataset Format

YOLO dataset structure (output of `generate_dataset.py`):
```
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
```
