#!/bin/bash

# 智能压缩脚本：根据文件夹大小选择压缩方式
# 超过阈值使用分卷压缩，否则使用tar+pigz压缩

set -e

# 默认参数
INPUT_FOLDER=""
THRESHOLD="20G"  # 默认阈值
THREADS=$(nproc)  # 默认使用所有CPU核心
SPLIT_SIZE="10G"  # 分卷大小
OUTPUT_DIR="."   # 输出目录，默认当前目录

# 帮助信息
show_help() {
    cat << EOF
用法: $0 -i <文件夹路径> [选项]

必需参数:
  -i, --input <路径>        输入文件夹路径

可选参数:
  -t, --threshold <大小>    阈值大小，超过则分卷压缩
                            支持的单位: K, M, G, T
  -p, --threads <数量>      压缩线程数（默认: CPU核心数）
  -s, --split-size <大小>   分卷大小
  -o, --output <路径>       输出目录（默认: 当前目录）
  -h, --help               显示此帮助信息

示例:
  $0 -i /path/to/folder
  $0 -i /path/to/folder -t 20G -p 8
  $0 -i /path/to/folder -t 5G -s 1G -o /output/dir

EOF
}

# 将大小转换为字节
size_to_bytes() {
    local size=$1
    local number=${size%[KMGT]*}
    local unit=${size#${number}}
    
    case "$unit" in
        K|k) echo $((number * 1024)) ;;
        M|m) echo $((number * 1024 * 1024)) ;;
        G|g) echo $((number * 1024 * 1024 * 1024)) ;;
        T|t) echo $((number * 1024 * 1024 * 1024 * 1024)) ;;
        *) echo "$number" ;;
    esac
}

# 格式化字节为人类可读格式
bytes_to_human() {
    local bytes=$1
    if [ $bytes -lt 1024 ]; then
        echo "${bytes}B"
    elif [ $bytes -lt $((1024*1024)) ]; then
        echo "$((bytes/1024))K"
    elif [ $bytes -lt $((1024*1024*1024)) ]; then
        echo "$((bytes/1024/1024))M"
    else
        echo "$((bytes/1024/1024/1024))G"
    fi
}

# 获取文件夹大小（字节）
get_folder_size() {
    local folder=$1
    du -sb "$folder" | awk '{print $1}'
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -p|--threads)
            THREADS="$2"
            shift 2
            ;;
        -s|--split-size)
            SPLIT_SIZE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知参数 '$1'"
            show_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$INPUT_FOLDER" ]; then
    echo "错误: 必须指定输入文件夹路径 (-i)"
    show_help
    exit 1
fi

# 检查输入文件夹是否存在
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "错误: 输入文件夹不存在: $INPUT_FOLDER"
    exit 1
fi

# 检查必需工具
if ! command -v pigz &> /dev/null; then
    echo "错误: 未找到 pigz 工具，请先安装: sudo apt-get install pigz"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 获取文件夹名称（去掉路径）
FOLDER_NAME=$(basename "$INPUT_FOLDER")

# 转换阈值为字节
THRESHOLD_BYTES=$(size_to_bytes "$THRESHOLD")

# 获取文件夹大小
echo "正在计算文件夹大小..."
FOLDER_SIZE=$(get_folder_size "$INPUT_FOLDER")
FOLDER_SIZE_HUMAN=$(bytes_to_human "$FOLDER_SIZE")

echo "============================================"
echo "文件夹: $INPUT_FOLDER"
echo "大小: $FOLDER_SIZE_HUMAN ($FOLDER_SIZE 字节)"
echo "阈值: $THRESHOLD ($THRESHOLD_BYTES 字节)"
echo "线程数: $THREADS"
echo "============================================"

# 根据大小选择压缩方式
if [ $FOLDER_SIZE -gt $THRESHOLD_BYTES ]; then
    echo "文件夹大小超过阈值，使用分卷压缩..."
    OUTPUT_FILE="${OUTPUT_DIR}/${FOLDER_NAME}.tar.gz"
    
    # 分卷压缩
    echo "开始分卷压缩（每卷 $SPLIT_SIZE）..."
    tar -cvf - -C "$(dirname "$INPUT_FOLDER")" "$FOLDER_NAME" | \
        pigz -p $THREADS | \
        split -b "$SPLIT_SIZE" - "${OUTPUT_FILE}."
    
    echo ""
    echo "分卷压缩完成！"
    echo "输出文件: ${OUTPUT_FILE}.*"
    echo ""
    echo "解压方法:"
    echo "  cat ${OUTPUT_FILE}.* | pigz -d | tar -xvf -"
    
else
    echo "文件夹大小未超过阈值，使用普通压缩..."
    OUTPUT_FILE="${OUTPUT_DIR}/${FOLDER_NAME}.tar.gz"
    
    # 普通压缩
    echo "开始压缩..."
    tar -cvf - -C "$(dirname "$INPUT_FOLDER")" "$FOLDER_NAME" | \
        pigz -p $THREADS > "$OUTPUT_FILE"
    
    echo ""
    echo "压缩完成！"
    echo "输出文件: $OUTPUT_FILE"
    
    # 显示压缩后大小
    if [ -f "$OUTPUT_FILE" ]; then
        COMPRESSED_SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null)
        COMPRESSED_SIZE_HUMAN=$(bytes_to_human "$COMPRESSED_SIZE")
        RATIO=$((100 - COMPRESSED_SIZE * 100 / FOLDER_SIZE))
        echo "压缩后大小: $COMPRESSED_SIZE_HUMAN"
        echo "压缩率: ${RATIO}%"
    fi
    
    echo ""
    echo "解压方法:"
    echo "  pigz -dc $OUTPUT_FILE | tar -xvf -"
fi

echo ""
echo "完成！"