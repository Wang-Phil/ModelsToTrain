#!/bin/bash
# Linux文件恢复脚本 - 针对models目录下删除的文件

echo "=========================================="
echo "Linux文件恢复工具 - models目录"
echo "=========================================="
echo ""

MODELS_DIR="/home/ln/wangweicheng/ModelsTotrain/models"
RECOVERY_DIR="/home/ln/wangweicheng/ModelsTotrain/recovered_files"

# 创建恢复目录
mkdir -p "$RECOVERY_DIR"

echo "方法1: 检查是否有备份或临时文件..."
echo "----------------------------------------"
# 检查常见备份位置
BACKUP_LOCATIONS=(
    "$HOME/.local/share/Trash"
    "$HOME/Trash"
    "/tmp"
    "$MODELS_DIR/../backup"
    "$MODELS_DIR/../models_backup"
)

for location in "${BACKUP_LOCATIONS[@]}"; do
    if [ -d "$location" ]; then
        echo "检查: $location"
        find "$location" -name "*.py" -type f 2>/dev/null | grep -i model | head -5
    fi
done

echo ""
echo "方法2: 尝试从.pyc文件反编译（需要安装uncompyle6或decompyle3）..."
echo "----------------------------------------"
if command -v uncompyle6 &> /dev/null || command -v decompyle3 &> /dev/null; then
    echo "找到反编译工具，尝试恢复..."
    for pyc_file in "$MODELS_DIR/__pycache__"/*.pyc; do
        if [ -f "$pyc_file" ]; then
            filename=$(basename "$pyc_file" .cpython-*.pyc)
            output_file="$RECOVERY_DIR/${filename}_recovered.py"
            echo "反编译: $pyc_file -> $output_file"
            if command -v uncompyle6 &> /dev/null; then
                uncompyle6 "$pyc_file" > "$output_file" 2>/dev/null || true
            elif command -v decompyle3 &> /dev/null; then
                decompyle3 "$pyc_file" > "$output_file" 2>/dev/null || true
            fi
        fi
    done
    echo "反编译的文件保存在: $RECOVERY_DIR"
else
    echo "未安装反编译工具。安装方法："
    echo "  pip install uncompyle6"
    echo "  或"
    echo "  pip install decompyle3"
fi

echo ""
echo "方法3: 使用testdisk/photorec恢复（需要安装）..."
echo "----------------------------------------"
if command -v photorec &> /dev/null; then
    echo "photorec已安装，可以使用以下命令恢复："
    echo "  sudo photorec /log /home/ln/wangweicheng/ModelsTotrain/recovered_files"
    echo "  然后选择磁盘 -> 选择分区 -> 选择文件系统类型 -> 选择恢复位置"
else
    echo "photorec未安装。安装方法："
    echo "  Ubuntu/Debian: sudo apt-get install testdisk"
    echo "  CentOS/RHEL: sudo yum install testdisk"
fi

echo ""
echo "方法4: 检查最近修改的文件（可能包含已删除文件的信息）..."
echo "----------------------------------------"
echo "最近7天修改的Python文件："
find "$MODELS_DIR/.." -name "*.py" -type f -mtime -7 2>/dev/null | head -10

echo ""
echo "方法5: 检查是否有git历史记录..."
echo "----------------------------------------"
if [ -d "$MODELS_DIR/../.git" ]; then
    echo "发现git仓库，可以使用以下命令恢复："
    echo "  cd $MODELS_DIR/.."
    echo "  git log --all --full-history -- models/"
    echo "  git checkout <commit-hash> -- models/<filename>"
else
    echo "未发现git仓库"
fi

echo ""
echo "=========================================="
echo "恢复建议："
echo "1. 首先检查回收站和备份位置"
echo "2. 如果文件最近删除，立即停止写入操作"
echo "3. 尝试使用photorec进行深度扫描"
echo "4. 检查是否有版本控制系统（git/svn）"
echo "5. 从.pyc文件反编译（部分恢复）"
echo "=========================================="


