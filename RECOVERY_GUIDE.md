# Linux文件恢复指南 - models目录

## 快速恢复方法

### 方法1: 检查回收站和备份
```bash
# 检查常见回收站位置
ls -la ~/.local/share/Trash/files/
ls -la ~/Trash/

# 检查是否有备份目录
find /home/ln/wangweicheng -name "*backup*" -type d 2>/dev/null
find /home/ln/wangweicheng -name "*models*" -type d 2>/dev/null
```

### 方法2: 使用photorec恢复（推荐）
photorec是testdisk套件的一部分，可以恢复各种文件系统上删除的文件。

**安装：**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install testdisk

# CentOS/RHEL
sudo yum install testdisk
```

**使用步骤：**
1. 运行恢复脚本：
   ```bash
   chmod +x recover_deleted_files.sh
   ./recover_deleted_files.sh
   ```

2. 或手动运行photorec：
   ```bash
   sudo photorec /log /home/ln/wangweicheng/ModelsTotrain/recovered_files
   ```
   
3. 在photorec界面中：
   - 选择包含models目录的磁盘分区
   - 选择文件系统类型（XFS）
   - 选择恢复位置
   - 等待扫描完成

### 方法3: 从.pyc文件反编译
如果models目录下有`__pycache__`文件夹，可以尝试从编译后的.pyc文件恢复源代码。

**安装反编译工具：**
```bash
pip install uncompyle6
# 或
pip install decompyle3
```

**反编译：**
```bash
# 使用uncompyle6
uncompyle6 models/__pycache__/filename.cpython-*.pyc > recovered_file.py

# 批量反编译
for pyc in models/__pycache__/*.pyc; do
    filename=$(basename "$pyc" .cpython-*.pyc)
    uncompyle6 "$pyc" > "recovered_files/${filename}_recovered.py"
done
```

### 方法4: 检查文件系统日志（XFS）
对于XFS文件系统，可以尝试：
```bash
# 检查是否有文件系统日志
sudo xfs_logprint /dev/sda1 | grep -i "models" | tail -50
```

### 方法5: 使用extundelete（仅限ext文件系统）
如果文件系统是ext2/3/4：
```bash
sudo apt-get install extundelete
sudo extundelete /dev/sda1 --restore-directory models/
```

## 重要提示

1. **立即停止写入操作**：删除文件后，立即停止对该磁盘的写入操作，避免覆盖已删除的文件数据。

2. **优先使用photorec**：对于XFS文件系统，photorec是最可靠的工具。

3. **检查备份**：优先检查是否有自动备份或手动备份。

4. **部分恢复**：即使无法完全恢复，从.pyc文件反编译也能恢复部分代码。

5. **预防措施**：
   - 使用git进行版本控制
   - 定期备份重要文件
   - 使用rsync或tar定期备份

## 运行恢复脚本

```bash
cd /home/ln/wangweicheng/ModelsTotrain
chmod +x recover_deleted_files.sh
./recover_deleted_files.sh
```

脚本会自动：
- 检查常见备份位置
- 尝试从.pyc文件反编译
- 提供详细的恢复建议

## 恢复后的文件位置

恢复的文件将保存在：
```
/home/ln/wangweicheng/ModelsTotrain/recovered_files/
```

