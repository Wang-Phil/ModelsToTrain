import glob
import os
import shutil

image_list=glob.glob('data1/*/*.*')
print(image_list)
file_dir='data'
if os.path.exists(file_dir):
    print('true')
    #os.rmdir(file_dir)
    shutil.rmtree(file_dir)#删除再建立
    os.makedirs(file_dir)
else:
    os.makedirs(file_dir)

from sklearn.model_selection import train_test_split

train_files, temp_files = train_test_split(image_list, test_size=0.3, random_state=42)

val_files, test_files = train_test_split(temp_files, test_size=2/3, random_state=42)

train_dir='train'
val_dir='val'
test_dir='test'
train_root=os.path.join(file_dir,train_dir)
val_root=os.path.join(file_dir,val_dir)
test_root = os.path.join(file_dir, test_dir)


# 定义函数来复制文件
def copy_files(files, root):
    for file in files:
        file_class = file.replace("\\", "/").split('/')[-2]
        file_name = file.replace("\\", "/").split('/')[-1]
        file_class_path = os.path.join(root, file_class)
        if not os.path.isdir(file_class_path):
            os.makedirs(file_class_path)
        shutil.copy(file, os.path.join(file_class_path, file_name))

# 复制训练集文件
copy_files(train_files, train_root)

# 复制验证集文件
copy_files(val_files, val_root)

# 复制测试集文件
copy_files(test_files, test_root)