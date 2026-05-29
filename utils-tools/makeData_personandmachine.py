import glob
import os
import shutil
from sklearn.model_selection import train_test_split

# Load images from the test directory
image_list = glob.glob('data/test/*/*.*')
print(image_list)

# Set up new data directory for split
file_dir = 'newdata'
if os.path.exists(file_dir):
    shutil.rmtree(file_dir)  # Delete existing directory
os.makedirs(file_dir)  # Create new directory

# Split test files into manual and model testing
manual_test_files, model_test_files = train_test_split(image_list, test_size=0.5, random_state=42)

# Define directories for manual and model testing
manual_test_dir = 'person'
model_test_dir = 'machine'

# Create paths
manual_test_root = os.path.join(file_dir, manual_test_dir)
model_test_root = os.path.join(file_dir, model_test_dir)

# Function to copy files
def copy_files(files, root):
    for file in files:
        file_class = file.replace("\\", "/").split('/')[-2]
        file_name = file.replace("\\", "/").split('/')[-1]
        file_class_path = os.path.join(root, file_class)
        if not os.path.isdir(file_class_path):
            os.makedirs(file_class_path)
        shutil.copy(file, os.path.join(file_class_path, file_name))

# Copy manual test files
copy_files(manual_test_files, manual_test_root)

# Copy model test files
copy_files(model_test_files, model_test_root)
