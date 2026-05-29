import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample

# 定义分类类别
classes = ('Acetabular Loosening', 'Dislocation', 'Fracture', 'Good Place', 'Native Hip', 'Spacer', 'Stem Loosening', 'Wear')

# 定义图像变换
transform_test = transforms.Compose([
    transforms.Resize((224, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.38205248, 0.38212535, 0.3820855], std=[0.2676706, 0.26768437, 0.26759237])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
model = torch.load('/data/wangweicheng/ModelsToTrains/checkpoints/ConvNext_tiny224*448/best93.564.pth')
model.eval()
model.to(DEVICE)

# 数据目录
test_path = './data/test/'
test_classes = sorted(os.listdir(test_path))  # 确保文件夹排序顺序一致

# 初始化列表来存储真实标签和预测得分
y_true = []
y_score = []

# 遍历每个类的文件夹
for idx, class_name in enumerate(test_classes):
    class_path = os.path.join(test_path, class_name)
    testList = sorted(os.listdir(class_path))  # 确保图像文件排序顺序一致

    for file in testList:
        img = Image.open(os.path.join(class_path, file))
        img = transform_test(img)
        img.unsqueeze_(0)
        img = Variable(img).to(DEVICE)
        
        with torch.no_grad():
            out = model(img)
            score = torch.softmax(out, dim=1)
        
        y_true.append(idx)  # 按顺序添加标签
        y_score.append(score.cpu().numpy()[0])  # 按顺序添加分数

# 转换为 numpy 数组
y_true = np.array(y_true)
y_score = np.array(y_score)

# 将标签转换为二进制形式
y_true_binary = label_binarize(y_true, classes=range(len(classes)))

# 计算每个类的 ROC 曲线并保存微平均和宏平均结果
fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}

for i in range(len(classes)):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_binary[:, i], y_score[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

print("各类AUC:")
print(roc_auc_dict)

# 计算微平均 ROC 曲线和 AUC
fpr_micro, tpr_micro, _ = roc_curve(y_true_binary.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# 计算宏平均 ROC 曲线和 AUC
all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(classes))]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(len(classes)):
    mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])

mean_tpr /= len(classes)
roc_auc_macro = auc(all_fpr, mean_tpr)
rng = np.random.RandomState(seed=42)
# 定义函数来计算 AUC 的置信区间
def calculate_auc_ci(y_true, y_score, n_bootstraps=1000, alpha=0.95):
    rng = np.random.RandomState(seed=42)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        # 通过随机抽样生成一个bootstrap样本
        indices = resample(np.arange(len(y_true)), random_state=rng)
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(y_true[indices], y_score[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower_bound = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + alpha) / 2 * 100)

    return lower_bound, upper_bound

# 计算每个类的AUC置信区间
auc_ci_dict = {}
for i in range(len(classes)):
    auc_ci_dict[i] = calculate_auc_ci(y_true_binary[:, i], y_score[:, i])

print("各类AUC的95%置信区间:")
for i in range(len(classes)):
    print(f"{classes[i]}: AUC = {roc_auc_dict[i]:.2f} (95% CI: {auc_ci_dict[i][0]:.2f} - {auc_ci_dict[i][1]:.2f})")


# 计算微平均AUC的置信区间
micro_ci = calculate_auc_ci(y_true_binary.ravel(), y_score.ravel())

# 计算宏平均AUC的置信区间（通过各类AUC的bootstrap均值）
macro_bootstrapped_scores = []

for _ in range(1000):
    indices = resample(range(len(classes)), random_state=rng)
    macro_bootstrapped_scores.append(np.mean([roc_auc_dict[i] for i in indices]))

macro_sorted_scores = np.sort(macro_bootstrapped_scores)
macro_lower_bound = np.percentile(macro_sorted_scores, 2.5)
macro_upper_bound = np.percentile(macro_sorted_scores, 97.5)

print(f"Micro平均AUC: {roc_auc_micro:.2f} (95% CI: {micro_ci[0]:.2f} - {micro_ci[1]:.2f})")
print(f"Macro平均AUC: {roc_auc_macro:.2f} (95% CI: {macro_lower_bound:.2f} - {macro_upper_bound:.2f})")
