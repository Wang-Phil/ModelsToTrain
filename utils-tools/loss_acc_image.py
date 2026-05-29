import json
import matplotlib.pyplot as plt

# 定义读取并绘制函数
def plot_results(file_dir):
    # 读取json文件
    with open(file_dir + '/result.json', 'r', encoding='utf-8') as file:
        logs = json.load(file)
    
    epoch_list = logs['epoch_list']
    train_loss_list = logs['train_loss']
    val_loss_list = logs['val_loss']
    train_acc_list = logs['train_acc']
    val_acc_list = logs['val_acc']

    # 绘制 Loss 曲线
    fig = plt.figure(1)
    plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
    plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
    plt.legend(["Train Loss", "Val Loss"], loc="upper right")
    plt.xlabel(u'epoch')
    plt.ylabel(u'loss')
    plt.title('Model Loss')
    plt.savefig(file_dir + "/loss.png")
    plt.close(1)

    # 绘制 Accuracy 曲线
    fig2 = plt.figure(2)
    plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
    plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
    plt.legend(["Train Acc", "Val Acc"], loc="lower right")
    plt.title("Model Acc")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.savefig(file_dir + "/acc.png")
    plt.close(2)

if __name__ == '__main__':
    file_dir = 'checkpoints/ConvNext/'  # 结果保存路径
    plot_results(file_dir)
