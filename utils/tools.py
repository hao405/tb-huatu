import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=False):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


# 该文件路径: utils/tools.py
# 请将此函数添加到文件末尾

def visualize_c_est(input_data, c_est_data, save_dir, epoch, phase_name, feature_dim=0):
    """
    接口函数：绘制输入特征与 c_est 状态的对比图。

    参数:
        input_data: 输入序列 (Tensor), shape [Batch, Seq, Feat]
        c_est_data: 状态序列 (Tensor), shape [Batch, Seq]
        save_dir: 图片保存文件夹路径
        epoch: 当前的 Epoch 数
        phase_name: 阶段名称 (用于文件名)
        feature_dim: 要可视化的特征维度索引 (默认 0)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 数据预处理：取 Batch 中的第 0 个样本，并转为 numpy
    # 如果已经是 numpy 则不转换，如果是 Tensor 则 detach 并转 cpu
    if isinstance(input_data, torch.Tensor):
        inp = input_data[0].detach().cpu().numpy()
    else:
        inp = input_data[0]

    if isinstance(c_est_data, torch.Tensor):
        est = c_est_data[0].detach().cpu().numpy()
    else:
        est = c_est_data[0]

    seq_len = inp.shape[0]

    # 2. 绘图
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # 左轴：输入特征
    color1 = 'tab:blue'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel(f'Input Feature (Dim {feature_dim})', color=color1)
    ax1.plot(inp[:, feature_dim], color=color1, label='Input Feature', linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 右轴：c_est 状态 (使用阶梯图 step)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('c_est (Discrete State)', color=color2)
    ax2.step(range(seq_len), est, color=color2, where='mid', label='c_est', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f'Training Visualization - {phase_name} - Epoch {epoch}')
    plt.tight_layout()

    # 3. 保存
    file_name = f"{phase_name}_Epoch{epoch}.png"
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()
    # print(f"Saved visualization to {os.path.join(save_dir, file_name)}")