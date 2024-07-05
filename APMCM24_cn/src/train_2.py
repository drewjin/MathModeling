import sys 
sys.path.append("/Users/drew/Documents/MathModeling/MathModeling/APMCM24_cn/src")
from tqdm import tqdm
from torch.utils.data import DataLoader

import torch 
import torch.nn as nn

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as skl
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt

from os.path import join
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from models import AttnRegressor
from utils import Dataset_2

low, mid, high = [0, 0.4724686843808853], [0.4724686843808853, 0.537526261188053], [0.537526261188053, 1]

data_root = r'/Users/drew/Documents/MathModeling/MathModeling/APMCM24_cn/data'
train_path = join(data_root, 'train_cluster.csv')
train = pd.read_csv(train_path, index_col='id')
initial_features = list(train.columns)[:-2]

net = AttnRegressor(nheads=8)
num_epochs = 100
batch_size = 64
lr = 1e-3
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

X_train, X_test, y_train, y_test = [
    torch.from_numpy(array.values) 
    for array in train_test_split(
    train[initial_features], train[['洪水概率', '聚类']], 
    test_size=0.1, random_state=114514)
]

train_set = Dataset_2(X_train, y_train[:, 0], y_train[:, 1])
test_set = Dataset_2(X_test, y_test[:, 0], y_test[:, 1])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 根据设备选择CPU或GPU
net.to(device)  # 将模型移动到设备上

def _classify(O_reg, label_cls, low, mid, high):
    cnt = 0
    for _, (o_reg, l_cls) in enumerate(zip(O_reg, label_cls)):
        o_cls = None
        if o_reg < low[1]:
            o_cls = 0
        elif o_reg >= mid[0] and o_reg < mid[1]:
            o_cls = 1
        elif o_reg >= high[0] and o_reg <= high[1]:
            o_cls = 2
        if o_cls == l_cls.item():
            cnt += 1
    return cnt

def _evaluate(net, data_loader, criterion, device):
    net.eval()  # 将模型设置为评估模式
    running_loss = 0.0
    total_samples = len(data_loader)
    CNT = 0
    total_MAE = 0
    total_R2 = 0
    with torch.no_grad():  # 在评估过程中不计算梯度
        with tqdm(total=len(data_loader), desc=f"<TEST>") as pbar:
            for batch_idx, (X_batch, (y_batch, y_cls)) in enumerate(data_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                O = net(X_batch)
                loss = criterion(O.squeeze(-1), y_batch)
                running_loss += loss.item() * X_batch.size(0)  # 累加损失
                CNT += _classify(O, y_cls, low, mid, high)
                ACC3 = CNT / ((batch_idx+1)*batch_size)
                MAE = mean_absolute_error(y_batch.cpu().numpy(), O.squeeze(-1).detach().cpu().numpy())
                R2 = r2_score(y_batch.cpu().numpy(), O.squeeze(-1).detach().cpu().numpy())
                pbar.set_postfix({
                    'Loss': '{0:1.3f}'.format(running_loss / (batch_idx + 1)),
                    'Acc_3': '{0:1.3f}'.format(ACC3),
                    'MAE': '{0:1.2f}'.format(MAE),
                    'R2': '{0:1.2f}'.format(R2)
                })
                pbar.update(1)
                total_MAE += MAE
                total_R2 += R2

    average_loss = running_loss / total_samples  # 计算平均损失
    accuracy = CNT / total_samples  # 计算准确率
    mae = total_MAE / len(data_loader)
    r2 = total_R2 / len(data_loader)

    print(f"Val Loss: {average_loss:.4f} Val Acc_3: {accuracy:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    return accuracy

def _train(net, data_loader, optimizer, criterion, device, epoch):
    net.train()  # 设置模型为训练模式
    running_loss = 0.0  # 初始化损失记录
    num_batches = len(data_loader)  # 获取总批次数
    CNT = 0
    total_MAE = 0
    total_R2 = 0
    with tqdm(total=num_batches, desc=f"[EPOCH]<TRAIN>{epoch + 1} / {num_epochs}") as pbar:
        for batch_idx, (X_batch, (y_batch, y_cls)) in enumerate(data_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # 将数据移动到设备上
            optimizer.zero_grad()  # 梯度清零
            O = net(X_batch)  # 前向传播
            loss = criterion(O.squeeze(-1), y_batch)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            running_loss += loss.item()  # 更新损失
            # 使用tqdm.set_postfix()动态更新批次损失
            CNT += _classify(O, y_cls, low, mid, high)
            ACC3 = CNT / ((batch_idx+1)*batch_size)
            MAE = mean_absolute_error(y_batch.cpu().numpy(), O.squeeze(-1).detach().cpu().numpy())
            R2 = r2_score(y_batch.cpu().numpy(), O.squeeze(-1).detach().cpu().numpy())
            pbar.set_postfix({
                'Loss': '{0:1.3f}'.format(running_loss / (batch_idx + 1)),
                'Acc_3': '{0:1.3f}'.format(ACC3),
                'MAE': '{0:1.3f}'.format(MAE),
                'R2': '{0:1.3f}'.format(R2)
            })
            pbar.update(1)
    average_loss = running_loss / num_batches
    acc3 = CNT / num_batches
    total_MAE += total_MAE / len(data_loader)
    total_R2 += total_R2 / len(data_loader)
    print(f"Loss: {average_loss:.4f}, Acc_3: {acc3:.4f}, MAE: {total_MAE:.4f}, R2: {total_R2:.4f}")

weight_dir = r'/Users/drew/Documents/MathModeling/MathModeling/APMCM24_cn/weights'
# 训练和评估循环
best_acc = 0
for epoch in range(num_epochs):
    _train(net, train_loader, optimizer, criterion, device, epoch)
    acc = _evaluate(net, test_loader, criterion, device)  
    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), join(weight_dir, 'model.pth'))
