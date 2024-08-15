# 导入所需的库
import torch
import math
import time
import argparse
import glob
import os
import random
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
from random import sample
from utils import encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor

# 设置节点数量和批处理大小
nnodes = 200  # 节点数量
batch_size = 40  # 批处理大小

# 定义加载边的函数
def load_edges(path='../Dataset/train/edges/0/', adjes=[], tag='train'):
    tag = path[11:-9]  # 从路径中提取标签
    val_count = [99, 100, 118, 171, 187]  # 验证集文件计数
    test_count = [0, 127, 150, 219, 239, 0, 0, 0, 0]  # 测试集文件计数
    nnodes = 200  # 节点数量
    files = os.listdir(path)  # 读取文件夹内的文件列表
    while ('.ipynb_checkpoints' in files):
        files.remove('.ipynb_checkpoints')
    num_png = len(files)  # 文件数量

    # 读取边的数据
    for i in range(num_png):
        if tag == 'val':
            edges = np.genfromtxt("{}{}.csv".format(path, str(val_count[int(path[-2])] + i)), delimiter=',', dtype=np.int32)
        elif tag == 'test':
            edges = np.genfromtxt("{}{}.csv".format(path, str(test_count[int(path[-2])] + i)), delimiter=',', dtype=np.int32)
        else:
            edges = np.genfromtxt("{}{}.csv".format(path, str(i)), delimiter=',', dtype=np.int32)
        
        # 构建邻接矩阵
        adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])), shape=(8000, 8000), dtype=np.float32)
        
        # 对邻接矩阵进行对称化和归一化处理
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adjes.append(adj)

    print('already load {} edges:'.format(tag), i)
    return adjes

# 定义加载节点的函数
def load_nodes(path='../Dataset/train_nodes.csv'):
    print('Loading {} dataset of Novel nodes...'.format(path[13:-4]))
    idx_features_labels = np.genfromtxt(path, delimiter=',', dtype=np.dtype(np.float32))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    nsamples = len(labels)
    print('nsamples : ', len(labels))  # 样本数量
    nnodes = 200
    batch_size = 40
    nbatches = nsamples / (batch_size * nnodes)

    print("Number of samples: %d" % nsamples)
    print("Number of batches: %d" % nbatches)

    # 构建mini-batches列表
    batches = []
    for i in range(int(nbatches)):
        batches.append([features[i * batch_size * nnodes: (i + 1) * batch_size * nnodes, :], labels[i * batch_size * nnodes: (i + 1) * batch_size * nnodes]])

    # 处理特征、标签和批次数据为tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    for batch in batches:
        batch[0] = torch.FloatTensor(np.array(batch[0].todense()))
        batch[1] = torch.LongTensor(np.where(batch[1])[1])
    batches = batches[1:]
    return batches

print('Start loading')

# 加载训练集边数据
edge_path = '../Dataset/train/edges/'
train_adjes = []
for j in range(5):
    train_adjes = load_edges(path=edge_path + str(j) + '/', adjes=train_adjes)
print('len(train_adjes)', len(train_adjes))

# 加载验证集边数据
edge_path = '../Dataset/val/edges/'
val_adjes = []
for j in range(5):
    val_adjes = load_edges(path=edge_path + str(j) + '/', adjes=val_adjes)
print('len(val_adjes)', len(val_adjes))

# 加载训练集节点数据
train_batches = load_nodes(path="../Dataset/train_nodes1.csv")
print('len(train_batches)', len(train_batches))

# 加载验证集节点数据
val_batches = load_nodes(path="../Dataset/val_nodes1.csv")
print('len(val_batches)', len(val_batches))

# 模型设置
def accuracy(output, labels):
    preds = output
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class GraphConvolution(Module):
    """
    简单的GCN层，类似于 https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    # 重置参数
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # 前向传播函数
        support = torch.mm(input, self.weight)  # 矩阵相乘 XW
        output = torch.spmm(adj, support)  # 稀疏矩阵相乘 AXW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        # 打印对象的描述
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# 定义NN类，实现神经网络模型
class NN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(NN, self).__init__()
        # 定义多个图卷积层和一个全连接层
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # 前向传播过程
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# 加载数据和模型设置
model = NN(nfeat=12, nhid=32, nclass=5, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
print(model)

# 定义训练函数
def train(epoch):
    model.train()
    t = time.time()
    x = list(range(len(train_batches)))
    random.shuffle(x)
    for j in x:
        batch = train_batches[j]
        adj = train_adjes[j]
        optimizer.zero_grad()
        output = model.forward(batch[0], adj)
        loss_train = F.nll_loss(output, batch[1])
        acc_train = accuracy(output.max(1)[1], batch[1])
        loss_train.backward()
        optimizer.step()

    # 打印训练结果
    print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_train: {:.4f}'.format(acc_train.item()), 'time_train:{}'.format(time.time() - t))

    # 验证过程
    labels, preds = [], []
    for j in range(len(val_batches)):
        batch = val_batches[j]
        adj = val_adjes[j]
        output = model.forward(batch[0], adj)
        loss_val = F.nll_loss(output, batch[1])
        labels.append(batch[1])
        preds.append(output.max(1)[1])

    labels = torch.stack(labels, dim=0)
    preds = torch.stack(preds, dim=0)
    labels = torch.reshape(labels, (-1,))
    preds = torch.reshape(preds, (-1,))
    acc_val = accuracy(preds, labels)
    recall_val = recall_score(labels, preds, average='macro')
    precision_val = precision_score(labels, preds, average='macro')
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)

    # 打印验证结果
    print('loss_val:{:.4f}'.format(loss_val.item()), 'acc_val: {:.4f}'.format(acc_val.item()), 'recall_val: {:.4f}'.format(recall_val.item()), "precision = {:.4f}".format(precision_val.item()), "f1 = {:.4f}".format(f1_val.item()), "val_time = {}".format(time.time() - t))

    return acc_val.item()

# 训练流程
acc_values = []
save_values = []  # 记录最好的模型
bad_counter = 0
best = 0
for i in range(10):
    t_total = time.time()
    for epoch in range(20):
        acc_values.append(train(epoch))
        if best == 0 or acc_values[-1] > best:
            best = acc_values[-1]
            torch.save(model.state_dict(), 'gcn0_32_first.pkl')
            save_values.append(epoch + i * 100)
        else:
            bad_counter += 1
        if bad_counter == 100:
            break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    model.load_state_dict(torch.load('gcn0_32_12.pkl'))
print(save_values)


