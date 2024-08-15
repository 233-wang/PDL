# 导入所需的库
import csv
import numpy as np
import os
import shutil

def To_int(lis):  # 数据清洗函数，将列表中的字符串转换为整数
    data = []
    # 根据列表长度处理数据
    if len(lis) == 8:
        # 正常情况，直接转换
        for row in lis:
            row = int(float(row))
            data.append(row)
        return data
    elif len(lis) == 5:
        # 缺失3个数据，补充为'-1'
        for j in range(3):
            lis.append('-1')
    elif len(lis) == 2:
        # 缺失6个数据，补充为'-1'
        for j in range(6):
            lis.append('-1')
    else:
        # 其他情况，缺失的数据补充为'255'
        for j in range(8 - len(lis)):
            lis.append('255')
    # 转换数据
    for row in lis:
        row = int(float(row))
        data.append(row)
    return data

def write_csv(filepath, way, row): # 写CSV文件的函数
    if filepath is None:
        filepath = "preprocess_well_origin.csv"
    with open(filepath, way, encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

class Graph():
    def __init__(self, num_of_nodes, N=50, directed=True):
        # 图的初始化
        self.num_of_nodes = num_of_nodes  # 节点数
        self.directed = directed  # 是否为有向图
        self.edge_matrix = np.zeros((N, N))  # 邻接矩阵
        self.weight_matrix = np.zeros((N, N))  # 权重矩阵
        self.adjacency_list = {node: set() for node in range(num_of_nodes)}  # 邻接表

    def add_node(self):
        # 添加节点
        self.num_of_nodes += 1

    def add_edge(self, node1, node2, weight):
        # 添加边
        if self.edge_matrix[node1][node2]:  # 如果两节点已连接
            self.weight_matrix[node1][node2] += weight
        else:  # 如果两节点未连接
            self.edge_matrix[node1][node2] = 1
            self.weight_matrix[node1][node2] = weight

    def record(self):  # 记录图的属性：边数、最大度、节点数
        rec = []
        rec.append(np.sum(self.edge_matrix))
        rec.append(np.max(self.weight_matrix))
        rec.append(self.num_of_nodes)
        return rec

# 数据集文件路径
filepath = '../../../Dataset/Car Hacking Dataset/normal_16_id.csv'
path = '../../Dataset50_40/train/nodes/'
batch_size = 40
nnodes = 50

# 读取CSV文件，生成数据集
csvreader = csv.reader(open(filepath, encoding='utf-8'))
dataset = []
labelset = []
line = []
labeline = []
i = 0
for i, row in enumerate(csvreader):
    if i % nnodes == 0 and i != 0:
        dataset.append(line)
        line = []
    line.append(row[1])  # 只保留ID以创建图

# 创建字典、节点数据集和属性数据集
dic_search = {'': 0}
node_dataset = []
att_dataset = []
buchong = []
step = 0
j = 0
for row in dataset:  # 生成时间相关图并提取图属性
    i = 0
    graph = Graph(0, nnodes)
    dic_search.clear()
    for now in row:
        if i == 0:
            i = 1
            last = now
            continue
        if last not in dic_search.keys():
            dic_search[last] = len(dic_search)
            graph.add_node()
        if now not in dic_search.keys():
            dic_search[now] = len(dic_search)
            graph.add_node()
        graph.add_edge(dic_search[now], dic_search[last], 1)
        last = now
    buchong.append(graph.record())  # 记录图属性
    step += 1

# 初始化数据集
normalset = []
attackset = []
attack_num = 0
normal_num = 0
count_attack = 0
count_normal = 0

# 创建路径
normal_path = path + '0/'
if not os.path.exists(normal_path):
    os.makedirs(normal_path)

# 再次迭代，记录点的ID和有效载荷值
tt = []
csvreader = csv.reader(open(filepath, encoding='utf-8'))
for step, row in enumerate(csvreader):
    if (step + 1) > (len(buchong) * nnodes):
        break
    tt = row[3:]
    while '' in tt:
        tt.remove('')
    tt = To_int(tt)  # 添加有效载荷
    tt.insert(0, int(row[1], 16))  # 添加ID
    tt.extend(buchong[int(step / nnodes)])  # 添加图属性
    tt.append(0)
    normalset.append(tt)
    if (step + 1) % nnodes == 0:
        count_normal += 1
    if count_normal == batch_size:
        write_path = normal_path + str(normal_num) + '.csv'
        for rr in normalset:
            write_csv(write_path, 'at', rr)
        count_normal = 0
        normalset = []
        normal_num += 1

# 打印加载过程信息
print('load over {} ,num_attack= {},num_normal={},total num = {}'.format(filepath, attack_num, normal_num, len(buchong) / batch_size))

# 移除正常数据
tag = 0
path = '../../Dataset50_40/train/nodes/'
orignpath = path + str(tag) + '/'
files = os.listdir(orignpath)
num_png = len(files)
lenval = int(num_png * 0.8)
print('location of the split data:{}, {}'.format(lenval, num_png))

v_goal = path[:-12] + "val/nodes/" + str(tag) + "/"
if not os.path.exists(v_goal):
    os.makedirs(v_goal)

for i in range(lenval, num_png):
    shutil.move(path + str(tag) + "/" + str(i) + ".csv", v_goal)

vfiles = os.listdir(v_goal)
num_pngv = len(vfiles)
print('|val set|', num_pngv)
