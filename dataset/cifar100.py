from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from torch.utils.data import Dataset

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder(datset):
    """
    return server-dependent path to store the data
    """
    # hostname = socket.gethostname()
    # if hostname.startswith('visiongpu'):
    #     data_folder = '/data/vision/phillipi/rep-learn/datasets'
    # elif hostname.startswith('yonglong-home'):
    #     data_folder = '/home/yonglong/Data/data'
    # else:
    #     data_folder = './data/'
    if datset == 'car':
        data_folder = './data/car_hacking/train_224'
    elif datset == 'CICIOV2024':
        data_folder = './data/CICIOV2024/train'
    elif datset == 'ROAD':
        data_folder = './data/ROAD/train'        
    else:
        data_folder  = './data/'       

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    
    return data_folder
def get_test_data_folder(datset):
    """
    return server-dependent path to store the data
    """
    # hostname = socket.gethostname()
    # if hostname.startswith('visiongpu'):
    #     data_folder = '/data/vision/phillipi/rep-learn/datasets'
    # elif hostname.startswith('yonglong-home'):
    #     data_folder = '/home/yonglong/Data/data'
    # else:
    #     data_folder = './data/'
    if datset == 'car':
        data_folder = './data/car_hacking/test_224'
    elif datset == 'CICIOV2024':
        data_folder = './data/CICIOV2024/test'
    elif datset == 'ROAD':
        data_folder = './data/ROAD/test'        
    else:
        data_folder = './data/'        

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    
    return data_folder

class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_cifar100_dataloaders_d(batch_size=128, num_workers=8, is_instance=False,dataset = 'car'):
    """
    cifar 100
    """
    train_data_folder = get_data_folder(dataset)
    test_data_folder = get_test_data_folder(dataset)
     # 调试信息：检查数据路径
    print(f"Train data folder: {train_data_folder}")
    print(f"Test data folder: {test_data_folder}")
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        # train_set = CIFAR100Instance(root=data_folder,
        #                              download=False,
        #                              train=True,
        #                              transform=train_transform)
        train_set  = d_CustomDataset(root_dir=train_data_folder, transform=train_transform, is_sample=False)
        n_data = len(train_set)
    else:
        # train_set = datasets.CIFAR100(root=data_folder,
        #                               download=False,
        #                               train=True,
        #                               transform=train_transform)
        train_set  = d_CustomDataset(root_dir=train_data_folder, transform=train_transform, is_sample=False)
         # 调试信息：检查训练集长度
    print(f"Train set length: {len(train_set)}")
    if len(train_set) == 0:
        raise ValueError("Train set is empty")
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    # test_set = datasets.CIFAR100(root=data_folder,
    #                              download=False,
    #                              train=False,
    #                              transform=test_transform)
    test_set  = CustomDataset(root_dir=test_data_folder, transform=test_transform, is_sample=False)
    # 调试信息：检查测试集长度
    print(f"Test set length: {len(test_set)}")
    if len(test_set) == 0:
        raise ValueError("Test set is empty")
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


def get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=False,dataset = 'car'):
    """
    cifar 100
    """
    train_data_folder = get_data_folder(dataset)
    test_data_folder = get_test_data_folder(dataset)
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        # train_set = CIFAR100Instance(root=data_folder,
        #                              download=False,
        #                              train=True,
        #                              transform=train_transform)
        train_set  = CustomDataset(root_dir=train_data_folder, transform=train_transform, is_sample=False)
        n_data = len(train_set)
    else:
        # train_set = datasets.CIFAR100(root=data_folder,
        #                               download=False,
        #                               train=True,
        #                               transform=train_transform)
        train_set  = CustomDataset(root_dir=train_data_folder, transform=train_transform, is_sample=False)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    # test_set = datasets.CIFAR100(root=data_folder,
    #                              download=False,
    #                              train=False,
    #                              transform=test_transform)
    test_set  = CustomDataset(root_dir=test_data_folder, transform=test_transform, is_sample=False)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader
class d_CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_sample=False):
        """
        Args:
            root_dir (string): 数据集目录路径。
            transform (callable, optional): 应用于样本的可选变换。
            is_sample (bool): 是否进行样本采样（测试集一般不进行采样）。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_sample = is_sample
        

        # 自动检测类别数
        self.classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 遍历数据集目录，收集所有图像的路径和相应的标签
        self.images = []
        self.labels = []
        for cls_name in self.classes:
            label_folder = os.path.join(self.root_dir, cls_name)
            for img_file in os.listdir(label_folder):
                if img_file.endswith('.png') or img_file.endswith('.jpg'):
                    self.images.append(os.path.join(label_folder, img_file))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')  
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label,index
    
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_sample=False):
        """
        Args:
            root_dir (string): 数据集目录路径。
            transform (callable, optional): 应用于样本的可选变换。
            is_sample (bool): 是否进行样本采样（测试集一般不进行采样）。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_sample = is_sample
          
        # 自动检测类别数
        self.classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 遍历数据集目录，收集所有图像的路径和相应的标签
        self.images = []
        self.labels = []
        for cls_name in self.classes:
            label_folder = os.path.join(self.root_dir, cls_name)
            for img_file in os.listdir(label_folder):
                if img_file.endswith('.png') or img_file.endswith('.jpg'):
                    self.images.append(os.path.join(label_folder, img_file))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')  
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

class CustomDatasetWithSampling(Dataset):
    def __init__(self, root_dir, k=4096, mode='exact', is_sample=True, percent=1.0, transform=None):
        """
        Args:
            root_dir (string): 数据集目录路径。
            k (int): 负样本的采样数。
            mode (string): 正样本采样模式 ('exact'或'relax')。
            is_sample (bool): 是否进行样本采样。
            percent (float): 负样本采样百分比。
            transform (callable, optional): 应用于样本的可选变换。
        """
        self.root_dir = root_dir
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        self.transform = transform

        # 自动检测类别数
        self.classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        num_classes = len(self.classes)

        # 遍历数据集目录，收集所有图像的路径和相应的标签
        self.images = []
        self.labels = []
        for cls_name in self.classes:
            label_folder = os.path.join(self.root_dir, cls_name)
            # 调试信息：检查类别文件夹
            print(f"Processing folder: {label_folder}")
            for img_file in os.listdir(label_folder):
                  # 调试信息：检查图像文件
                print(f"Found file: {img_file}")
                if img_file.endswith('.png') or img_file.endswith('.jpg'):
                    self.images.append(os.path.join(label_folder, img_file))
                    self.labels.append(self.class_to_idx[cls_name])
                     # 调试信息：检查加载的图像数量
        print(f"Loaded {len(self.images)} images from {self.root_dir}") 
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.root_dir}")

        # 创建正样本和负样本的索引
        self.cls_positive = [[] for _ in range(num_classes)]
        self.cls_negative = [[] for _ in range(num_classes)]
        for idx, label in enumerate(self.labels):
            self.cls_positive[label].append(idx)
            for other_label in range(num_classes):
                if other_label != label:
                    self.cls_negative[label].append(idx)

        # 转换为numpy数组，以便于采样
        self.cls_positive = [np.array(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.array(self.cls_negative[i]) for i in range(num_classes)]

        # 调整负样本的数量
        if 0 < percent < 1:
            for i in range(num_classes):
                n = int(len(self.cls_negative[i]) * percent)
                self.cls_negative[i] = np.random.permutation(self.cls_negative[i])[:n]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')  # 转换为单通道图像
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        if not self.is_sample:
            return image, label, index

        # 根据mode选择正样本索引
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[label], 1)[0]
        else:
            raise NotImplementedError("Sampling mode not implemented.")

        # 选择负样本索引
        replace = self.k > len(self.cls_negative[label])
        neg_idx = np.random.choice(self.cls_negative[label], self.k, replace=replace)
        sample_idx = np.hstack((np.array([pos_idx]), neg_idx))

        return image, label, index, sample_idx

class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset:
    初始化时的输入参数:
    root: 数据集的存储路径。
    train: 指定是训练集还是测试集。
    transform: 应用于图像的转换函数。
    target_transform: 应用于标签的转换函数。
    download: 是否下载数据集。
    k: 在采样负样本时使用的参数。
    mode: 采样模式（'exact'或'relax'）。
    is_sample: 是否进行样本采样。
    percent: 负样本采样的百分比。
    输出：
    如果is_sample为False，输出包括：
    img: 转换后的图像。
    target: 对应的标签。
    index: 传入的样本索引。
    如果is_sample为True，输出还包括：
    sample_idx: 根据mode参数计算的，包含正样本和负样本索引的数组。
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n] for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)  # 将图片从数组转换为PIL图像

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            return img, target, index  # 直接返回
        else:
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)

            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar100_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0,dataset = 'car'):
    """
    cifar 100:
    输出：
    train_loader: 用于在模型训练过程中以批次的形式提供训练数据。
    test_loader: 用于在模型测试或评估阶段以批次的形式提供测试数据。
    n_data:这是一个整数值,表示训练集中的总数据数量。
    """
    train_data_folder = get_data_folder(dataset)
    test_data_folder = get_test_data_folder(dataset)
    # 用于下面的transform参数，这里暂时不使用传统增强技术
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    # # 原：CIFAR100InstanceSample —> 需要修改成自己的版本
    # train_set = CIFAR100InstanceSample(root=data_folder,
    #                                    download=True,
    #                                    train=True,
    #                                    transform=train_transform,
    #                                    k=k,
    #                                    mode=mode,
    #                                    is_sample=is_sample,
    #                                    percent=percent)
    
    # 创建自己数据集的训练集实例
    train_set = CustomDatasetWithSampling(root_dir=train_data_folder,
                                        k=k,
                                        mode=mode,
                                        is_sample=is_sample,
                                        percent=percent,
                                        transform=train_transform)    
    n_data = len(train_set)
    # 查看n_data情况
    # print("n_data",n_data)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    # test_set = datasets.CIFAR100(root=data_folder,
    #                              download=True,
    #                              train=False,
    #                              transform=test_transform)
    # 使用自己的test数据集
    test_set = CustomDataset(root_dir=test_data_folder, transform=test_transform, is_sample=False)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data
