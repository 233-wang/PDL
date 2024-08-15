from __future__ import print_function

import os
import argparse
import socket
import time
from thop import profile

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'Hsnet', ])
    parser.add_argument('--dataset', type=str, default='car', choices=['cifar100','car','CICIOV2024','ROAD'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def print_model_details(model):
    total_params = 0
    total_bytes = 0

    # Dict to map PyTorch data types to the size in bytes
    dtype_to_bytes = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1
    }

    for name, param in model.named_parameters():
        num_params = param.numel()
        param_bytes = num_params * dtype_to_bytes.get(param.dtype, 4)  # Default to 4 bytes if dtype is not mapped
        total_params += num_params
        total_bytes += param_bytes
        # print(f"Parameter: {name}, Type: {param.dtype}, Count: {num_params}, Bytes: {param_bytes}")

    print(f"Total parameters: {total_params}")
    total_size_mb = total_bytes / 1024 / 1024  # Convert bytes to megabytes
    print(f"Total model size: {total_size_mb:.2f} MB")


def main():
    best_f1 = 0
    best_auc = 0
    opt = parse_option()
    # print('opt.dataset',opt.dataset)
    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers,dataset=opt.dataset)
        n_cls = 100
    elif opt.dataset == 'car':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers,dataset=opt.dataset)
        n_cls = 5
    elif opt.dataset == 'CICIOV2024':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers,dataset=opt.dataset)
        n_cls = 6   
    elif opt.dataset == 'ROAD':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers,dataset=opt.dataset)
        n_cls = 6                      
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls)
    print_model_details(model)
        # 计算学生模型的FLOPs和参数
    data = torch.randn(32, 3, 32, 32)
    print("Teacher name:")
    print(opt.model)
    flops_s, params_s = profile(model, inputs=(data, ), verbose=False)
    print(f"Model FLOPs: {flops_s / 1e9:.2f} GFLOPs")
    print(f"Model Parameters: {params_s / 1e6:.2f} M")
    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss,f1,auc = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if f1 > best_f1 or auc > best_auc:
            best_f1 = max(f1, best_f1)
            best_auc = max(auc, best_auc)
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, f"{opt.model}_{opt.dataset}_best.pth")
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'f1': f1,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}_{dataset}.pth'.format(epoch=epoch, dataset=opt.dataset))

            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best f1:', best_f1)
    print('best AUC:', best_auc)
    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, f"{opt.model}_{opt.dataset}_last.pth")
    torch.save(state, save_file)


if __name__ == '__main__':
    main()