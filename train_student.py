"""
the general training framework
"""

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
import sys

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample,get_cifar100_dataloaders_d

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP
from distiller_zoo.discriminatorLoss import discriminatorLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init



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





def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='car', choices=['cifar100','car','CICIOV2024','ROAD'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2','Hsnet'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst','discriminatorLoss'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]





def print_model_size(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    print(f"Total parameters: {total_params}")
    total_size = total_params * 4 / 1024 / 1024  # Assuming each parameter is a float32
    print(f"Model size: {total_size:.2f} MB")
#  ####Pandora
def create_discriminator_criterion(args, num_classes):
    d = discriminator.Discriminator(outputs_size=1000, K=8).cuda()
    d = torch.nn.DataParallel(d)
    update_parameters = {'params': d.parameters(), "lr": args.d_lr}
    discriminators_criterion = discriminatorLoss(d, num_classes=num_classes).cuda()
    if len(args.gpus) > 1:
        discriminators_criterion = torch.nn.DataParallel(discriminators_criterion, device_ids=args.gpus)
    return discriminators_criterion, update_parameters
# ########   

def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    
    # 加载模型文件并删除额外的键
    state_dict = torch.load(model_path)
    for key in list(state_dict['model'].keys()):
        if "total_ops" in key or "total_params" in key:
            del state_dict['model'][key]
    # 打印模型和状态字典的键
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict['model'].keys())
    
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys
    
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    new_state_dict = model.state_dict()
    for key in model_keys:
        if key in state_dict_keys:
            if new_state_dict[key].shape == state_dict['model'][key].shape:
                new_state_dict[key] = state_dict['model'][key]
            else:
                print(f"Shape mismatch for key {key}: model {new_state_dict[key].shape}, state_dict {state_dict['model'][key].shape}")
        else:
            print(f"Missing key in state_dict: {key}")
    
    model.load_state_dict(state_dict['model'])
    print('==> done')
    return model

def main():
    best_f1 = 0
    best_auc = 0
    opt = parse_option()
    
    
    
    ##Pandora
    # args = parse_args(argv)
    # utils.general_setup(args.save, args.gpus)

    # logging.info("Arguments parsed.\n{}".format(pprint.pformat(vars(args))))

    # Convert batch_size to the true number of loading-images since we use multiple crops from the same image within a minibatch
    # args.batch_size = math.ceil(args.batch_size / args.num_crops)
    ####

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    """ 
    若想修改数据集需要从这里入手:
    1. 理解 get_cifar100_dataloaders_sample以及get_cifar100_dataloaders ；
    2. 从训练入手代码，找到具体的训练数据；
    3. 修改model通道参数，使用自己数据集相同形状的数据代替原数据；
    4. 修改get_cifar100_dataloaders_sample，切换成自己的数据集。
    """
    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode,dataset=opt.dataset)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_d(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True,dataset=opt.dataset)
        n_cls = 100
    elif opt.dataset == 'car':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        k=opt.nce_k,
                                                                        mode=opt.mode,dataset=opt.dataset)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_d(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True,dataset=opt.dataset)     
        n_cls = 5      
    elif opt.dataset == 'CICIOV2024':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        k=opt.nce_k,
                                                                        mode=opt.mode,dataset=opt.dataset)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_d(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True,dataset=opt.dataset) 
        n_cls = 6   
    elif opt.dataset == 'ROAD':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        k=opt.nce_k,
                                                                        mode=opt.mode,dataset=opt.dataset)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_d(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True,dataset=opt.dataset) 
        n_cls = 6               
    else:
        raise NotImplementedError(opt.dataset)
    """
    修改model思路：
    1. 修改model的结构，将原来的三通道转为单通道的形式;
    2. 一般来说，只需修改类别初始化开头关于通道的相关参数即可，后面的不用修改。
    """
    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    print("Student model name:")
    print(opt.model_s)
    print("Teacher model size:")
    print_model_size(model_t)
    print("Student model size:")
    print_model_size(model_s)
    data = torch.randn(32, 3, 32, 32)
    # 计算教师模型的FLOPs和参数
    flops_t, params_t = profile(model_t, inputs=(data, ), verbose=False)
    print(f"Teacher Model FLOPs: {flops_t / 1e9:.2f} GFLOPs")
    print(f"Teacher Model Parameters: {params_t / 1e6:.2f} M")

    # 计算学生模型的FLOPs和参数
    flops_s, params_s = profile(model_s, inputs=(data, ), verbose=False)
    print(f"Student Model FLOPs: {flops_s / 1e9:.2f} GFLOPs")
    print(f"Student Model Parameters: {params_s / 1e6:.2f} M")
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
    elif opt.distill == 'discriminatorLoss':
        ##pandora
        # 确保 `models` 参数正确传递给 `discriminatorLoss`
        models = [model_s, model_t]  # 示例，具体内容取决于 `discriminatorLoss` 的实现
        criterion_kd = discriminatorLoss(models = model_s,num_classes=6)
        ##
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        # print('data',train_loader)
        # sys.exit()
        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        # print ("train_acc",train_acc)
        # sys.exit()
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss,f1,auc = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if f1 > best_f1 or auc > best_auc:
            best_f1 = max(f1, best_f1)
            best_auc = max(auc, best_auc)
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_f1': best_f1,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print(' best_f1:', best_f1)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()