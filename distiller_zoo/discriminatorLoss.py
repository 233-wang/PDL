import torch
import torch.nn as nn
import torch.nn.functional as F

class betweenLoss(nn.Module):
    def __init__(self, gamma=[1,1,1,1,1,1], loss=nn.L1Loss()):
        super(betweenLoss, self).__init__()
        self.gamma = gamma
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)

        length = len(outputs)
        
        res = sum([self.gamma[i]*self.loss(outputs[i], targets[i]) for i in range(length)])

        return res

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()

# def create_discriminator_criterion(args):
#     d = discriminator.Discriminator(outputs_size=1000, K=8).cuda()
#     d = torch.nn.DataParallel(d)
#     update_parameters = {'params': d.parameters(), "lr": args.d_lr}
#     discriminators_criterion = discriminatorLoss(d).cuda()
#     if len(args.gpus) > 1:
#         discriminators_criterion = torch.nn.DataParallel(discriminators_criterion, device_ids=args.gpus)
#     return discriminators_criterion, update_parameters
##pandora
def create_discriminator_criterion(args, num_classes):
    d = discriminator.Discriminator(outputs_size=1000, K=8).cuda()
    d = torch.nn.DataParallel(d)
    update_parameters = {'params': d.parameters(), "lr": args.d_lr}
    discriminators_criterion = discriminatorLoss(d, num_classes=num_classes).cuda()
    if len(args.gpus) > 1:
        discriminators_criterion = torch.nn.DataParallel(discriminators_criterion, device_ids=args.gpus)
    return discriminators_criterion, update_parameters
##

class discriminatorLoss(nn.Module):
    def __init__(self, models,num_classes, loss=nn.BCEWithLogitsLoss()):
        super(discriminatorLoss, self).__init__()  # 调用父类的 __init__ 方法
        self.models = models
        self.loss = loss
        
        ##pandora
        self.num_classes = num_classes  # 添加 num_classes 属性
        ##

    def forward(self, outputs, targets):
        ##pandora 
        # 确保 outputs 和 targets 是列表
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        if isinstance(targets, torch.Tensor):
            targets = [targets]

        # 确保 outputs 和 targets 形状一致
        inputs = [torch.cat((i, j), dim=1) for i, j in zip(outputs, targets)]
        inputs = torch.cat(inputs, dim=0)

        # print(f"Inputs shape before reshaping: {inputs.shape}")
        
        # 确保输入的维度为4D，假设channels = 3
        batch_size = inputs.size(0)
        channels = 3  # 确定通道数为3
        spatial_dim = inputs.size(1) // channels
        height = width = int(spatial_dim ** 0.5)

        if channels * height * width != inputs.size(1):
            raise ValueError(f"Cannot reshape inputs of shape {inputs.shape} to [batch_size, channels, height, width].")

        inputs = inputs.view(batch_size, channels, height, width)
# 
        # print(f"Inputs shape after reshaping: {inputs.shape}")

        # 通过模型前向传播
        try:
            output = self.models(inputs)
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            raise

        # 确保目标张量与输出大小一致
        target_size = output.size(1)
        target = torch.FloatTensor([[1] + [0]*(target_size-1) if i < batch_size // 2 else [0, 1] + [0]*(target_size-2) for i in range(batch_size)])
        target = target.to(output.device)

        # print(f"Output shape: {output.shape}")
        # print(f"Target shape: {target.shape}")

        ##
        # inputs = [torch.cat((i,j),0) for i, j in zip(outputs, targets)]
        # inputs = torch.cat(inputs, 1)
        # batch_size = inputs.size(0)
        # target = torch.FloatTensor([[1, 0] for _ in range(batch_size//2)] + [[0, 1] for _ in range(batch_size//2)])
        # target = target.to(inputs[0].device)
        #output = self.models(inputs)
        res = self.loss(output, target)
        return res
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class discriminatorFakeLoss(nn.Module):
    def forward(self, outputs, targets):
        res = (0*outputs[0]).sum()
        return res





