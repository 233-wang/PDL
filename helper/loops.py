from __future__ import print_function, division

import sys
import time
import torch
import torch.nn as nn 
import numpy as np
from sklearn.preprocessing import label_binarize
from .util import AverageMeter, accuracy
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix
##pandora
from models.utils_FKD import Recover_soft_label



def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """
    进行标准的训练过程。

    参数:
    epoch (int): 当前的训练轮次。
    train_loader (DataLoader): 用于训练的数据加载器。
    model (torch.nn.Module): 要训练的神经网络模型。
    criterion (torch.nn.Module): 用于计算损失的函数。
    optimizer (torch.optim.Optimizer): 用于优化模型的优化器。
    opt (object): 包含训练配置的对象。
    """
    # 将模型设置为训练模式
    model.train()
    criterion.train()

    inference_times = AverageMeter()  # 用于记录单次推理时间
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_targets = []
    all_predictions = []
    pred_scores = []
    all_pred_probs = []    
    # 计算类权重
    classes = np.unique(train_loader.dataset.classes)
    class_weights = compute_class_weight('balanced', classes=classes, y=train_loader.dataset.classes)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    if torch.cuda.is_available():
        class_weights = class_weights.cuda()
    # 创建带权重的损失函数
    test_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        # 打印输入数据的形状
        print(f"Batch input shape: {inputs.shape}")

        start_inference_time = time.time()
        output = model(input)
        inference_time = time.time() - start_inference_time
        inference_times.update(inference_time)
        loss = test_criterion(output, target)
        probabilities = softmax(output, dim=1)
        if torch.cuda.is_available():
                probabilities = probabilities.cpu()  # 将概率移至CPU
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        preds = output.argmax(dim=1)
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())
        
        pred_scores.extend(probabilities.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    f1 = f1_score(all_targets, all_predictions, average='macro')
    try:
        pred_scores = np.array(pred_scores)
        auc = roc_auc_score(all_targets, pred_scores, multi_class='ovo')
    except ValueError as e:
        print("AUC calculation error:", e)
        auc = None

    print(f" * F1 Score: {f1:.3f}")
    if auc is not None:
        print(f" * AUC Score: {auc:.3f}")
    print(f"Average Inference Time per Batch: {inference_times.avg:.5f} sec")
    return top1.avg, losses.avg



def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation
    """
    # set modules as train()
    all_targets = []
    all_pred_probs = []  
    all_predictions = []   
    pred_scores = []
       
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]
    inference_times = AverageMeter() 
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        # 查看data的shape
        # print("data.shape:",len(data))  #len(data)=3，data为3
        # sys.exit()
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        start_inference_time = time.time()
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        inference_time = time.time() - start_inference_time
        inference_times.update(inference_time)        
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]
        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        elif opt.distill == 'discriminatorLoss':
            outputs = model_s(input, is_feat=True, before=True)
            features, output = outputs  # 解包特征和最终输出

           # Assuming `soft_label` and `soft_no_softmax` can be obtained from `output`
            soft_label = torch.softmax(output, dim=-1)
            soft_no_softmax = output

            # Flatten the tensors to ensure they have the same dimensions
            output = output.view(output.size(0), -1)
            soft_no_softmax = soft_no_softmax.view(soft_no_softmax.size(0), -1)

            g_loss_output = criterion_cls(output, target)  # Reusing criterion_cls as g_loss
            d_loss_value = criterion_kd([output], [soft_no_softmax])
            if isinstance(g_loss_output, tuple):
                g_loss_value, output = g_loss_output
            else:
                g_loss_value = g_loss_output
            loss_kd = g_loss_value + d_loss_value
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        probabilities = softmax(logit_s, dim=1)
        if torch.cuda.is_available():
                probabilities = probabilities.cpu()  # 将概率移至CPU      
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
  
        
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        # # 这里需要修改
        # num_classes = 5
        # all_targets.extend(label_binarize(target.detach().cpu().numpy(), classes=range(num_classes)))
        # all_pred_probs.extend(torch.softmax(logit_s, dim=1).detach().cpu().numpy())
        preds = logit_s.argmax(dim=1)
        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())
        pred_scores.extend(probabilities.detach().numpy())
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
    f1 = f1_score(all_targets, all_predictions, average='macro')
    try:
        pred_scores = np.array(pred_scores)
        auc = roc_auc_score(all_targets, pred_scores, multi_class='ovo')
    except ValueError as e:
        print("AUC calculation error:", e)
        auc = None

    print(f" * F1 Score: {f1:.3f}")
    if auc is not None:
        print(f" * AUC Score: {auc:.3f}")
    print(f"Average Inference Time per Batch: {inference_times.avg:.5f} sec")
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Prepare to collect targets and predictions for F1 and AUC
    all_targets = []
    all_probabilities = []  # Collect probabilities instead of indices

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # Softmax to convert output logits to probabilities
            probabilities = torch.softmax(output, dim=1)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # Append data for metrics calculation
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        # Concatenate all probabilities for AUC computation
        all_probabilities = np.vstack(all_probabilities)

        # Calculate F1 and AUC
        f1 = f1_score(all_targets, np.argmax(all_probabilities, axis=1), average='macro')
        auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovo', average='macro')

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} F1 Score: {f1:.3f} AUC: {auc:.3f}'
              .format(top1=top1, top5=top5, f1=f1, auc=auc))
    return top1.avg, top5.avg, losses.avg,f1,auc
