from __future__ import print_function
# 这行代码确保在 Python 2.x 版本中使用 Python 3.x 的 print 函数特性。

import torch
import torch.nn as nn
# 导入 PyTorch 库，一个流行的深度学习框架。torch.nn 提供了神经网络相关的类和方法。

class ABLoss(nn.Module):
    """Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
    code: https://github.com/bhheo/AB_distillation
    """
    # ABLoss 类继承自 PyTorch 的 nn.Module，是一个用于激活边界蒸馏的自定义损失函数。

    def __init__(self, feat_num, margin=1.0):
        super(ABLoss, self).__init__()
        # 初始化 nn.Module 的基类。feat_num 是特征数量，margin 是边界的边距值，默认为 1.0。

        self.w = [2**(i-feat_num+1) for i in range(feat_num)]
        # 初始化权重 self.w，这是一个列表，根据特征数量动态生成。

        self.margin = margin
        # 设置 margin 属性。

    def forward(self, g_s, g_t):
        # forward 方法定义了损失函数的计算方式。g_s 和 g_t 分别是学生模型和教师模型的激活。

        bsz = g_s[0].shape[0]
        # 计算批次大小（batch size），取 g_s 中第一个元素的第一个维度。

        losses = [self.criterion_alternative_l2(s, t) for s, t in zip(g_s, g_t)]
        # 计算每对学生和教师模型激活之间的损失。

        losses = [w * l for w, l in zip(self.w, losses)]
        # 将损失与对应的权重相乘。

        # 下面两行对损失进行规范化处理。
        losses = [l / bsz for l in losses]
        # 将每个损失除以批次大小，进行规范化。

        losses = [l / 1000 * 3 for l in losses]
        # 将每个损失除以 1000 并乘以 3，进一步规范化。

        return losses
        # 返回最终计算得到的损失列表。

    def criterion_alternative_l2(self, source, target):
        # 定义一个辅助方法 criterion_alternative_l2 来计算 L2 损失的一个变种。

        loss = ((source + self.margin) ** 2 * ((source > -self.margin) & (target <= 0)).float() +
                (source - self.margin) ** 2 * ((source <= self.margin) & (target > 0)).float())
        # 计算源和目标之间的损失，考虑到 margin。

        return torch.abs(loss).sum()
        # 返回损失的绝对值之和。
