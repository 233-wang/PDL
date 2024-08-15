import torch
from torch import nn
from .memory import ContrastMemory

eps = 1e-7


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        # 假设输入x的形状是[32, 128]
        # bsz - 批次大小，这里是32
        bsz = x.shape[0]

        # m - 每个样本的特征数量减去1，这里是127 (128 - 1)
        m = x.size(1) - 1

        # Pn - 噪声分布的概率，这里假设为均匀分布
        Pn = 1 / float(self.n_data)

        # P_pos - 选择每个样本的第一个特征
        P_pos = x.select(1, 0)
        # log_D1 - 正样本对的对数概率
        # 形状为[32]
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # P_neg - 选择每个样本剩余的127个特征
        P_neg = x.narrow(1, 1, m)
        # log_D0 - 负样本对的对数概率
        # 形状为[32, 127]
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        # 计算最终的损失，负和形式
        # log_D1.sum(0) 是一个标量，log_D0.view(-1, 1).sum(0) 也是一个标量
        # 最终输出是一个标量，表示平均损失
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
