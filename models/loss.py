import torch
import torch.nn as nn

"""
Area Loss
"""


class AreaLoss(nn.Module):
    def __init__(self, topk=25):
        super(AreaLoss, self).__init__()
        self.topk = topk
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, p, main_out, features):
        loss = torch.sum(p) / (p.shape[0] * p.shape[2] * p.shape[3])
        if self.topk != 0:
            pred_idx = torch.topk(self.softmax(main_out), self.topk, dim=1)[1]
            for j in range(3, self.topk):
            # for j in range(self.topk):
                feat = features[[k for k in range(p.size(0))], pred_idx[:, j], :, :]
                loss += (torch.sum(feat) / (p.shape[0] * p.shape[2] * p.shape[3]))

        return loss


class AeraLossLossBeta(nn.Module):
    def __init__(self, topk=25):
        super(AeraLossLossBeta, self).__init__()
        self.topk = topk
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, p, main_out, features):
        loss = torch.sum(p) / (p.shape[0] * p.shape[2] * p.shape[3])
        if self.topk != 0:
            pred_idx = torch.topk(self.softmax(main_out), self.topk, dim=1)[1]
            pred_idx = pred_idx[:, 3:]
            pred_idx = pred_idx.unsqueeze(2).unsqueeze(3).repeat(1, 1, features.size(2), features.size(2))
            out = torch.gather(features, dim=1, index=pred_idx)
            loss += (torch.sum(out) / (p.shape[0] * p.shape[2] * p.shape[3] * (self.topk - 3)))

        return torch.abs(loss)


"""
Weighted Entropy Loss
"""


class WeightedEntropyLoss(nn.Module):
    def __init__(self, miu=0.5, sigma=0.1, beta=0.):
        super(WeightedEntropyLoss, self).__init__()

        self.miu = miu
        self.sigma = sigma
        self.eps = torch.finfo(torch.float32).eps
        self.beta = beta

    def _gaussian(self, p):
        return torch.exp(-(p - self.miu) ** 2 / (2 * self.sigma ** 2)) + self.beta

    def forward(self, p):
        return - torch.sum(
            (p * torch.log(p + self.eps) + (1 - p) * torch.log(1 - p + self.eps)) * self._gaussian(p)) / \
               (p.shape[0] * p.shape[2] * p.shape[3])
