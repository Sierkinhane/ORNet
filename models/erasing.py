import numpy as np
import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction

class Dropout(InplaceFunction):

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        # See Note [Export inplace]
        r, _ = g.op("Dropout", input, ratio_f=p, is_test_i=not train, outputs=2)
        return r

    @classmethod
    def forward(cls, ctx, input, region_mask, train=False, inplace=False):
        ctx.train = train
        ctx.inplace = inplace
        if not ctx.train:  # ctx.p == 0 or
            return input

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        ctx.noise = cls._make_noise(input)
        b, c, h, w = input.shape
        a_mask = input.view(b, -1)
        region_mask = region_mask.view(b, -1)
        for bs in range(b):
            prob_mask = torch.as_tensor(
                np.random.binomial(1, 1 - a_mask[bs].detach().cpu().numpy().astype(np.float64), h * w),
                dtype=torch.float32).cuda()
            ctx.noise[bs] = ((1 - region_mask[bs]) * prob_mask + region_mask[bs]).view(1, h, w)
        output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:
            return grad_output * ctx.noise, None, None, None
        else:
            return grad_output, None, None, None

"""
Choose pixels at the area of attention randomly
"""


class AttentiveErasing(nn.Module):
    def __init__(self, fill_value=0., p=0.5, min_factor=0., max_factor=0.5, min_height=4, min_width=4, topk=1):
        super(AttentiveErasing, self).__init__()

        self.fill_value = fill_value  # 'zero' 'one' 'uniform'
        self.p = p
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.min_height = min_height
        self.min_width = min_width
        self.topk = topk

    def forward(self, x):

        factor = np.random.uniform(self.min_factor, self.max_factor)
        mask = torch.ones_like(x, requires_grad=False)
        for j in range(x.size(0)):
            if torch.rand(1) < self.p:
                try:
                    max_, center = torch.topk(x[j, 0].flatten(), 1)

                    # cidx = torch.randint(0, maxs_.size(0), (1,))
                    # max_, center = maxs_[cidx], centers[cidx]

                    center = (center // x.size(3), center % x.size(3))

                    min_ = torch.min(x[j, 0])
                    threshold = max_ - (max_ - min_) * factor

                    proposals = torch.nonzero((x[j, 0] > threshold))

                    miny, minx = proposals[:, 0].min(dim=0)[0], proposals[:, 1].min(dim=0)[0]
                    maxy, maxx = proposals[:, 0].max(dim=0)[0], proposals[:, 1].max(dim=0)[0]

                    max_height = maxy - miny
                    max_width = maxx - minx

                    h = torch.randint(self.min_height, max_height // 2, (1,)).cuda()
                    w = torch.randint(self.min_width, max_width // 2, (1,)).cuda()

                    h_start = center[0] - h if center[0] - h >= 0 else 0
                    h_end = center[0] + h if center[0] + h <= x.size(3) else x.size(3)

                    w_start = center[1] - w if center[1] - w >= 0 else 0
                    w_end = center[1] + w if center[1] + w <= x.size(3) else x.size(3)

                    proposals[:, 0] = (proposals[:, 0] > h_start).long() * proposals[:, 0]
                    proposals[:, 0] = (proposals[:, 0] < h_end).long() * proposals[:, 0]

                    proposals[:, 1] = (proposals[:, 1] > w_start).long() * proposals[:, 1]
                    proposals[:, 1] = (proposals[:, 1] < w_end).long() * proposals[:, 1]

                    idx = torch.nonzero(proposals[:, 0] * proposals[:, 1] > 0)
                    new_proposals = proposals[idx, :].squeeze()
                    mask[j, 0][new_proposals[:, 0], new_proposals[:, 1]] = self.fill_value

                except:
                    pass
                else:
                    pass

        x = Dropout.apply(0.6 * x + 0.2, mask, self.training, False)

        return x, mask