import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as S

# from utils import model_info

"""
clear UNet implementation.
"""

BN_MOMEMTUM = 0.01


class UNetConv(nn.Module):
    """
    conv-bn-relu-conv-bn-relu
    """

    def __init__(self, c_in, c_out, activation='relu'):
        super(UNetConv, self).__init__()

        if activation == 'relu':
            self.UConv = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out, momentum=BN_MOMEMTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out, momentum=BN_MOMEMTUM),
                nn.ReLU(inplace=True),
            )
        elif activation == 'leakyrelu':
            self.UConv = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out, momentum=BN_MOMEMTUM),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out, momentum=BN_MOMEMTUM),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):
        return self.UConv(x)


class Up(nn.Module):
    """
    Upscaling then double conv(implemented by https://github.com/milesial/Pytorch-UNet)
    """

    def __init__(self, c_in, c_out, activation='relu'):
        super(Up, self).__init__()

        self.conv = UNetConv(c_in, c_out, activation=activation)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, c_in, n_classes, c_base=4, activation='relu'):
        super(UNet, self).__init__()

        self.down_1 = S(nn.MaxPool2d(2), UNetConv(c_in, c_base * 2, activation=activation), )
        self.down_2 = S(nn.MaxPool2d(2), UNetConv(c_base * 2, c_base * 4, activation=activation), )
        self.down_3 = S(nn.MaxPool2d(2), UNetConv(c_base * 4, c_base * 8, activation=activation), )

        self.up_1 = Up(c_base * 12, c_base * 4, activation=activation)
        self.up_2 = Up(c_base * 6, c_base * 3, activation=activation)
        self.up_3 = Up(c_base * 3 + c_in, n_classes, activation=activation)

    def forward(self, x1):
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        x = self.up_1(x4, x3)
        x = self.up_2(x, x2)
        x = self.up_3(x, x1)

        return x


class UNetOri(nn.Module):
    def __init__(self, c_in, c_base=4, activation='relu'):
        super(UNetOri, self).__init__()

        self.inconv = UNetConv(c_in, c_base)
        self.down_1 = S(nn.MaxPool2d(2), UNetConv(c_base, c_base * 2, activation=activation), )
        self.down_2 = S(nn.MaxPool2d(2), UNetConv(c_base * 2, c_base * 4, activation=activation), )
        self.down_3 = S(nn.MaxPool2d(2), UNetConv(c_base * 4, c_base * 8, activation=activation), )

        self.up_1 = Up(c_base * 12, c_base * 4, activation=activation)
        self.up_2 = Up(c_base * 6, c_base * 3, activation=activation)
        self.up_3 = Up(c_base * 4, c_base, activation=activation)
        self.outconv = UNetConv(c_base, c_in)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        x = self.up_1(x4, x3)
        x = self.up_2(x, x2)
        x = self.up_3(x, x1)
        x = self.outconv(x)

        return x


if __name__ == '__main__':
    model = UNet(64, 200)
    # model_info(model)
    inp = torch.randn(2, 64, 56, 56)
    out = model(inp)
