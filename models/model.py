import os
import sys

import torch.nn.functional as F

sys.path.append(os.getcwd())
from models.unet import UNet
from models.evaluator import Evaluator
from models.erasing import *
from torchvision import models



class CoarseModel(nn.Module):
    def __init__(self, c_base, num_classes=200):
        super(CoarseModel, self).__init__()

        self.att_layer4 = UNet(64, num_classes, c_base=c_base, activation='leakyrelu')
        # self.att_layer4 = UNet(64, num_classes, c_base=c_base, activation='relu')

        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0,
                                   bias=False)
        self.att_conv3 = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1,
                                   bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.att_gap = nn.AvgPool2d(56)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        model = models.resnet50(pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = model.avgpool

        if num_classes == 200:
            self.fc = nn.Linear(2048, num_classes)
        else:
            model = models.resnet50(pretrained=False)
            self.layer4 = model.layer4
            self.fc = model.fc

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        sx = self.att_layer4(x)
        att = torch.sigmoid(self.bn_att3(self.att_conv3(sx)))

        # att branch
        sx = self.att_conv2(sx)
        sx = self.att_gap(sx)
        sx = sx.view(sx.size(0), -1)

        # main branch
        mx = x * att
        mx = self.layer1(mx)
        mx = self.layer2(mx)
        mx = self.layer3(mx)
        mx = self.layer4(mx)
        mx = self.avgpool(mx)
        mx = mx.view(mx.size(0), -1)
        mx = self.fc(mx)

        return sx, mx, att

class FineModel(nn.Module):
    def __init__(self, c_base, att_checkpoint, cls_checkpoint, max_factor=0.5, num_classes=200):
        super(FineModel, self).__init__()

        model = CoarseModel(c_base, num_classes=num_classes)

        for p in model.parameters():
            p.requires_grad = False
        for p in model.att_layer4.up_2.parameters():
            p.requires_grad = True
        for p in model.att_layer4.up_3.parameters():
            p.requires_grad = True
        for p in model.att_conv3.parameters():
            p.requires_grad = True
        for p in model.bn_att3.parameters():
            p.requires_grad = True

        # initialization from the coarse model
        model.load_state_dict(
            torch.load(att_checkpoint, map_location='cpu')['state_dict'])

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.att_layer4 = model.att_layer4
        self.att_conv3 = model.att_conv3
        self.bn_att3 = model.bn_att3

        self.erasing = AttentiveErasing(max_factor=max_factor)

        model = Evaluator(num_classes=num_classes)
        if cls_checkpoint != '':
            model.load_state_dict(torch.load(cls_checkpoint, map_location='cpu')['state_dict'])
        for p in model.parameters():
            p.requires_grad = False

        self.vgg = model

    def forward(self, x, inference=False):

        sx = self.conv1(x)
        sx = self.bn1(sx)
        sx = self.relu(sx)
        sx = self.maxpool(sx)

        sx = self.att_layer4(sx)
        att = torch.sigmoid(self.bn_att3(self.att_conv3(sx)))

        if self.training and bool(self.erasing):
            att_dropout, _ = self.erasing(att)
        else:
            att_dropout = att

        # evaluation
        if inference:
            bx = x
        else:
            bx = x * F.interpolate(att_dropout, size=(x.size(3), x.size(3)))
        bx = self.vgg(bx)

        return bx, att, att_dropout, sx
