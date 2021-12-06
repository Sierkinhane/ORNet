import torch.nn as nn
from torchvision import models


class Evaluator(nn.Module):
    def __init__(self, num_classes=200):
        super(Evaluator, self).__init__()

        # create an evaluator
        self.model = models.vgg16(pretrained=True)
        if num_classes != 1000:
            self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)
