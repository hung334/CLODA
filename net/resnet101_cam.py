import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50


# resnet 101
#~~~~~~~~~~~~~~~~~~~~~~~~~
class Net(nn.Module):

    def __init__(self, n_classes=20):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.resnet101 = resnet50.resnet101(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet101.conv1, self.resnet101.bn1, self.resnet101.relu, self.resnet101.maxpool,
                                    self.resnet101.layer1)
        self.stage2 = nn.Sequential(self.resnet101.layer2)
        self.stage3 = nn.Sequential(self.resnet101.layer3)
        self.stage4 = nn.Sequential(self.resnet101.layer4)

        self.classifier = nn.Conv2d(2048, self.n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])


    def forward(self, x, return_feature=False):

        x = self.stage1(x)
        x = self.stage2(x) # .detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        feat = x
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)
        if return_feature:
            return x, feat
        else:
            return x

    def train(self, mode=True):
        for p in self.resnet101.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet101.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
#~~~~~~~~~~~~~~~~~~~~~~~~~


class CAM(Net):

    def __init__(self, n_classes=20):
        super(CAM, self).__init__(n_classes=n_classes)

    def forward(self, x, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, self.classifier.weight)
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x