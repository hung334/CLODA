import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50 as resnet50
from net import resnet50_v2_CBAM as resnet50_CBAM
from icecream import ic







# CBAM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CLIMS_CBAM(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(CLIMS_CBAM, self).__init__()

        self.resnet101 = resnet50.resnet101(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet101.conv1, self.resnet101.bn1, self.resnet101.relu,
                                    self.resnet101.maxpool, self.resnet101.layer1)

        self.stage2 = nn.Sequential(self.resnet101.layer2)
        self.stage3 = nn.Sequential(self.resnet101.layer3)
        self.stage4 = nn.Sequential(self.resnet101.layer4)
        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])
        
        self.ca1 = ChannelAttention(in_planes=256)
        self.sa1 = SpatialAttention()
        
        self.ca2 = ChannelAttention(in_planes=512)
        self.sa2 = SpatialAttention()
        
        self.ca3 = ChannelAttention(in_planes=1024)#2048
        self.sa3 = SpatialAttention()
        
        #self.ca4 = ChannelAttention(in_planes=2048)
        #self.sa4 = SpatialAttention()
        
        print('Net have resnet101 CBAM')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # w-ood
    def feature_list(self, x):
        out_list = []
        x = self.stage1(x)
        x = self.stage2(x) # .detach()
        out_list.append(x)

        x = self.stage3(x)
        out_list.append(x)
        x = self.stage4(x)
        x = torchutils.gap2d(x, keepdims=True)
        out_list.append(x)

        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        return x, out_list
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):

        x = self.stage1(x)
        #x = self.ca1(x) * x
        #x = self.sa1(x) * x
        x = self.stage2(x)
        #x = self.ca2(x) * x
        #x = self.sa2(x) * x
        x = self.stage3(x)
        #x = self.ca3(x) * x
        #x = self.sa3(x) * x
        x = self.stage4(x)
        #x = self.ca4(x) * x
        #x = self.sa4(x) * x
        #ic(x.shape)
        #x = self.ca(x) * x
        #x = self.sa(x) * x
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)   
        cam = cam[0] + cam[1].flip(-1)
        

        
        x = self.classifier(x)
        return x,cam#torch.sigmoid(x),cam
    

    def train(self, mode=True):
        super(CLIMS_CBAM, self).train(mode)
        for p in self.resnet101.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet101.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        #return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
        #return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()))
        #return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(nn.ModuleList([self.classifier,self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1]).parameters()))
        
        return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()),
                list(nn.ModuleList([self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1]).parameters()))
                #list(nn.ModuleList([self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1]).parameters()))

class CAM_CBAM(CLIMS_CBAM):

    def __init__(self, stride=16, n_classes=20):
        super(CAM_CBAM, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x, separate=False):
        x = self.stage1(x)
        #x = self.ca1(x) * x
        #x = self.sa1(x) * x
        x = self.stage2(x)
        #x = self.ca2(x) * x
        #x = self.sa2(x) * x
        x = self.stage3(x)
        #x = self.ca3(x) * x
        #x = self.sa3(x) * x
        x = self.stage4(x)
        #x = self.ca4(x) * x
        #x = self.sa4(x) * x
        #x = self.ca(x) * x
        #x = self.sa(x) * x
                
        x = F.conv2d(x, self.classifier.weight)
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#w-ood
class CLIMS_wood(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(CLIMS_wood, self).__init__()

        self.resnet101 = resnet50.resnet101(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet101.conv1, self.resnet101.bn1, self.resnet101.relu,
                                    self.resnet101.maxpool, self.resnet101.layer1)

        self.stage2 = nn.Sequential(self.resnet101.layer2)
        self.stage3 = nn.Sequential(self.resnet101.layer3)
        self.stage4 = nn.Sequential(self.resnet101.layer4)
        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # w-ood
    def feature_list(self, x):
        out_list = []
        x = self.stage1(x)
        x = self.stage2(x) # .detach()
        out_list.append(x)

        x = self.stage3(x)
        out_list.append(x)
        x = self.stage4(x)
        x = torchutils.gap2d(x, keepdims=True)
        out_list.append(x)

        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        return x, out_list
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def wood(self, x):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)
        
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)   
        cam = cam[0] + cam[1].flip(-1)
        
        x = torchutils.gap2d(x, keepdims=True)
        
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)
        return x,cam#torch.sigmoid(x),cam
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)
    
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)   
        cam = cam[0] + cam[1].flip(-1)
        
        x = self.classifier(x)
        return x,cam#torch.sigmoid(x),cam
    

    def train(self, mode=True):
        super(CLIMS_wood, self).train(mode)
        for p in self.resnet101.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet101.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        #return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
        return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()))

class CAM_wood(CLIMS_wood):

    def __init__(self, stride=16, n_classes=20):
        super(CAM_wood, self).__init__(stride=stride, n_classes=n_classes)

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

    def forward1(self, x, weight, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, weight)

        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def forward2(self, x, weight, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, weight * self.classifier.weight)

        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)
        return x



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



