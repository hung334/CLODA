import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50_v2 as resnet50
from net import resnet50_v2_CBAM as resnet50_CBAM
from icecream import ic
from net.amm import AMM
from net.GCNet import  GlobalContextBlock


# GCNet
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CLIMS_GCNet(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(CLIMS_GCNet, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                    self.resnet50.maxpool, self.resnet50.layer1)

        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])
        
        
        self.GC1 = GlobalContextBlock(inplanes=256, ratio=0.25)

        self.GC2 = GlobalContextBlock(inplanes=512, ratio=0.25)
        
        self.GC3 = GlobalContextBlock(inplanes=1024, ratio=0.25)
        
        print('Net have GCNet')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):

        x = self.stage1(x)
        x = self.GC1(x)
        x = self.stage2(x)
        x = self.GC2(x)
        x = self.stage3(x)
        x = self.GC3(x)
        x = self.stage4(x)
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)   
        cam = cam[0] + cam[1].flip(-1)
        

        
        x = self.classifier(x)
        return x,cam#torch.sigmoid(x),cam
    

    def train(self, mode=True):
        super(CLIMS_GCNet, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        #return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
        #return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()))
        #return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(nn.ModuleList([self.classifier,self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1]).parameters()))
        
        return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()),
                list(nn.ModuleList([self.GC1,self.GC2,self.GC3]).parameters()))
                #list(nn.ModuleList([self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1]).parameters()))

class CAM_GCNet(CLIMS_GCNet):

    def __init__(self, stride=16, n_classes=20):
        super(CAM_GCNet, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x, separate=False):
        x = self.stage1(x)
        x = self.GC1(x)
        x = self.stage2(x)
        x = self.GC2(x)
        x = self.stage3(x)
        x = self.GC3(x)
        x = self.stage4(x)
                
        x = F.conv2d(x, self.classifier.weight)
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# AMM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CLIMS_AMM(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(CLIMS_AMM, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                    self.resnet50.maxpool, self.resnet50.layer1)

        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])
        
        
        self.amm1 = AMM(gate_channels=256)

        self.amm2 = AMM(gate_channels=512)
        
        self.amm3 = AMM(gate_channels=1024)
        
        print('Net have AMM')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):

        x = self.stage1(x)
        x = self.amm1(x)
        x = self.stage2(x)
        x = self.amm2(x)
        x = self.stage3(x)
        x = self.amm3(x)
        x = self.stage4(x)
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)   
        cam = cam[0] + cam[1].flip(-1)
        

        
        x = self.classifier(x)
        return x,cam#torch.sigmoid(x),cam
    

    def train(self, mode=True):
        super(CLIMS_AMM, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        #return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
        #return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()))
        #return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(nn.ModuleList([self.classifier,self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1]).parameters()))
        
        return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()),
                list(nn.ModuleList([self.amm1,self.amm2,self.amm3]).parameters()))
                #list(nn.ModuleList([self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1]).parameters()))

class CAM_AMM(CLIMS_AMM):

    def __init__(self, stride=16, n_classes=20):
        super(CAM_AMM, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x, separate=False):
        x = self.stage1(x)
        x = self.amm1(x)
        x = self.stage2(x)
        x = self.amm2(x)
        x = self.stage3(x)
        x = self.amm3(x)
        x = self.stage4(x)
                
        x = F.conv2d(x, self.classifier.weight)
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#CBAM_v2

class CLIMS_CBAM_V2(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(CLIMS_CBAM_V2, self).__init__()

        self.resnet50 = resnet50_CBAM.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                    self.resnet50.maxpool, self.resnet50.layer1)

        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)
        
        print('Net have CBAM_V2')
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
        return torch.sigmoid(x),cam
    

    def train(self, mode=True):
        super(CLIMS_CBAM_V2, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False
    
    def trainable_parameters(self):
        
        return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(nn.ModuleList([self.classifier]).parameters()),
                )
class CAM_CBAM_V2(CLIMS_CBAM_V2):

    def __init__(self, stride=16, n_classes=20):
        super(CAM_CBAM_V2, self).__init__(stride=stride, n_classes=n_classes)

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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#saliency
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CLIMS_saliency(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(CLIMS_saliency, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                    self.resnet50.maxpool, self.resnet50.layer1)

        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        #self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)
        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)
        self.classifier_21 = nn.Conv2d(20, 21, 1, bias=False)
        
        self.backbone = nn.ModuleList([self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])
        '''
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 1, 3, stride=2),
            
            nn.AdaptiveAvgPool2d((512, 512)),
            nn.Sigmoid(),
        )
        print('decoder layer')
        '''
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        #sal_map = self.decoder(x)
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)   
        cam = cam[0] + cam[1].flip(-1)
        
        
        x = self.classifier(x)
        if(self.n_classes==21):
            x = self.classifier_21(x)
        return torch.sigmoid(x),cam#,sal_map
    

    def train(self, mode=True):
        super(CLIMS_saliency, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        #return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
        #return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()))
        #return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(nn.ModuleList([self.classifier,self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1]).parameters()))
        
        return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()),
                list(self.newly_added.parameters()),
                list(nn.ModuleList([self.classifier_21]).parameters())#,list(nn.ModuleList([self.decoder]).parameters())
               )





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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Residual CBAM
class CLIMS_RES_CBAM(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(CLIMS_RES_CBAM, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                    self.resnet50.maxpool, self.resnet50.layer1)

        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])
        
        self.ca1 = ChannelAttention(in_planes=256)
        self.sa1 = SpatialAttention()
        self.relu1 = nn.ReLU(inplace=True)
        
        self.ca2 = ChannelAttention(in_planes=512)
        self.sa2 = SpatialAttention()
        self.relu2 = nn.ReLU(inplace=True)
        
        self.ca3 = ChannelAttention(in_planes=1024)#2048
        self.sa3 = SpatialAttention()
        self.relu3 = nn.ReLU(inplace=True)
        
        #self.ca4 = ChannelAttention(in_planes=2048)
        #self.sa4 = SpatialAttention()
        
        print('Net have Residual CBAM')
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
        residual_1 = x
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x += residual_1
        x = self.relu1(x)
        x = self.stage2(x)
        residual_2 = x
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        x += residual_2
        x = self.relu2(x)
        x = self.stage3(x)
        residual_3 = x
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        x += residual_3
        x = self.relu3(x)
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
        super(CLIMS_RES_CBAM, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        #return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
        #return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()))
        #return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(nn.ModuleList([self.classifier,self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1]).parameters()))
        
        return (list(nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()),
                list(nn.ModuleList([self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1,self.relu3,self.relu2,self.relu1]).parameters()))
                #list(nn.ModuleList([self.ca3,self.sa3,self.ca2,self.sa2,self.ca1,self.sa1]).parameters()))

class CAM_RES_CBAM(CLIMS_RES_CBAM):

    def __init__(self, stride=16, n_classes=20):
        super(CAM_RES_CBAM, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x, separate=False):
        x = self.stage1(x)
        residual_1 = x
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x += residual_1
        x = self.relu1(x)
        x = self.stage2(x)
        residual_2 = x
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        x += residual_2
        x = self.relu2(x)
        x = self.stage3(x)
        residual_3 = x
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        x += residual_3
        x = self.relu3(x)
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

    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CLIMS_CBAM(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(CLIMS_CBAM, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                    self.resnet50.maxpool, self.resnet50.layer1)

        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
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
        
        print('Net have CBAM')
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
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x = self.stage2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        x = self.stage3(x)
        x = self.ca3(x) * x
        x = self.sa3(x) * x
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
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
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
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x = self.stage2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        x = self.stage3(x)
        x = self.ca3(x) * x
        x = self.sa3(x) * x
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

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                    self.resnet50.maxpool, self.resnet50.layer1)

        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
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
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
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



class Net(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(Net, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool, self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool, self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        return x

    def train(self, mode=True):
        super(Net, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CLIMS(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(CLIMS, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool, self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool, self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    # -------------------------------------------------------------
    # For DDP & syncBN training, must set 'requires_grad = False' before passing to DDP
    # https://discuss.pytorch.org/t/how-does-distributeddataparallel-handle-parameters-whose-requires-grad-flag-is-false/90736/1
        self._freeze_layers()

    def _freeze_layers(self):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False
    # --------------------------------------------------------------

    def _initialize_weights(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
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
        x = self.stage2(x)

        x = self.stage3(x)
        x = self.stage4(x)

        x = self.classifier(x)
        return torch.sigmoid(x)

    def train(self, mode=True):
        super(CLIMS, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        #return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
        return (list(nn.ModuleList([self.stage3, self.stage4]).parameters()), list(self.newly_added.parameters()))



class Net_CAM(Net):

    def __init__(self, stride=16, n_classes=20):
        super(Net_CAM, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)

        return x, cams, feature


class Net_CAM_Feature(Net):

    def __init__(self, stride=16, n_classes=20):
        super(Net_CAM_Feature, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)  # bs*2048*32*32

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)
        cams = cams / (F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2) * feature.unsqueeze(1)  # bs*20*2048*32*32
        cams_feature = cams_feature.view(cams_feature.size(0), cams_feature.size(1), cams_feature.size(2), -1)
        cams_feature = torch.mean(cams_feature, -1)

        return x, cams_feature, cams


class CAM(CLIMS):

    def __init__(self, stride=16, n_classes=20):
        super(CAM, self).__init__(stride=stride, n_classes=n_classes)

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


class Class_Predictor(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(representation_size, num_classes, 1, bias=False)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_classes, -1)  # bs*20*2048
        mask = label > 0  # bs*20

        feature_list = [x[i][mask[i]] for i in range(batch_size)]  # bs*n*2048
        prediction = [self.classifier(y.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for y in feature_list]
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]

        loss = 0
        acc = 0
        num = 0
        for logit, label in zip(prediction, labels):
            if label.shape[0] == 0:
                continue
            loss_ce = F.cross_entropy(logit, label)
            loss += loss_ce
            acc += (logit.argmax(dim=1) == label.view(-1)).sum().float()
            num += label.size(0)

        return loss / batch_size, acc / num
