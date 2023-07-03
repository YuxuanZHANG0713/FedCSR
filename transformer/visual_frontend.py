"""
Author: Lei Liu
Copyright (c) 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
import torch.nn.init as init
from transformer.Modules import Linear

from utils.matrix import frame_label_align

class ResNetLayer(nn.Module):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return


    def forward(self, inputBatch):
        
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch



class ResNet(nn.Module):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        # self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        return


    def forward(self, inputBatch):
        
        batch = self.layer1(inputBatch)
        
        
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        # outputBatch = self.avgpool(batch)
        # outputBatch =  F.adaptive_avg_pool2d(batch, (1,1))
        return batch



class VisualFrontend(nn.Module):

    """
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self):
        super(VisualFrontend, self).__init__()
        # self.resnet = ResNet()
        model = models.resnet18(pretrained=True)
        model.conv1= nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.resnet = torch.nn.Sequential(*(list(model.children())[:-1]))

        self.frontend3D = nn.Sequential(
                            nn.Conv3d(3, 64, kernel_size=(5,3,3), stride=(1,1,1), padding=(2,1,1), bias=False),
                            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
                            nn.ReLU())

       
        
        return


    def forward(self, inputBatch, mask=None):

        batchsize = inputBatch.shape[0]
        framelength = inputBatch.shape[1]
        inputBatch = inputBatch.transpose(1,2)
        outputBatch = self.frontend3D(inputBatch)
        outputBatch = outputBatch.transpose(1, 2)

        outputBatch = outputBatch.reshape(outputBatch.shape[0]*outputBatch.shape[1], outputBatch.shape[2], outputBatch.shape[3], outputBatch.shape[4])
        # outputBatch = self.dropout(outputBatch)
        outputBatch = self.resnet(outputBatch)
        
        # outputBatch = self.avgpool(outputBatch)
        outputBatch = outputBatch.reshape(batchsize, framelength, -1)
       
        if mask != None:
            outputBatch *= mask
        
        return outputBatch


class Lip_Hand_CNN(nn.Module):
    def __init__(self, num_classes=9):
        super(Lip_Hand_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=7,
                stride=1,
                padding=3
            ),  # (1, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3),  # (1, 21, 21)
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 5, 7, 1, 0),  # N = (W − F + 2P )/S+1, W:输入图片大小. (8, 15, 15)
            # nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),  # (8, 5, 5)
            nn.Dropout(0.3)
        )
        self.lin = nn.Sequential(
            nn.Linear(200, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )
        self.out = nn.Linear(64, num_classes)

    def forward(self, x, output_feature=True):
        

        batchsize = x.shape[0]
        framelength = x.shape[1]

        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) #展平多维的卷积图成 (batch_size, 8 * 5 * 5)
        lin = self.lin(x)

        lin = lin.reshape(batchsize, framelength, -1)
        if output_feature == True:
            return lin
        else:
            output = self.out(lin)
            return output


class Hand_ANN(nn.Module):
    def __init__(self, num_classes=6):
        super(Hand_ANN, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.25)
        )
        self.out = nn.Linear(64, num_classes)

    def forward(self, x, output_feature=True):
        

        batchsize = x.shape[0]
        framelength = x.shape[1]

        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])

        lin = self.lin(x)

        lin = lin.reshape(batchsize, framelength, -1)
        if output_feature == True:
            return lin
        else:
            output = self.out(lin)
            return output


class PosFrontend(nn.Module):

    """
    A position feature extraction module. Generates a 512-dim feature vector per video frame.
    
    """

    def __init__(self, d_in, d_out):
        super(PosFrontend, self).__init__()
        
        self.linear1 = Linear(d_in, d_out)
        self.bn1 = torch.nn.BatchNorm1d(d_out, momentum=0.01, eps=0.001)
       
        return


    def forward(self, inputBatch):
        # print(inputBatch.shape)
        
        outputBatch = self.linear1(inputBatch)
        outputBatch = outputBatch.transpose(1,2)
        outputBatch = self.bn1(outputBatch)
        outputBatch = outputBatch.transpose(1,2)
        outputBatch = F.relu(outputBatch)
        # outputBatch = self.linear2(outputBatch)
        
        return outputBatch
