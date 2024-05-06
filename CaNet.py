import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt

import cv2

idim = 144


patch_dropout1 = 0.1
patch_dropout2 = 0.1

MLP_dropout = 0.2 #0.2

attention_dropout1 = 0.1
attention_dropout2 = 0.1 #0.1


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            pass
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net


    


def DoubleUnet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = DoubleUnet(in_channels=1, n_cls=2, n_filters=64)
    return init_net(net, init_type, init_gain, gpu_id)

def DoubleUnet2_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = DoubleUnet2(in_channels=1, n_cls=2, n_filters=16)
    return init_net(net, init_type, init_gain, gpu_id)

def CaUnet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = CaUnet(in_channels=1, n_cls=2, n_filters=64)
    return init_net(net, init_type, init_gain, gpu_id)
def CaSUnet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = CaSUnet(in_channels=1, n_cls=1, n_filters=16)
    return init_net(net, init_type, init_gain, gpu_id)

def DTransUnet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = DTransUnet(in_channels=1, n_cls=2, n_filters=16)
    return init_net(net, init_type, init_gain, gpu_id)

def MHCAUnet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = MHCAUnet(in_channels=1 , n_cls=2, n_filters=64)
    return init_net(net, init_type, init_gain, gpu_id)
def TransformerUnet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = TransformerUnet(in_channels=2, n_cls=2, n_filters=64)
    return init_net(net, init_type, init_gain, gpu_id)


#双Unet 聚合 Ynet
class DoubleUnet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(DoubleUnet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        #modal 1
        #encoder
        self.block1_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block1_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block1_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block1_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block1_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block1_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        #modal 2
        self.block2_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block2_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block2_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block2_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block2_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block2_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_4 = nn.ConvTranspose2d(16 * n_filters * 2 , 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block2_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)


        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

        #pool
        self.pool = nn.AdaptiveAvgPool2d(output_size=(257,1024))

    def forward(self, x):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x1 = torch.zeros(x.shape[0],1,idim,idim).to(device)
        x2 = torch.zeros(x.shape[0],1,idim,idim).to(device)

        for i in range(x.shape[0]):
            x1[i,0] = x[i,0]
            x2[i,0] = x[i,1]

        #modal 1 encoder
        xds0 = self.block1_1_2_left(self.block1_1_1_left(x1))
        xds1 = self.block1_2_2_left(self.block1_2_1_left(self.pool1_1(xds0)))
        xds2 = self.block1_3_2_left(self.block1_3_1_left(self.pool1_2(xds1)))
        xds3 = self.block1_4_2_left(self.block1_4_1_left(self.pool1_3(xds2)))
        x = self.block1_5_2_left(self.block1_5_1_left(self.pool1_3(xds3)))

        #modal 2 encoder
        yds0 = self.block2_1_2_left(self.block2_1_1_left(x2))
        yds1 = self.block2_2_2_left(self.block2_2_1_left(self.pool2_1(yds0)))
        yds2 = self.block2_3_2_left(self.block2_3_1_left(self.pool2_2(yds1)))
        yds3 = self.block2_4_2_left(self.block2_4_1_left(self.pool2_3(yds2)))
        y = self.block2_5_2_left(self.block2_5_1_left(self.pool2_3(yds3)))

        #pool fuse
        z = torch.cat([x, y], dim=1)

        #modal 2 decoder
        y = self.block2_4_2_right(self.block2_4_1_right(torch.cat([self.upconv2_4(z), yds3], 1)))
        y = self.block2_3_2_right(self.block2_3_1_right(torch.cat([self.upconv2_3(y), yds2], 1)))
        y = self.block2_2_2_right(self.block2_2_1_right(torch.cat([self.upconv2_2(y), yds1], 1)))
        y = self.block2_1_2_right(self.block2_1_1_right(torch.cat([self.upconv2_1(y), yds0], 1)))

        x = self.conv1x1(y)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)
#CT梯度  
class DoubleUnet2(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(DoubleUnet2, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        #modal 1
        #encoder
        self.block1_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block1_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block1_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block1_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block1_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block1_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        #modal 2
        self.block2_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block2_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block2_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block2_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block2_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block2_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_4 = nn.ConvTranspose2d(16 * n_filters * 2 , 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block2_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)


        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

        #pool
        self.pool = nn.AdaptiveAvgPool2d(output_size=(257,1024))

    def forward(self, x):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x1 = torch.zeros(x.shape[0],1,idim,idim).to(device)
        x2 = torch.zeros(x.shape[0],1,idim,idim).to(device)

        for i in range(x.shape[0]):
            x1[i,0] = x[i,0]
            x2[i,0] = x[i,1]

        #modal 1 encoder
        xds0 = self.block1_1_2_left(self.block1_1_1_left(x1))
        xds1 = self.block1_2_2_left(self.block1_2_1_left(self.pool1_1(xds0)))
        xds2 = self.block1_3_2_left(self.block1_3_1_left(self.pool1_2(xds1)))
        xds3 = self.block1_4_2_left(self.block1_4_1_left(self.pool1_3(xds2)))
        x = self.block1_5_2_left(self.block1_5_1_left(self.pool1_3(xds3)))

        #modal 2 encoder
        yds0 = self.block2_1_2_left(self.block2_1_1_left(x2))
        yds1 = self.block2_2_2_left(self.block2_2_1_left(self.pool2_1(yds0)))
        yds2 = self.block2_3_2_left(self.block2_3_1_left(self.pool2_2(yds1)))
        yds3 = self.block2_4_2_left(self.block2_4_1_left(self.pool2_3(yds2)))
        y = self.block2_5_2_left(self.block2_5_1_left(self.pool2_3(yds3)))

        #pool fuse
        z = torch.cat([x, y], dim=1)

        #modal 2 decoder
        y = self.block2_4_2_right(self.block2_4_1_right(torch.cat([self.upconv2_4(z), xds3], 1)))
        y = self.block2_3_2_right(self.block2_3_1_right(torch.cat([self.upconv2_3(y), xds2], 1)))
        y = self.block2_2_2_right(self.block2_2_1_right(torch.cat([self.upconv2_2(y), xds1], 1)))
        y = self.block2_1_2_right(self.block2_1_1_right(torch.cat([self.upconv2_1(y), xds0], 1)))

        x = self.conv1x1(y)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)
#unet 基线
class Unet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_4 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        ds3 = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))
        x = self.block_5_2_left(self.block_5_1_left(self.pool_3(ds3)))

        x = self.block_4_2_right(self.block_4_1_right(torch.cat([self.upconv_4(x), ds3], 1)))
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        # x = self.conv1x1(x)
        return x
        # if self.n_cls == 1:
        #     return torch.sigmoid(x)
        # else:
        #     return F.softmax(x, dim=1)


class TransformerUnet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(TransformerUnet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_4 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

        self.vit_img_dim = 16
        self.vit = ViT(img_dim=self.vit_img_dim, in_channels=1024, embedding_dim=1024,
                head_num = 8, mlp_dim = 2048, block_num = 1, patch_dim=1, classification=False)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        ds3 = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))
        x = self.block_5_2_left(self.block_5_1_left(self.pool_3(ds3)))
        x = self.vit(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim) 
        x = self.block_4_2_right(self.block_4_1_right(torch.cat([self.upconv_4(x), ds3], 1)))
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)
#只有注意力
class MHCAUnet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(MHCAUnet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        #modal 1
        self.block1_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block1_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block1_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block1_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block1_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block1_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)



        #modal 2
        self.block2_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block2_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block2_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block2_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block2_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block2_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_4 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block2_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)


        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

        self.vit_img_dim = 16
        self.vit = ViT3(img_dim=self.vit_img_dim, in_channels=1024, embedding_dim=1024,
                head_num = 8, mlp_dim = 64, block_num = 1, patch_dim=1, classification=False)

    def forward(self, x):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x1 = torch.zeros(x.shape[0],1,idim,idim).to(device)
        x2 = torch.zeros(x.shape[0],1,idim,idim).to(device)

        for i in range(x.shape[0]):
            x1[i,0] = x[i,0]
            x2[i,0] = x[i,1]

        #modal 1 encoder
        xds0 = self.block1_1_2_left(self.block1_1_1_left(x1))
        xds1 = self.block1_2_2_left(self.block1_2_1_left(self.pool1_1(xds0)))
        xds2 = self.block1_3_2_left(self.block1_3_1_left(self.pool1_2(xds1)))
        xds3 = self.block1_4_2_left(self.block1_4_1_left(self.pool1_3(xds2)))
        x = self.block1_5_2_left(self.block1_5_1_left(self.pool1_3(xds3)))

        #modal 2 encoder
        yds0 = self.block2_1_2_left(self.block2_1_1_left(x2))
        yds1 = self.block2_2_2_left(self.block2_2_1_left(self.pool2_1(yds0)))
        yds2 = self.block2_3_2_left(self.block2_3_1_left(self.pool2_2(yds1)))
        yds3 = self.block2_4_2_left(self.block2_4_1_left(self.pool2_3(yds2)))
        y = self.block2_5_2_left(self.block2_5_1_left(self.pool2_3(yds3)))

        #transformer
        y = self.vit(x,y)
        y = rearrange(y, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # b*1024*16*16

        #modal 2 decoder
        y = self.block2_4_2_right(self.block2_4_1_right(torch.cat([self.upconv2_4(y), yds3], 1)))
        y = self.block2_3_2_right(self.block2_3_1_right(torch.cat([self.upconv2_3(y), yds2], 1)))
        y = self.block2_2_2_right(self.block2_2_1_right(torch.cat([self.upconv2_2(y), yds1], 1)))
        y = self.block2_1_2_right(self.block2_1_1_right(torch.cat([self.upconv2_1(y), yds0], 1)))

        x = self.conv1x1(y)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)

#提出的方法，cross Attention Unet
class CaUnet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(CaUnet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        #modal 1
        #encoder
        self.block1_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block1_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block1_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block1_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block1_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block1_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        #decoder
        self.upconv1_4 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block1_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv1_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block1_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv1_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block1_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv1_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block1_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        #modal 2
        self.block2_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block2_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block2_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block2_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block2_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block2_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_4 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block2_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)


        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

        self.vit_img_dim = 16
        self.vit = ViT2(img_dim=self.vit_img_dim, in_channels=1024, embedding_dim=1024,
                head_num = 4, mlp_dim = 1024, block_num = 4, patch_dim=1, classification=False)

    def forward(self, x):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x1 = torch.zeros(x.shape[0],1,idim,idim).to(device)
        x2 = torch.zeros(x.shape[0],1,idim,idim).to(device)

        for i in range(x.shape[0]):
            x1[i,0] = x[i,0]
            x2[i,0] = x[i,1]

        #modal 1 encoder
        xds0 = self.block1_1_2_left(self.block1_1_1_left(x1))
        xds1 = self.block1_2_2_left(self.block1_2_1_left(self.pool1_1(xds0)))
        xds2 = self.block1_3_2_left(self.block1_3_1_left(self.pool1_2(xds1)))
        xds3 = self.block1_4_2_left(self.block1_4_1_left(self.pool1_3(xds2)))
        x = self.block1_5_2_left(self.block1_5_1_left(self.pool1_3(xds3)))

        #modal 2 encoder
        yds0 = self.block2_1_2_left(self.block2_1_1_left(x2))
        yds1 = self.block2_2_2_left(self.block2_2_1_left(self.pool2_1(yds0)))
        yds2 = self.block2_3_2_left(self.block2_3_1_left(self.pool2_2(yds1)))
        yds3 = self.block2_4_2_left(self.block2_4_1_left(self.pool2_3(yds2)))
        y = self.block2_5_2_left(self.block2_5_1_left(self.pool2_3(yds3)))

        #transformer
        x,y = self.vit(x,y)

        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # b*1024*16*16
        y = rearrange(y, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # b*1024*16*16

        #modal 1 decoder
        x = self.block1_4_2_right(self.block1_4_1_right(torch.cat([self.upconv1_4(x), xds3], 1)))
        x = self.block1_3_2_right(self.block1_3_1_right(torch.cat([self.upconv1_3(x), xds2], 1)))
        x = self.block1_2_2_right(self.block1_2_1_right(torch.cat([self.upconv1_2(x), xds1], 1)))
        x = self.block1_1_2_right(self.block1_1_1_right(torch.cat([self.upconv1_1(x), xds0], 1)))

        #modal 2 decoder
        y = self.block2_4_2_right(self.block2_4_1_right(torch.cat([self.upconv2_4(y), yds3], 1)))
        y = self.block2_3_2_right(self.block2_3_1_right(torch.cat([self.upconv2_3(y), yds2], 1)))
        y = self.block2_2_2_right(self.block2_2_1_right(torch.cat([self.upconv2_2(y), yds1], 1)))
        y = self.block2_1_2_right(self.block2_1_1_right(torch.cat([self.upconv2_1(y), yds0], 1)))

        x1 = torch.cat((x,y),dim=1)
        x = self.conv1x1(x1)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)

class CaSUnet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(CaSUnet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        #modal 1 encoder
        self.block1_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block1_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block1_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block1_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block1_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block1_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        #modal 2 encoder
        self.block2_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block2_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block2_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block2_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block2_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block2_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        #unet decoder
        self.upconv2_4 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block2_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)


        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

        # self.vit_img_dim = 16
        # self.vit = ViT4(img_dim=self.vit_img_dim, in_channels=16 * n_filters, embedding_dim=512,
        #         head_num = 16, mlp_dim = 3072, block_num = 2, patch_dim=1, classification=False)
        self.vit_img_dim = 9
        self.vit = ViT4(img_dim=self.vit_img_dim, in_channels=16 * n_filters, embedding_dim=16 * n_filters,
                head_num = 4, mlp_dim = 256, block_num = 2, patch_dim=1, classification=False)
        # self.vit_img_dim = 9
        # self.vit = ViT4(img_dim=self.vit_img_dim, in_channels=16 * n_filters, embedding_dim=16 * n_filters,
        #         head_num = 2, mlp_dim = 144, block_num = 4, patch_dim=1, classification=False)

    def forward(self, x):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x1 = torch.zeros(x.shape[0],1,idim,idim).to(device)
        x2 = torch.zeros(x.shape[0],1,idim,idim).to(device)

        for i in range(x.shape[0]):
            x1[i,0] = x[i,0]
            x2[i,0] = x[i,1]

        # x1 = x[:,0,:,:]
        # x2 = x[:,1,:,:]

        #modal 1 encoder
        xds0 = self.block1_1_2_left(self.block1_1_1_left(x1))
        xds1 = self.block1_2_2_left(self.block1_2_1_left(self.pool1_1(xds0)))
        xds2 = self.block1_3_2_left(self.block1_3_1_left(self.pool1_2(xds1)))
        xds3 = self.block1_4_2_left(self.block1_4_1_left(self.pool1_3(xds2)))
        x = self.block1_5_2_left(self.block1_5_1_left(self.pool1_4(xds3)))

        #modal 2 encoder
        yds0 = self.block2_1_2_left(self.block2_1_1_left(x2))
        yds1 = self.block2_2_2_left(self.block2_2_1_left(self.pool2_1(yds0)))
        yds2 = self.block2_3_2_left(self.block2_3_1_left(self.pool2_2(yds1)))
        yds3 = self.block2_4_2_left(self.block2_4_1_left(self.pool2_3(yds2)))
        y = self.block2_5_2_left(self.block2_5_1_left(self.pool2_4(yds3)))

        # print(yds0.shape,yds1.shape,yds2.shape,yds3.shape,y.shape)

        #transformer
        y = self.vit(y,x)

        # plt.subplot(1,5,1)
        # plt.imshow(x1.detach().cpu().numpy()[0,0])
        # plt.subplot(1,5,2)
        # plt.imshow(x2.detach().cpu().numpy()[0,0])


        y = rearrange(y, "b (x y) c -> b c x y", x=self.vit_img_dim , y=self.vit_img_dim)  # b*1024*16*16

        # plt.subplot(1,5,3)
        # plt.imshow(yo.detach().cpu().numpy()[0,10])
        # plt.subplot(1,5,4)
        # plt.imshow(x.detach().cpu().numpy()[0,10])
        # plt.subplot(1,5,5)
        # plt.imshow(y.detach().cpu().numpy()[0,10])
        # plt.show()

        #modal 2 decoder
        y = self.block2_4_2_right(self.block2_4_1_right(torch.cat([self.upconv2_4(y), yds3], 1)))
        y = self.block2_3_2_right(self.block2_3_1_right(torch.cat([self.upconv2_3(y), yds2], 1)))
        y = self.block2_2_2_right(self.block2_2_1_right(torch.cat([self.upconv2_2(y), yds1], 1)))
        y = self.block2_1_2_right(self.block2_1_1_right(torch.cat([self.upconv2_1(y), yds0], 1)))

        x = self.conv1x1(y)

        # plt.subplot(1,5,3)
        # plt.imshow(x.detach().cpu().numpy()[0,0])
        # plt.show()

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


#双TransUnet 聚合
class DTransUnet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(DTransUnet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        #modal 1
        #encoder
        self.block1_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block1_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block1_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block1_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block1_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool1_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block1_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block1_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        #modal 2
        self.block2_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block2_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block2_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block2_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block2_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block2_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_4 = nn.ConvTranspose2d(16 * n_filters , 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block2_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv2_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block2_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)


        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

        self.vit_img_dim = 16
        self.vit = ViT5(img_dim=self.vit_img_dim, in_channels=256, embedding_dim=256,
                head_num = 16, mlp_dim = 3072, block_num = 2, patch_dim=1, classification=False)

    def forward(self, x):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x1 = torch.zeros(x.shape[0],1,idim,idim).to(device)
        x2 = torch.zeros(x.shape[0],1,idim,idim).to(device)

        for i in range(x.shape[0]):
            x1[i,0] = x[i,0]
            x2[i,0] = x[i,1]

        #modal 1 encoder
        xds0 = self.block1_1_2_left(self.block1_1_1_left(x1))
        xds1 = self.block1_2_2_left(self.block1_2_1_left(self.pool1_1(xds0)))
        xds2 = self.block1_3_2_left(self.block1_3_1_left(self.pool1_2(xds1)))
        xds3 = self.block1_4_2_left(self.block1_4_1_left(self.pool1_3(xds2)))
        x = self.block1_5_2_left(self.block1_5_1_left(self.pool1_3(xds3)))

        #modal 2 encoder
        yds0 = self.block2_1_2_left(self.block2_1_1_left(x2))
        yds1 = self.block2_2_2_left(self.block2_2_1_left(self.pool2_1(yds0)))
        yds2 = self.block2_3_2_left(self.block2_3_1_left(self.pool2_2(yds1)))
        yds3 = self.block2_4_2_left(self.block2_4_1_left(self.pool2_3(yds2)))
        y = self.block2_5_2_left(self.block2_5_1_left(self.pool2_3(yds3)))

        #transformer
        y = self.vit(x,y)

        y = rearrange(y, "b (x y) c -> b c x y", x=self.vit_img_dim , y=self.vit_img_dim)  # b*1024*16*16


        #modal 2 decoder
        y = self.block2_4_2_right(self.block2_4_1_right(torch.cat([self.upconv2_4(y), yds3], 1)))
        y = self.block2_3_2_right(self.block2_3_1_right(torch.cat([self.upconv2_3(y), yds2], 1)))
        y = self.block2_2_2_right(self.block2_2_1_right(torch.cat([self.upconv2_2(y), yds1], 1)))
        y = self.block2_1_2_right(self.block2_1_1_right(torch.cat([self.upconv2_1(y), yds0], 1)))

        x = self.conv1x1(y)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)

#no MHCA
class ViT5(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=False, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim  # 1
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2  # 256
        self.token_dim = in_channels * (patch_dim ** 2)  # 1024

        self.projection1 = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024
        self.projection2 = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout1 = nn.Dropout(patch_dropout1)
        self.dropout2 = nn.Dropout(patch_dropout2)

        self.transformer = TransformerEncoder5(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x,y):
        img_patches1 = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)
        img_patches2 = rearrange(y,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size1, tokens1, _ = img_patches1.shape
        batch_size2, tokens2, _ = img_patches2.shape

        project1 = self.projection1(img_patches1)
        project2 = self.projection2(img_patches2)
        token1 = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size1)
        token2 = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size2)

        patches1 = torch.cat([token1, project1], dim=1)
        patches2 = torch.cat([token2, project2], dim=1)
        patches1 += self.embedding[:tokens1 + 1, :]
        patches2 += self.embedding[:tokens2 + 1, :]

        x = self.dropout1(patches1)
        y = self.dropout2(patches2)

        y = self.transformer(x,y)

        y = y[:, 1:, :]

        return y
#2 to 1
class ViT4(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=False, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim  # 1
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2  # 256
        self.token_dim = in_channels * (patch_dim ** 2)  # 1024

        self.projection1 = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024
        self.projection2 = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024

        self.embedding = nn.Parameter(torch.rand(self.num_tokens, embedding_dim)) #position embedding

        self.cls_token1 = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout1 = nn.Dropout(patch_dropout1)
        self.dropout2 = nn.Dropout(patch_dropout2)

        self.transformer = TransformerEncoder2(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x,y):
        # print(x.shape)
        img_patches1 = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)
        img_patches2 = rearrange(y,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)
        # print(img_patches1.shape)

        # batch_size1, tokens1, _ = img_patches1.shape
        # batch_size2, tokens2, _ = img_patches2.shape

        project1 = self.projection1(img_patches1)
        project2 = self.projection2(img_patches2)

        # token1 = repeat(self.cls_token1, 'b ... -> (b batch_size) ...',
        #                batch_size=batch_size1)
        # token2 = repeat(self.cls_token2, 'b ... -> (b batch_size) ...',
        #                batch_size=batch_size2)

        # patches1 = torch.cat([token1, project1], dim=1)
        # patches2 = torch.cat([token2, project2], dim=1)

        patches1 = project1
        patches2 = project2

        # patches1 += self.embedding[:tokens1, :]
        # patches2 += self.embedding[:tokens2, :]
        patches1 += self.embedding
        patches2 += self.embedding

        x = self.dropout1(patches1)
        y = self.dropout2(patches2)


        y = self.transformer(x,y)

        # y = y[:, 1:, :]

        return y

#双流结构的VIT 只有Attention       
class ViT3(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=False, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim  # 1
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2  # 256
        self.token_dim = in_channels * (patch_dim ** 2)  # 1024

        self.projection1 = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024
        self.projection2 = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout1 = nn.Dropout(patch_dropout1)
        self.dropout2 = nn.Dropout(patch_dropout2)

        self.transformer = TransformerEncoder3(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x,y):
        img_patches1 = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)
        img_patches2 = rearrange(y,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size1, tokens1, _ = img_patches1.shape
        batch_size2, tokens2, _ = img_patches2.shape

        project1 = self.projection1(img_patches1)
        project2 = self.projection2(img_patches2)
        token1 = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size1)
        token2 = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size2)

        patches1 = torch.cat([token1, project1], dim=1)
        patches2 = torch.cat([token2, project2], dim=1)
        patches1 += self.embedding[:tokens1 + 1, :]
        patches2 += self.embedding[:tokens2 + 1, :]

        x = self.dropout1(patches1)
        y = self.dropout2(patches2)

        y = self.transformer(x,y)

        y = y[:, 1:, :]

        return y

#双流结构的VIT       
class ViT2(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=False, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim  # 1
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2  # 256
        self.token_dim = in_channels * (patch_dim ** 2)  # 1024

        self.projection1 = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024
        self.projection2 = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout1 = nn.Dropout(patch_dropout1)
        self.dropout2 = nn.Dropout(patch_dropout2)

        self.transformer = TransformerEncoder2(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x,y):
        img_patches1 = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)
        img_patches2 = rearrange(y,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size1, tokens1, _ = img_patches1.shape
        batch_size2, tokens2, _ = img_patches2.shape

        project1 = self.projection1(img_patches1)
        project2 = self.projection2(img_patches2)
        token1 = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size1)
        token2 = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size2)

        patches1 = torch.cat([token1, project1], dim=1)
        patches2 = torch.cat([token2, project2], dim=1)
        patches1 += self.embedding[:tokens1 + 1, :]
        patches2 += self.embedding[:tokens2 + 1, :]

        x = self.dropout1(patches1)
        y = self.dropout2(patches2)

        x,y = self.transformer(x,y)

        x = x[:, 1:, :]
        y = y[:, 1:, :]

        return x,y

#单流结构的VIT
class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=False, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim  # 1
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2  # 256
        self.token_dim = in_channels * (patch_dim ** 2)  # 1024

        self.projection = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(patch_dropout1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # print(x.shape)
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)
        # print(img_patches.shape)
        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)

        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[:tokens + 1, :]

        x = self.dropout(patches)
        x = self.transformer(x)
        x = x[:, 1:, :]

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()
        print(f"block_num:{block_num}")
        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x

class TransformerEncoder2(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()
        print(f"block_num:{block_num}")
        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock4(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x,y):
        for layer_block in self.layer_blocks:
            y = layer_block(x,y)

        return y
    
class TransformerEncoder3(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()
        print(f"block_num:{block_num}")
        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock3(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x,y):
        for layer_block in self.layer_blocks:
            y = layer_block(x,y)

        return y
  
class TransformerEncoder4(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()
        print(f"block_num:{block_num}")
        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock4(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x,y):
        for layer_block in self.layer_blocks:
            x,y = layer_block(x,y)

        return x,y

class TransformerEncoder5(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()
        print(f"block_num:{block_num}")
        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock5(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x,y):
        for layer_block in self.layer_blocks:
            y = layer_block(x,y)

        return y

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(attention_dropout1)

    def forward(self, x):
        #正确写法
        _x = self.layer_norm1(x)
        _x = self.multi_head_attention(_x)
        _x = self.dropout(_x)
        x = x + _x

        _x = self.layer_norm2(x)
        _x = self.mlp(_x)
        x = x + _x

        return x

class TransformerEncoderBlock2(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_cross_attention = MultiHeadCrossAttention(embedding_dim, head_num)
        self.multi_head_attention1 = MultiHeadAttention(embedding_dim, head_num)
        self.multi_head_attention2 = MultiHeadAttention(embedding_dim, head_num)

        self.mlp1 = MLP(embedding_dim, mlp_dim)
        self.mlp2 = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)


        self.layer_norm4 = nn.LayerNorm(embedding_dim)
        self.layer_norm5 = nn.LayerNorm(embedding_dim)
        self.layer_norm6 = nn.LayerNorm(embedding_dim)

        self.dropout1 = nn.Dropout(attention_dropout1)
        self.dropout2 = nn.Dropout(attention_dropout2)

    def forward(self, x, y):
        #MHSA
        _x = self.layer_norm1(x)
        _y = self.layer_norm4(y)

        _x = self.multi_head_attention1(_x)
        _y = self.multi_head_attention2(_y)

        x = x + _x
        y = y + _y

        #MHCA
        _x = self.layer_norm2(x)
        _y = self.layer_norm5(y)

        _x,_y = self.multi_head_cross_attention(_x,_y)

        _x = self.dropout1(_x)
        _y = self.dropout2(_y)

        x = x + _x
        y = y + _y

        #MLP
        _x = self.layer_norm3(x)
        _y = self.layer_norm6(y)

        _x = self.mlp1(_x)
        _y = self.mlp2(_y)

        x = x + _x
        y = y + _y

        return x,y
    
class TransformerEncoderBlock3(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_cross_attention = MultiHeadCrossAttention2(embedding_dim, head_num)

        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        self.layer_norm4 = nn.LayerNorm(embedding_dim)

        self.dropout2 = nn.Dropout(attention_dropout2)

    def forward(self, x, y):
        #MHCA
        _x = self.layer_norm1(x)
        _y = self.layer_norm2(y)
        _y = self.multi_head_cross_attention(_x,_y)
        _y = self.dropout2(_y)

        y = y + _y

        return y
  
class TransformerEncoderBlock4(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_cross_attention = MultiHeadCrossAttention2(embedding_dim, head_num)

        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        self.layer_norm4 = nn.LayerNorm(embedding_dim)

        self.dropout2 = nn.Dropout(attention_dropout2)

    def forward(self, x, y):

        #MHCA
        _x = self.layer_norm1(x)
        _y = self.layer_norm2(y)
        _y = self.multi_head_cross_attention(_x,_y)
        _y = self.dropout2(_y)
        y = y + _y
        #MLP
        _y = self.layer_norm4(y)
        _y = self.mlp(_y)
        y = y + _y

        return y
    
class TransformerEncoderBlock5(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()


        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm4 = nn.LayerNorm(embedding_dim)

        self.dropout2 = nn.Dropout(attention_dropout2)
        #pool
        self.pool = nn.AdaptiveAvgPool2d(output_size=(257,embedding_dim))

    def forward(self, x, y):
        z = torch.cat([x, y], dim=2)
        y = self.pool(z)
        #MLP
        _y = self.layer_norm4(y)
        _y = self.mlp(_y)
        y = y + _y

        return y

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = math.sqrt(embedding_dim / head_num)

        #CT
        self.qkv_layer1 = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        #PET
        self.qkv_layer2 = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, y,mask=None):
        # print(x.shape)
        qkv1 = self.qkv_layer1(x)
        qkv2 = self.qkv_layer2(y)

        query1, key1, value1 = tuple(rearrange(qkv1, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        query2, key2, value2 = tuple(rearrange(qkv2, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))

        #Cross Attention 交叉注意力机制核心
        energy1 = torch.einsum("... i d , ... j d -> ... i j", query2, key1) / self.dk
        energy2 = torch.einsum("... i d , ... j d -> ... i j", query1, key2) / self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention1 = torch.softmax(energy1, dim=-1)
        attention2 = torch.softmax(energy2, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention1, value1)
        y = torch.einsum("... i j , ... j d -> ... i d", attention2, value2)

        x = rearrange(x, "b h t d -> b t (h d)")
        y = rearrange(y, "b h t d -> b t (h d)")

        x = self.out_attention1(x)
        y = self.out_attention2(y)

        return x,y
    
class MultiHeadCrossAttention2(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = math.sqrt(embedding_dim / head_num)
        # self.dk = (embedding_dim / head_num) ** (-0.5)

        #m1
        self.qkv_layer1 = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        # self.out_attention1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        #m2
        self.qkv_layer2 = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        # self.out_attention2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.out_attention = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)
        #pool
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(81,embedding_dim)) #256
        # self.lin = nn.Linear(embedding_dim * 2 , embedding_dim)

    def forward(self, x, y,mask=None):
        # print(x.shape)
        qkv1 = self.qkv_layer1(x)
        qkv2 = self.qkv_layer2(y)

        query1, key1, value1 = tuple(rearrange(qkv1, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        query2, key2, value2 = tuple(rearrange(qkv2, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))

        #Cross Attention 交叉注意力机制核心
        energy1 = torch.einsum("... i d , ... j d -> ... i j", query2, key1) / self.dk
        energy2 = torch.einsum("... i d , ... j d -> ... i j", query1, key2) / self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention1 = torch.softmax(energy1, dim=-1)
        attention2 = torch.softmax(energy2, dim=-1)

        # plt.subplot(1,2,1)
        # plt.imshow(attention1.detach().cpu().numpy()[0,1])
        # plt.subplot(1,2,2)
        # plt.imshow(attention2.detach().cpu().numpy()[0,1])
        # plt.show()

        x = torch.einsum("... i j , ... j d -> ... i d", attention1, value1)
        y = torch.einsum("... i j , ... j d -> ... i d", attention2, value2)

        #多头拼回去
        x = rearrange(x, "b h t d -> b t (h d)")
        y = rearrange(y, "b h t d -> b t (h d)")

        z = torch.cat([x, y], dim=2)
        
        # xa = torch.sigmoid(x[0,:,:]).detach().cpu().numpy()
        
        # xs = np.ones((81))
        # for i in range(xa.shape[1]):
        #     xs = np.dot(xs,xa[:,i])

        # xs = cv2.resize(x.detach().cpu().numpy()[0,:,0].reshape(9,9),(144, 144), interpolation=cv2.INTER_LINEAR)
        # ys = cv2.resize(np.sum(y.detach().cpu().numpy()[0,:,:],axis=-1).reshape(9,9),(144, 144), interpolation=cv2.INTER_LINEAR)
        # plt.subplot(1,4,3)
        # plt.imshow(xs,cmap='hot')
        # plt.subplot(1,4,4)
        # plt.imshow(ys,cmap='hot')
        # plt.show()

        y = self.out_attention(z)
        # ys = cv2.resize(y.detach().cpu().numpy()[0,:,0].reshape(9,9),(144, 144), interpolation=cv2.INTER_LINEAR)
        # plt.subplot(1,5,4)
        # plt.imshow(xs,cmap='hot')
        # plt.subplot(1,5,5)
        # plt.imshow(ys,cmap='hot')

        #pca insert
        # u,s,v = torch.pca_lowrank(A=z,q=257,center=True,niter=8)
        # d = torch.matmul(z,v)
        # dT = torch.transpose(d, dim0=1, dim1=2)
        # res = torch.matmul(dT,z)
        # return res
        #old
        # x = self.out_attention1(x)
        # y = self.out_attention2(y)
        # z = torch.cat([x, y], dim=2)
        # y = self.pool(z)
        # y = self.lin(z)
        return y
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = math.sqrt(embedding_dim / head_num)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) / self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x

class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(MLP_dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(MLP_dropout)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x
 
