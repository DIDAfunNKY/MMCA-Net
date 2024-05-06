import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat


patch_dropout1 = 0.5
patch_dropout2 = 0.5

MLP_dropout = 0.5

attention_dropout1 = 0.5
attention_dropout2 = 0.5


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
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net

def CaNet_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = CaUnet(img_dim=256,
                    in_channels=1,
                    out_channels=128,  # 128
                    head_num=4,  # 4
                    mlp_dim=64,  # 64
                    block_num=4,  # 1
                    patch_dim=16)  # 16

    return init_net(net, init_type, init_gain, gpu_id)

def CaSNet_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = CaSUnet(img_dim=256,
                    in_channels=1,
                    out_channels=128,  # 128
                    head_num=4,  # 4
                    mlp_dim=64,  # 64
                    block_num=4,  # 1
                    patch_dim=16)  # 16

    return init_net(net, init_type, init_gain, gpu_id)

class CaUnet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(int(out_channels * 2 / 4), 1, kernel_size=1)
        self.encoder = Encoder2(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim )
        self.decoder = Decoder2(out_channels)

        
    def forward(self,x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x1 = torch.zeros(x.shape[0],1,256,256).to(device)
        x2 = torch.zeros(x.shape[0],1,256,256).to(device)

        for i in range(x.shape[0]):
            x1[i,0] = x[i,0]
            x2[i,0] = x[i,1]

        x, x0, x1, x2, x3, y, y0, y1, y2, y3 = self.encoder(x1,x2)
        x1,x2 = self.decoder(x, x0, x1, x2, x3, y, y0, y1, y2, y3)
        x1 = torch.cat((x1,x2),dim=1)



        x = self.conv1(x1)

        # return F.softmax(x, dim=1)

        return torch.sigmoid(x)
    

class CaSUnet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(int(out_channels * 1 / 4), 1, kernel_size=1)
        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim )
        self.decoder = Decoder(out_channels)
        
    def forward(self,x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x1 = torch.zeros(x.shape[0],1,256,256).to(device)
        x2 = torch.zeros(x.shape[0],1,256,256).to(device)

        for i in range(x.shape[0]):
            x1[i,0] = x[i,0]
            x2[i,0] = x[i,1]

        x, x0, x1, x2, x3 = self.encoder(x1,x2)
        x1 = self.decoder(x, x0, x1, x2, x3)


        x = self.conv1(x1)

        # return F.softmax(x, dim=1)

        return torch.sigmoid(x)
    


    
class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, n_cls=2, n_filters=16):
        super().__init__()
        self.in_channels = in_channels

        # TODO============================= 一般卷积 encoding =====================
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)

        self.x0 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.y0 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.norm11 = nn.BatchNorm2d(out_channels)
        self.norm12 = nn.BatchNorm2d(out_channels)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        #CT
        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)  # 16*16
        #PET
        self.encoder4 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder5 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder6 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)  # 16*16



        self.vit_img_dim = img_dim // patch_dim  # 256 // 16 = 16

        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.cat_fuse1 = cat_fuse(2048, 512)
        self.cat_fuse2 = cat_fuse(2048, 512)

    def forward(self, x,y):
        # print(x.shape)

        # TODO  conv ****** transunet *********
        x0 = self.norm11(self.conv1(x))
        x1 = self.relu1(x0)  # b*128*128*128
        x2 = self.encoder1(x1)  # b*256*64*64
        x3 = self.encoder2(x2)  # b*512*32*32
        x_out = self.encoder3(x3)  # b*1024*16*16
        x0 = self.x0(x)  # 自己添加

        # TODO  conv ****** transunet *********
        y0 = self.norm12(self.conv1(y))
        y1 = self.relu2(y0)  # b*128*128*128
        y2 = self.encoder4(y1)  # b*256*64*64
        y3 = self.encoder5(y2)  # b*512*32*32
        y_out = self.encoder6(y3)  # b*1024*16*16
        y0 = self.y0(y)  # 自己添加

        # TODO 以下为共同
        x = self.vit(x_out,y_out) #b*256*1024
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # b*1024*16*16

        x = self.cat_fuse1(x, x_out)  # b*512*16*16

        return x, x0, x1, x2, x3
    

class Encoder2(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, n_cls=2, n_filters=16):
        super().__init__()
        self.in_channels = in_channels

        # TODO============================= 一般卷积 encoding =====================
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)

        self.x0 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.y0 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.norm11 = nn.BatchNorm2d(out_channels)
        self.norm12 = nn.BatchNorm2d(out_channels)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        #CT
        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)  # 16*16
        #PET
        self.encoder4 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder5 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder6 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)  # 16*16



        self.vit_img_dim = img_dim // patch_dim  # 256 // 16 = 16

        self.vit = ViT2(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim = 1, classification=False)

        self.cat_fuse1 = cat_fuse(2048, 512)
        self.cat_fuse2 = cat_fuse(2048, 512)

    def forward(self, x,y):
        # print(x.shape)

        # TODO  conv ****** transunet *********
        x0 = self.norm11(self.conv1(x))
        x1 = self.relu1(x0)  # b*128*128*128
        x2 = self.encoder1(x1)  # b*256*64*64
        x3 = self.encoder2(x2)  # b*512*32*32
        x_out = self.encoder3(x3)  # b*1024*16*16
        x0 = self.x0(x)  # 自己添加

        # TODO  conv ****** transunet *********
        y0 = self.norm12(self.conv1(y))
        y1 = self.relu2(y0)  # b*128*128*128
        y2 = self.encoder4(y1)  # b*256*64*64
        y3 = self.encoder5(y2)  # b*512*32*32
        y_out = self.encoder6(y3)  # b*1024*16*16
        y0 = self.y0(y)  # 自己添加

        # TODO 以下为共同
        x,y = self.vit(x_out,y_out) #b*256*1024
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # b*1024*16*16
        y = rearrange(y, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # b*1024*16*16

        x = self.cat_fuse1(x, x_out)  # b*512*16*16
        y = self.cat_fuse2(y, y_out)  # b*512*16*16

        return x, x0, x1, x2, x3, y, y0, y1, y2, y3
    
class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decoder5 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder6 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder7 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder8 = DecoderBottleneck(out_channels, int(out_channels * 1 / 4))

    def forward(self,y, y0, y1, y2, y3):

        y = self.decoder5(y, y3)
        y = self.decoder6(y, y2)
        y = self.decoder7(y, y1)
        y = self.decoder8(y, y0)

        return y
    

class Decoder2(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(out_channels, int(out_channels * 1 / 4))

        self.decoder5 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder6 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder7 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder8 = DecoderBottleneck(out_channels, int(out_channels * 1 / 4))

    def forward(self, x, x0, x1, x2, x3, y, y0, y1, y2, y3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x, x0)

        y = self.decoder5(y, y3)
        y = self.decoder6(y, y2)
        y = self.decoder7(y, y1)
        y = self.decoder8(y, y0)

        return x,y
    
class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=False, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim  # 1
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2  # 256
        self.token_dim = in_channels * (patch_dim ** 2)  # 1024
        # print(self.token_dim)
        # print(embedding_dim)
        self.projection1 = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024
        self.projection2 = nn.Linear(self.token_dim, embedding_dim)  # 1024, 1024
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

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
        x = self.transformer(x,y)

        x = x[:, 1:, :]

        return x
    
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
        # x = patches1
        # y = patches2

        x,y = self.transformer(x,y)

        x = x[:, 1:, :]
        y = y[:, 1:, :]

        return x,y
    
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()
        print(f"block_num:{block_num}")
        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x,y):
        for layer_block in self.layer_blocks:
            x = layer_block(x,y)

        return x
    
class TransformerEncoder2(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()
        print(f"block_num:{block_num}")
        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock2(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x,y):
        for layer_block in self.layer_blocks:
            x,y = layer_block(x,y)

        return x,y
        
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_cross_attention = MultiHeadCrossAttention(embedding_dim, head_num)
        self.multi_head_attention1 = MultiHeadAttention(embedding_dim, head_num)
        self.multi_head_attention2 = MultiHeadAttention(embedding_dim, head_num)

        self.mlp1 = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)


        self.layer_norm4 = nn.LayerNorm(embedding_dim)
        self.layer_norm5 = nn.LayerNorm(embedding_dim)
        self.layer_norm6 = nn.LayerNorm(embedding_dim)

        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, y):
        #add
        _x = self.layer_norm1(x)
        _y = self.layer_norm4(y)

        _x = self.multi_head_attention1(_x)
        _y = self.multi_head_attention2(_y)

        x = x + _x
        y = y + _y

        _x = self.layer_norm2(x)
        _y = self.layer_norm5(y)

        _y = self.multi_head_cross_attention(_x,_y)

        y = y + _y

        _y = self.layer_norm3(y)

        _y = self.mlp1(_y)

        y = y + _y

        return y
    
class TransformerEncoderBlock2(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_cross_attention = MultiHeadCrossAttention2(embedding_dim, head_num)
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
        # _x = self.layer_norm2(x)
        # _y = self.layer_norm5(y)

        # _x,_y = self.multi_head_cross_attention(_x,_y)

        # _x = self.dropout1(_x)
        # _y = self.dropout2(_y)

        # x = x + _x
        # y = y + _y

        # x = self.layer_norm1(x)
        # y = self.layer_norm3(y)
        #add
        _x = self.layer_norm3(x)
        _y = self.layer_norm6(y)

        _x = self.mlp1(_x)
        _y = self.mlp2(_y)

        # _x = self.dropout1(_x)
        # _y = self.dropout2(_y)

        x = x + _x
        y = y + _y

        # x = self.layer_norm2(x)
        # y = self.layer_norm4(y)

        return x,y
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim / head_num) ** (1 / 2)

        #CT
        self.qkv_layer1 = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        # self.out_attention1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
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
        energy2 = torch.einsum("... i d , ... j d -> ... i j", query1, key2) / self.dk
        #self Attention
        # energy1 = torch.einsum("... i d , ... j d -> ... i j", query1, key1) / self.dk
        # energy2 = torch.einsum("... i d , ... j d -> ... i j", query2, key2) / self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention2 = torch.softmax(energy2, dim=-1)

        y = torch.einsum("... i j , ... j d -> ... i d", attention2, value2)

        y = rearrange(y, "b h t d -> b t (h d)")

        y = self.out_attention2(y)

        return y
    
class MultiHeadCrossAttention2(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim / head_num) ** (1 / 2)

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
        #self Attention
        # energy1 = torch.einsum("... i d , ... j d -> ... i j", query1, key1) / self.dk
        # energy2 = torch.einsum("... i d , ... j d -> ... i j", query2, key2) / self.dk

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
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim / head_num) ** 1 / 2

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
    
class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))  # width = out_channels

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)
        return x
    
class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.up_sample = up_sample(int(in_channels / 2), out_channels * 2, scale_factor)
        self.upconv_3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cat_f = cat_fuse(in_channels, out_channels)

    def forward(self, x, x_concat=None):
        # x = self.up_sample(x)  # 自己的上采样模块

        x = self.upsample(x)
        # if x_concat is not None:
        #     x = torch.cat([x_concat, x], dim=1)
        # x = self.layer(x)

        x = self.cat_f(x, x_concat)  # 融合模块

        return x
    
class cat_fuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cat_fuse, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x, x_cat):
        x = torch.cat([x, x_cat], dim=1)

        return self.conv2(x) + self.conv1(x)
    
class up_sample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(up_sample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, groups=2),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, groups=2),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   )

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=2),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True)
                                   )

    def forward(self, x):
        x = self.upsample(x)
        out = x + self.conv2(x) + self.conv3(x)
        return out

def DoubleUnet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = DoubleUnet(in_channels=2, n_cls=2, n_filters=16)
    return init_net(net, init_type, init_gain, gpu_id)

class DoubleUnet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super().__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters
        self.conv1x1 = nn.Conv2d(n_filters*4, self.n_cls, kernel_size=1, stride=1, padding=0)
        self.unet1 = Unet(in_channels=1, n_cls=2, n_filters=16)
        self.unet2 = Unet(in_channels=1, n_cls=2, n_filters=16)
        
    def forward(self,x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x1 = torch.zeros(x.shape[0],1,256,256).to(device)
        x2 = torch.zeros(x.shape[0],1,256,256).to(device)

        for i in range(x.shape[0]):
            x1[i,0] = x[i,0]
            x2[i,0] = x[i,1]

        x1 = self.unet1(x1)
        x2 = self.unet2(x2)

        x1 = torch.cat((x1,x2),dim=1)
        x = self.conv1x1(x1)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)
    
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels = 128, n_cls=2, n_filters=16):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.x0 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)  # 16*16

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(out_channels, int(out_channels * 1 / 4))

        self.cat_fuse = cat_fuse(2048, 512)

    def forward(self, x):

        # TODO  conv ****** transunet *********
        x0 = self.norm1(self.conv1(x))
        x1 = self.relu(x0)  # b*128*128*128
        x2 = self.encoder1(x1)  # b*256*64*64
        x3 = self.encoder2(x2)  # b*512*32*32
        x_out = self.encoder3(x3)  # b*1024*16*16
        x0 = self.x0(x)  # 自己添加
        
        x = x_out
        x = self.cat_fuse(x,x_out)
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x, x0)

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
    


