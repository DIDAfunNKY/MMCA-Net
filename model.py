import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import functools
from einops import rearrange
import torch.nn.functional as F
from functools import reduce
import numpy as np
from einops import rearrange, repeat



def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:

        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=1.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


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
    net = net.to(gpu_id)
    # init_weights(net, init_type, gain=init_gain)
    return net


def TGNet_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    # net = TransUNet(img_dim=256,
    #                 in_channels=input_nc,
    #                 out_channels=128,  # 128
    #                 head_num=8,  # 4
    #                 mlp_dim=64,  # 64
    #                 block_num=2,  #  1
    #                 patch_dim=16)  # 16
    # #VIT - Huge
    # net = TransUNet(img_dim=256,
    #                 in_channels=input_nc,
    #                 out_channels=128,  # 128
    #                 head_num=16,  #8 4
    #                 mlp_dim=5120,  # 64
    #                 block_num=32,  # 2 1
    #                 patch_dim=16)  # 16
    #VIT - Large
    # net = TransUNet(img_dim=256,
    #                 in_channels=input_nc,
    #                 out_channels=128,  # 128
    #                 head_num=16,  #8 4
    #                 mlp_dim=4096,  # 64
    #                 block_num=24,  # 2 1
    #                 patch_dim=16)  # 16
    # VIT - Base
    net = TransUNet(img_dim=256,
                    in_channels=input_nc,
                    out_channels=128,  # 128
                    head_num=16,  #8 4
                    mlp_dim=3072,  # 64
                    block_num=12,  # 2 1
                    patch_dim=16)  # 16

    return init_net(net, init_type, init_gain, gpu_id)


def unfold_TransUnet_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = unfold_TransUnet(img_dim=256,
                           in_channels=1,
                           out_channels=128,
                           head_num=4,
                           mlp_dim=64,
                           block_num=2,
                           patch_dim=16)
    return init_net(net, init_type, init_gain, gpu_id)


def TransUnet_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = TransUNet(img_dim=256,
                    in_channels=2,
                    out_channels=128,  # 128
                    head_num=4,  # 4
                    mlp_dim=2048,  # 64
                    block_num=2,  # 1
                    patch_dim=16)  # 16
    # net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    # net = Generator(img_channels=1, features=16, residuals=9)
    return init_net(net, init_type, init_gain, gpu_id)


def Unet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = Unet(in_channels=2, n_cls=1, n_filters=16)
    return init_net(net, init_type, init_gain, gpu_id)

def TransformerUnet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = TransformerUnet(in_channels=2, n_cls=2, n_filters=64)
    return init_net(net, init_type, init_gain, gpu_id)


def ResUnet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = ResUnet(in_channels=2, n_cls=2, n_filters=16)
    return init_net(net, init_type, init_gain, gpu_id)


def AttUnet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = AttUnet(in_channels=2, n_cls=2, n_filters=16)
    return init_net(net, init_type, init_gain, gpu_id)


def Zhao_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = Zhao(in_channels=2, n_cls=2, n_filters=16)
    return init_net(net, init_type, init_gain, gpu_id)


def Li_Dense_Unet_G(init_type='kaiming', init_gain=0.01, gpu_id='cuda:0'):
    net = Dense_unet(in_channels=2, n_cls=1, n_filters=16)
    return init_net(net, init_type, init_gain, gpu_id)


def Liu_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = Liu(in_channels=2, n_cls=2, n_filters=16)
    return init_net(net, init_type, init_gain, gpu_id)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.residual = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.residual(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, features=64, residuals=9):
        super().__init__()
        num_conv = 4
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, features, 7, 1, 3, padding_mode="reflect"),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.dense_1 = Single_level_densenet(features, num_conv)
        self.down_blocks = nn.Sequential(
            ConvBlock(features, features * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(features * 2, features * 4, kernel_size=3, stride=2, padding=1),
        )
        self.dense_2 = Single_level_densenet(features * 4, num_conv)
        self.down1_sk = SKConv(features * 4, features * 4)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(features * 4) for _ in range(residuals)]
        )
        self.dense_3 = Single_level_densenet(features * 4, num_conv)
        self.up_blocks = nn.Sequential(
            ConvBlock(features * 4, features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(features * 2, features * 1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.dense_4 = Single_level_densenet(features, num_conv)
        self.up1_sk = SKConv(features * 1, features * 1)
        self.last = nn.Conv2d(features, img_channels, 7, 1, 3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        x = self.dense_1(x)
        x = self.down_blocks(x)
        x = self.dense_2(x)
        x = self.down1_sk(x)
        x = self.res_blocks(x)
        x = self.dense_3(x)
        x = self.up_blocks(x)
        x = self.dense_4(x)
        x = self.up1_sk(x)
        x = self.last(x)
        return torch.tanh(x)


class Single_level_densenet(nn.Module):
    def __init__(self, filters, num_conv=3):
        super(Single_level_densenet, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters, filters, 3, padding=1))
            self.bn_list.append(nn.BatchNorm2d(filters))

    def forward(self, x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
        self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
        self.down1_sk = SKConv(ngf * 2, ngf * 2)
        self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)
        self.down2_sk = SKConv(ngf * 4, ngf * 4)
        self.down3 = Down(ngf * 4, ngf * 8, norm_layer, use_bias)
        self.down3_sk = SKConv(ngf * 8, ngf * 8)
        #   self.down4 = Down(ngf * 8,ngf * 16,norm_layer,use_bias)
        # self.down5 = Down(ngf * 16,ngf * 32,norm_layer,use_bias)
        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 8, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                               use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)
        # self.up1 = Up(ngf * 32, ngf * 16, norm_layer, use_bias)
        #  self.up1 = Up(ngf * 16, ngf * 8, norm_layer, use_bias)
        self.up1 = Up(ngf * 8, ngf * 4, norm_layer, use_bias)
        self.up1_sk = SKConv(ngf * 4, ngf * 4)
        self.up2 = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
        self.up2_sk = SKConv(ngf * 2, ngf * 2)
        self.up3 = Up(ngf * 2, ngf, norm_layer, use_bias)
        self.up3_sk = SKConv(ngf * 1, ngf * 1)
        self.outc = Outconv(ngf, output_nc)

    def forward(self, input):
        out = {}
        out['in'] = self.inc(input)

        out['d1'] = self.down1(out['in'])
        # out['d1_sk'] = self.down1_sk( out['d1'])

        out['d2'] = self.down2(out['d1'])
        #  out['d2_sk'] = self.down2_sk(out['d2'])
        out['d3'] = self.down3(out['d2'])
        out['d3_sk'] = self.down3_sk(out['d3'])
        #  out['d4'] = self.down4(out['d3'])

        # out['d5'] = self.down5(out['d4'])

        out['bottle'] = self.resblocks(out['d3_sk'])

        out['u1'] = self.up1(out['bottle'])
        out['u1_sk'] = self.up1_sk(out['u1'])

        out['u2'] = self.up2(out['u1_sk'])
        # out['u2_sk'] = self.up2_sk(out['u2'])
        out['u3'] = self.up3(out['u2'])
        #  out['u3_sk'] = self.up3_sk(out['u3'])
        # out['u4'] = self.up4(out['u3'])

        # out['u5'] = self.up5(out['u4'])

        return self.outc(out['u3'])


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=3, r=16, L=16):
        super(SKConv, self).__init__()
        d = 4
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=16, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())
            output.append(conv(input))
        # the part of fusion
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        # V1 = self.conv2(V)
        # V2 = self.pool(V1)
        # V2 = self.softmax(V2)
        # V  = V + V2
        return V


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
                      bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Down, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                            stride=2, padding=1, bias=use_bias),
                                  norm_layer(out_ch),
                                  nn.ReLU(True)
                                  )

    def forward(self, x):
        x = self.down(x)
        return x


# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(in_ch, out_ch,
            #           kernel_size=3, stride=1,
            #           padding=1, bias=use_bias),
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x


#
# def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
#              gpu_id='cuda:0'):
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm)
#
#     net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
#
#     return init_net(net, init_type, init_gain, gpu_id)
#
#
# # Defines the generator that consists of Resnet blocks between a few
# # downsampling/upsampling operations.
# class ResnetGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
#                  padding_type='reflect'):
#         assert (n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
#         self.input_nc = input_nc
#         self.output_nc = output_nc
#         self.ngf = ngf
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
#         self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
#         self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)
#
#         model = []
#         for i in range(n_blocks):
#             model += [ResBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                                use_bias=use_bias)]
#         self.resblocks = nn.Sequential(*model)
#
#         self.up1 = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
#         self.up2 = Up(ngf * 2, ngf, norm_layer, use_bias)
#
#         self.outc = Outconv(ngf, output_nc)
#
#     def forward(self, input):
#         out = {}
#         out['in'] = self.inc(input)
#         out['d1'] = self.down1(out['in'])
#         out['d2'] = self.down2(out['d1'])
#         out['bottle'] = self.resblocks(out['d2'])
#         out['u1'] = self.up1(out['bottle'])
#         out['u2'] = self.up2(out['u1'])
#
#         return self.outc(out['u2'])
#
#
# class Inconv(nn.Module):
#     def __init__(self, in_ch, out_ch, norm_layer, use_bias):
#         super(Inconv, self).__init__()
#         self.inconv = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
#                       bias=use_bias),
#             norm_layer(out_ch),
#             nn.ReLU(True)
#         )
#
#     def forward(self, x):
#         x = self.inconv(x)
#         return x
#
#
# class Down(nn.Module):
#     def __init__(self, in_ch, out_ch, norm_layer, use_bias):
#         super(Down, self).__init__()
#         self.down = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3,
#                       stride=2, padding=1, bias=use_bias),
#             norm_layer(out_ch),
#             nn.ReLU(True)
#         )
#
#     def forward(self, x):
#         x = self.down(x)
#         return x
#
#
# # Define a Resnet block
# class ResBlock(nn.Module):
#     def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         super(ResBlock, self).__init__()
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
#
#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         conv_block = []
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#
#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
#                        norm_layer(dim),
#                        nn.ReLU(True)]
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]
#
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
#                        norm_layer(dim)]
#
#         return nn.Sequential(*conv_block)
#
#     def forward(self, x):
#         out = x + self.conv_block(x)
#         return nn.ReLU(True)(out)
#
#
# class Up(nn.Module):
#     def __init__(self, in_ch, out_ch, norm_layer, use_bias):
#         super(Up, self).__init__()
#         self.up = nn.Sequential(
#             # nn.Upsample(scale_factor=2, mode='nearest'),
#             # nn.Conv2d(in_ch, out_ch,
#             #           kernel_size=3, stride=1,
#             #           padding=1, bias=use_bias),
#             nn.ConvTranspose2d(in_ch, out_ch,
#                                kernel_size=3, stride=2,
#                                padding=1, output_padding=1,
#                                bias=use_bias),
#             norm_layer(out_ch),
#             nn.ReLU(True)
#         )
#
#     def forward(self, x):
#         x = self.up(x)
#         return x
#
#
# class Outconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Outconv, self).__init__()
#         self.outconv = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x = self.outconv(x)
#         return x


def define_D(input_nc, ndf, netD, n_layers_D=6, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers = n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_id)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim / head_num) ** (1 / 2)

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
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #错误写法
        # _x = self.multi_head_attention(x)
        # _x = self.dropout(_x)
        # x = x + _x
        # x = self.layer_norm1(x)

        # _x = self.mlp(x)
        # x = x + _x
        # x = self.layer_norm2(x)
        
        #正确写法
        _x = self.layer_norm1(x)
        _x = self.multi_head_attention(_x)
        _x = self.dropout(_x)
        x = x + _x

        _x = self.layer_norm2(x)
        _x = self.mlp(_x)
        x = x + _x

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

        self.dropout = nn.Dropout(0.1)

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
        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)
        x = self.layer(x)

        # x = self.cat_f(x, x_concat)  # 融合模块

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


class DenseNet(nn.Module):
    def __init__(self, filters, num_conv=3):
        super(DenseNet, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters, filters, kernel_size=3, padding=1))
            self.bn_list.append(nn.BatchNorm2d(filters))

    def forward(self, x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        # TODO============================= 利用 unfold 函数 encoding =====================
        # self.conv_256 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias=False),
        #                               nn.BatchNorm2d(64),
        #                               nn.ReLU(inplace=True))
        # self.un_16 = nn.Sequential(nn.Conv2d(in_channels, 4, kernel_size=5, stride=1, padding=2, bias=False),
        #                            nn.BatchNorm2d(4),
        #                            nn.ReLU(inplace=True))
        # self.un_32 = nn.Sequential(nn.Conv2d(in_channels, 8, kernel_size=5, stride=1, padding=2, bias=False),
        #                            nn.BatchNorm2d(8),
        #                            nn.ReLU(inplace=True))
        # self.un_64 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2, bias=False),
        #                            nn.BatchNorm2d(16),
        #                            nn.ReLU(inplace=True))
        # self.un_128 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2, bias=False),
        #                             nn.BatchNorm2d(32),
        #                             nn.ReLU(inplace=True))
        # self.unfold = nn.Unfold(kernel_size=2, dilation=1, padding=0, stride=2)

        # TODO============================= 一般卷积 encoding =====================
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)

        self.x0 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)  # 16*16

        # # TODO============================= DenseNet encoding =====================
        # self.dense_f_1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True)
        # self.dense_1 = DenseNet(out_channels, num_conv=4)
        # self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dense_f_2 = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # self.dende_2 = DenseNet(2 * out_channels, num_conv=4)
        # self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dense_f_3 = nn.Conv2d(2 * out_channels, 4 * out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # self.dende_3 = DenseNet(4 * out_channels, num_conv=4)
        # self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dense_f_4 = nn.Conv2d(4 * out_channels, 8 * out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        # self.dende_4 = DenseNet(8 * out_channels, num_conv=4)
        # self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.vit_img_dim = img_dim // patch_dim  # 256 // 16 = 16
        self.vit_img_dim = 9
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)
        self.modal_fuse = modal_fuse(1, 3, 3)
        self.cat_fuse = cat_fuse(2048, 512)

    def forward(self, x):

        # # TODO   unfold 函数 **********
        # x0 = self.conv_256(x)  # b*64*256*256
        # x_un_16 = self.un_16(x)
        # x_un_32 = self.un_32(x)
        # x_un_64 = self.un_64(x)
        # x_un_128 = self.un_128(x)
        # unf_16 = self.unfold(x_un_16)
        # unf_32 = self.unfold(x_un_32)
        # unf_64 = self.unfold(x_un_64)
        # unf_128 = self.unfold(x_un_128)
        # x1 = unf_128.view(x.shape[0], -1, unf_128.shape[1], unf_128.shape[1])  # b*128*128*128
        # x2 = unf_64.view(x.shape[0], -1, unf_64.shape[1], unf_64.shape[1])  # b*256*64*64
        # x3 = unf_32.view(x.shape[0], -1, unf_32.shape[1], unf_32.shape[1])  # b*512*32*32
        # x_out = unf_16.view(x.shape[0], -1, unf_16.shape[1], unf_16.shape[1])  # b*1024*16*16

        # x1 = unf_128.view(x.shape[0], -1, 72, 72)  # b*128*72*72
        # x2 = unf_64.view(x.shape[0], -1, 36, 36)
        # x3 = unf_32.view(x.shape[0], -1, 18, 18)  # b*512*32*32
        # x_out = unf_16.view(x.shape[0], -1, 9, 9)  # b*1024*16*16

        # TODO  conv ****** transunet *********
        x0 = self.norm1(self.conv1(x))
        x1 = self.relu(x0)  # b*128*128*128
        x2 = self.encoder1(x1)  # b*256*64*64
        x3 = self.encoder2(x2)  # b*512*32*32
        x_out = self.encoder3(x3)  # b*1024*16*16
        x0 = self.x0(x)  # 自己添加

        # TODO 以下为共同
        x = self.vit(x_out)
        # print(x_out.shape)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)  # b*1024*16*16
        # print(x.shape)
        x = self.cat_fuse(x, x_out)  # b*512*16*16

        return x, x0, x1, x2, x3


class modal_fuse(nn.Module):
    def __init__(self, in_channels, out_channels, m):
        super(modal_fuse, self).__init__()
        self.conv3d = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, m), padding=(1, 1, 0)),
                                    nn.BatchNorm3d(out_channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        ct = x[:, 0, :, :].unsqueeze(1).unsqueeze(-1)
        suv = x[:, 1, :, :].unsqueeze(1).unsqueeze(-1)
        ki = x[:, 2, :, :].unsqueeze(1).unsqueeze(-1)
        w = torch.cat([ct, suv, ki], dim=4)
        w = self.conv3d(w).squeeze(-1)
        out = w * x
        return out


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(out_channels, int(out_channels * 1 / 4))

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 4), 1, kernel_size=1)

    def forward(self, x, x0, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x, x0)
        x = self.conv1(x)

        # return x
        return torch.sigmoid(x)


class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder = Decoder(out_channels)

    def forward(self, x):
        x, x0, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x0, x1, x2, x3)

        return x


class TransUnet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder = Decoder(out_channels)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)

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

        x = self.conv1x1(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)
        

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
                head_num = 4, mlp_dim = 3072, block_num = 1, patch_dim=1, classification=False)

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


class ResUnet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(ResUnet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv2d(in_channels, n_filters, kernel_size=5, stride=1, padding=2)
        self.block_1_2_left = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.res_1 = BasicConv2d(in_channels, n_filters, kernel_size=1, stride=1)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.res_2 = BasicConv2d(n_filters, 2 * n_filters, kernel_size=1, stride=1)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.res_3 = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=1, stride=1)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.res_4 = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=1, stride=1)

        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 512, 1/16
        self.block_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_5_2_left = BasicConv2d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.res_5 = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=1, stride=1)

        self.upconv_4 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.res_6 = BasicConv2d(16 * n_filters, 8 * n_filters, kernel_size=1, stride=1)

        self.upconv_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.res_7 = BasicConv2d(8 * n_filters, 4 * n_filters, kernel_size=1, stride=1)

        self.upconv_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.res_8 = BasicConv2d(4 * n_filters, 2 * n_filters, kernel_size=1, stride=1)

        self.upconv_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.res_9 = BasicConv2d(2 * n_filters, n_filters, kernel_size=1, stride=1)

        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x)) + self.res_1(x)
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0))) + self.res_2(self.pool_1(ds0))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1))) + self.res_3(self.pool_2(ds1))
        ds3 = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2))) + self.res_4(self.pool_3(ds2))
        x = self.block_5_2_left(self.block_5_1_left(self.pool_3(ds3))) + self.res_5(self.pool_3(ds3))

        x = self.block_4_2_right(self.block_4_1_right(torch.cat([self.upconv_4(x), ds3], 1))) + self.res_6(torch.cat([self.upconv_4(x), ds3], 1))
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1))) + self.res_7(torch.cat([self.upconv_3(x), ds2], 1))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1))) + self.res_8(torch.cat([self.upconv_2(x), ds1], 1))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1))) + self.res_9(torch.cat([self.upconv_1(x), ds0], 1))

        x = self.conv1x1(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


class AttUnet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(AttUnet, self).__init__()
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
        self.att_gate_5 = Attention_gate(8 * n_filters, scale_factor=2)

        self.upconv_4 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_4_1_right = BasicConv2d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.att_gate_4 = Attention_gate(4 * n_filters, scale_factor=2)

        self.upconv_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv2d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.att_gate_3 = Attention_gate(2 * n_filters, scale_factor=2)

        self.upconv_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.att_gate_2 = Attention_gate(n_filters, scale_factor=2)

        self.upconv_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))  # 256*256
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))  # 128*128
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))  # 64*64
        ds3 = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))  # 32*32
        x = self.block_5_2_left(self.block_5_1_left(self.pool_3(ds3)))  # 16*16

        x = self.block_4_2_right(self.block_4_1_right(torch.cat([self.upconv_4(x), ds3], 1))) + self.att_gate_5(ds3, x)
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1))) + self.att_gate_4(ds2, x)
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1))) + self.att_gate_3(ds1, x)
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1))) + self.att_gate_2(ds0, x)

        x = self.conv1x1(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


class Zhao(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(Zhao, self).__init__()
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
        # self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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
        suv_3 = self.upconv_1(self.upconv_2(self.upconv_3(x)))
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        suv_2 = self.upconv_1(self.upconv_2(x))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        suv_1 = self.upconv_1(x)
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))
        x_1 = self.conv1x1(suv_3)
        x_2 = self.conv1x1(suv_2)
        x_3 = self.conv1x1(suv_1)
        x = self.conv1x1(x)

        #权重融合参数
        if self.n_cls == 1:
            return torch.sigmoid(x + 0.5*x_1 + 0.3*x_2 + 0.1*x_3)
        else:
            return F.softmax(x + 0.5*x_1 + 0.3*x_2 + 0.1*x_3, dim=1)


class Liu(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(Liu, self).__init__()
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

        x = self.conv1x1(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


class ki_spatial(nn.Module):
    def __init__(self, n_filters):
        super(ki_spatial, self).__init__()
        self.conv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = nn.InstanceNorm2d(n_filters)
        self.s = nn.Sigmoid()

    def forward(self, x):
        cc = self.conv(x)
        xxx = self.norm(self.conv(x))
        x1 = self.s(self.norm(self.conv(x)))
        return self.norm(x + x1)


class Dense_unet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters, img_dim=256, patch_dim=16, head_num=4, mlp_dim=64, block_num=1):
        super(Dense_unet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv2d(2, n_filters, kernel_size=5, stride=1, padding=2)
        self.block_1_2_left = DenseNet(n_filters, num_conv=3)
        self.conv_1 = BasicConv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv2d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = DenseNet(2 * n_filters, num_conv=3)
        self.conv_2 = BasicConv2d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv2d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = DenseNet(4 * n_filters, num_conv=3)
        self.conv_3 = BasicConv2d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv2d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = DenseNet(8 * n_filters, num_conv=3)
        self.conv_4 = BasicConv2d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block_5_1_left = BasicConv2d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_5_2_left = DenseNet(16 * n_filters, num_conv=3)

        self.upconv_4 = nn.ConvTranspose2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_4_1_right = BasicConv2d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = DenseNet(8 * n_filters, num_conv=3)

        self.upconv_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_3_1_right = BasicConv2d((4+4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = DenseNet(4 * n_filters, num_conv=3)

        self.upconv_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_2_1_right = BasicConv2d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = DenseNet(2 * n_filters, num_conv=3)

        self.upconv_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.block_1_1_right = BasicConv2d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = DenseNet(n_filters, num_conv=3)

        self.conv1x1 = nn.Conv2d(n_filters, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))  # 64*256*256
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))  # 128*128*128
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))  # 256*64*64
        ds3 = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))  # 512*32*32
        ds4 = self.block_5_2_left(self.block_5_1_left(self.pool_4(ds3)))  # 1024*16*16

        x1 = self.block_4_2_right(self.block_4_1_right(torch.cat([self.upconv_4(ds4), self.conv_4(ds3)], 1)))  # 512*32*32
        x2 = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x1), self.conv_3(ds2)], 1)))  # 256*64*64
        x3 = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x2), self.conv_2(ds1)], 1)))  # 128*128*128
        x4 = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x3), self.conv_1(ds0)], 1)))  # 64*256*256

        s1 = self.upconv_1(self.upconv_2(self.upconv_3(x1)))
        s2 = self.upconv_1(self.upconv_2(x2))
        s3 = self.upconv_1(x3)

        out = self.conv1x1(x4 + s1 + s2 + s3)
        out1 = self.conv1x1(s1)
        out2 = self.conv1x1(s2)
        out3 = self.conv1x1(s3)

        if self.n_cls == 1:
            return torch.sigmoid(out+out1+out2+out3)
        else:
            return F.softmax(x, dim=1)


class Attention_gate(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super(Attention_gate, self).__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x, g):
        x0 = self.down(x)
        g = self.conv(g)
        x1 = F.relu(x0 + g)
        x1 = F.sigmoid(self.conv1x1(x1))
        x1 = self.up(x1)
        out = x * x1
        return out


class unfold_TransUnet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        # TODO============================= 利用 unfold 函数 encoding =====================
        self.x_16 = nn.Sequential(nn.Conv2d(in_channels, 4, kernel_size=5, stride=1, padding=2, bias=False),
                                  nn.BatchNorm2d(4),
                                  nn.ReLU(inplace=True))
        self.x_32 = nn.Sequential(nn.Conv2d(in_channels, 8, kernel_size=5, stride=1, padding=2, bias=False),
                                  nn.BatchNorm2d(8),
                                  nn.ReLU(inplace=True))
        self.x_64 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2, bias=False),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(inplace=True))
        self.x_128 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.unfold = nn.Unfold(kernel_size=2, dilation=1, padding=0, stride=2)
        self.fold = nn.Fold(output_size=(256, 256), kernel_size=2, stride=2)

        self.vit_img_dim = img_dim // patch_dim  # 256 // 16 = 16
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)
        # self.vit_32 = ViT(32, out_channels * 4, out_channels * 4,
        #                head_num, mlp_dim, block_num, patch_dim=1, classification=False)
        # self.vit_64 = ViT(64, out_channels * 2, out_channels * 2,
        #                   head_num, mlp_dim, block_num, patch_dim=1, classification=False)
        # self.vit_128 = ViT(128, out_channels, out_channels,
        #                   head_num, mlp_dim, block_num, patch_dim=1, classification=False)
        # self.DeConv3 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.conv32 = nn.Conv2d(out_channels * 4, 512, kernel_size=3, stride=1, padding=1)

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        self.conv_1 = nn.Conv2d(int(out_channels * 1 / 8), 1, kernel_size=1)

        # TODO============================= 一般卷积 encoding =====================
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)  # 16*16

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse_16 = FUSE(1024, 1024*3, 512)
        self.fuse_32 = FUSE(512, 512*4, 256)
        self.fuse_64 = FUSE(256, 256*4, 128)
        self.fuse_128 = FUSE(128, 128*4, 64)

        self.fu_16 = nn.Sequential(nn.Conv2d(1024*3, 1024*3, kernel_size=1),
                                   nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid())
        self.conv_1024 = nn.Conv2d(1024*3, 512, kernel_size=1)

        self.fu_32 = nn.Sequential(nn.Conv2d(512*4, 512*4, kernel_size=1),
                                   nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid())
        self.conv_512 = nn.Conv2d(512 * 4, 256, kernel_size=1)

        self.fu_64 = nn.Sequential(nn.Conv2d(256 * 4, 256 * 4, kernel_size=1),
                                   nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid())
        self.conv_256 = nn.Conv2d(256 * 4, 128, kernel_size=1)

        self.fu_128 = nn.Sequential(nn.Conv2d(128 * 4, 128 * 4, kernel_size=1),
                                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                    nn.Sigmoid())
        self.conv_128 = nn.Conv2d(128 * 4, 16, kernel_size=1)
        self.slu = nn.SiLU()

        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        ct = x[:, 0, :, :].unsqueeze(1)
        suv = x[:, 1, :, :].unsqueeze(1)
        ki = x[:, 2, :, :].unsqueeze(1)
        #
        # # TODO  conv SUV ***************
        # x_suv_0 = self.norm1(self.conv1(suv))
        # x_suv_1 = self.relu(x_suv_0)  # b*128*128*128
        # x_suv_2 = self.encoder1(x_suv_1)  # b*256*64*64
        # x_suv_3 = self.encoder2(x_suv_2)  # b*512*32*32
        # x_suv_out = self.encoder3(x_suv_3)  # b*1024*16*16
        # x_suv_out = self.vit(x_suv_out)
        # x_suv_out = rearrange(x_suv_out, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)
        # x_suv_out = self.relu(self.norm2(self.conv2(x_suv_out)))
        #
        # ds_suv_1 = self.decoder1(x_suv_out, x_suv_3)  # b*256*32*32
        # ds_suv_2 = self.decoder2(ds_suv_1, x_suv_2)  # b*128*64*64
        # ds_suv_3 = self.decoder3(ds_suv_2, x_suv_1)  # b*64*128*128
        # ds_suv_4 = self.decoder4(ds_suv_3)  # b*16*256*256
        # x_suv = self.conv_1(ds_suv_4)
        #
        # # TODO  conv Ki ***************
        # x_ki_0 = self.norm1(self.conv1(ki))
        # x_ki_1 = self.relu(x_ki_0)  # b*128*128*128
        # x_ki_2 = self.encoder1(x_ki_1)  # b*256*64*64
        # x_ki_3 = self.encoder2(x_ki_2)  # b*512*32*32
        # x_ki_out = self.encoder3(x_ki_3)  # b*1024*16*16
        # x_ki_out = self.vit(x_ki_out)
        # x_ki_out = rearrange(x_ki_out, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)
        # x_ki_out = self.relu(self.norm2(self.conv2(x_ki_out)))
        #
        # ds_ki_1 = self.decoder1(x_ki_out, x_ki_3)  # b*256*32*32
        # ds_ki_2 = self.decoder2(ds_ki_1, x_ki_2)  # b*128*64*64
        # ds_ki_3 = self.decoder3(ds_ki_2, x_ki_1)  # b*64*128*128
        # ds_ki_4 = self.decoder4(ds_ki_3)  # b*16*256*256
        # x_ki = self.conv_1(ds_ki_4)
        #
        # # TODO  conv CT ***************
        # x_ct_0 = self.norm1(self.conv1(ct))
        # x_ct_1 = self.relu(x_ct_0)  # b*128*128*128
        # x_ct_2 = self.encoder1(x_ct_1)  # b*256*64*64
        # x_ct_3 = self.encoder2(x_ct_2)  # b*512*32*32
        # x_ct_out = self.encoder3(x_ct_3)  # b*1024*16*16
        # x_ct_out = self.vit(x_ct_out)
        # x_ct_out = rearrange(x_ct_out, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)
        # x_ct_out = self.relu(self.norm2(self.conv2(x_ct_out)))
        #
        # ds_ct_1 = self.decoder1(x_ct_out, x_ct_3)  # b*256*32*32
        # # ds_ct_1 = ds_ct_1 + ds_suv_1 + ds_ki_1
        # ds_ct_1 = ds_ct_1 + ds_suv_1
        # ds_ct_2 = self.decoder2(ds_ct_1, x_ct_2)  # b*128*64*64
        # # ds_ct_2 = ds_ct_2 + ds_suv_2 + ds_ki_2
        # ds_ct_2 = ds_ct_2 + ds_suv_2
        # ds_ct_3 = self.decoder3(ds_ct_2, x_ct_1)  # b*64*128*128
        # # ds_ct_3 = ds_ct_3 + ds_suv_3 + ds_ki_3
        # ds_ct_3 = ds_ct_3 + ds_suv_3
        # ds_ct_4 = self.decoder4(ds_ct_3)  # b*16*256*256
        # # ds_ct_4 = ds_ct_4 + ds_suv_4 + ds_ki_4
        # ds_ct_4 = ds_ct_4 + ds_suv_4
        # x_ct = self.conv_1(ds_ct_4)  # b*1*256*256

        # TODO   unfold 函数   SUV **********
        x_16_suv = self.unfold(self.x_16(suv))  # patch_size: 16*16
        x_32_suv = self.unfold(self.x_32(suv))  # patch_size: 32*32
        x_64_suv = self.unfold(self.x_64(suv))  # patch_size: 32*32
        x_128_suv = self.unfold(self.x_128(suv))  # patch_size: 32*32
        x_suv_1 = x_128_suv.view(ct.shape[0], -1, x_128_suv.shape[1], x_128_suv.shape[1])  # b*128*128*128
        x_suv_2 = x_64_suv.view(ct.shape[0], -1, x_64_suv.shape[1], x_64_suv.shape[1])  # b*256*64*64
        x_suv_3 = x_32_suv.view(ct.shape[0], -1, x_32_suv.shape[1], x_32_suv.shape[1])  # b*512*32*32
        x_suv_out = x_16_suv.view(ct.shape[0], -1, x_16_suv.shape[1], x_16_suv.shape[1])  # b*1024*16*16
        x_suv_out = self.vit(x_suv_out)
        x_suv_out = rearrange(x_suv_out, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)
        # x_suv_out = self.relu(self.norm2(self.conv2(x_suv_out)))

        # ds_suv_1 = self.decoder1(x_suv_out, x_suv_3)  # b*256*32*32
        # ds_suv_2 = self.decoder2(ds_suv_1, x_suv_2)  # b*128*64*64
        # ds_suv_3 = self.decoder3(ds_suv_2, x_suv_1)  # b*64*128*128
        # ds_suv_4 = self.decoder4(ds_suv_3)  # b*16*256*256
        # x_suv = self.conv_1(ds_suv_4)

        # TODO   unfold 函数   Ki **********
        x_16_ki = self.unfold(self.x_16(ki))  # patch_size: 16*16
        x_32_ki = self.unfold(self.x_32(ki))  # patch_size: 32*32
        x_64_ki = self.unfold(self.x_64(ki))  # patch_size: 32*32
        x_128_ki = self.unfold(self.x_128(ki))  # patch_size: 32*32
        x_ki_1 = x_128_ki.view(ct.shape[0], -1, x_128_ki.shape[1], x_128_ki.shape[1])  # b*128*128*128
        x_ki_2 = x_64_ki.view(ct.shape[0], -1, x_64_ki.shape[1], x_64_ki.shape[1])  # b*256*64*64
        x_ki_3 = x_32_ki.view(ct.shape[0], -1, x_32_ki.shape[1], x_32_ki.shape[1])  # b*512*32*32
        x_ki_out = x_16_ki.view(ct.shape[0], -1, x_16_ki.shape[1], x_16_ki.shape[1])  # b*1024*16*16
        x_ki_out = self.vit(x_ki_out)
        x_ki_out = rearrange(x_ki_out, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)
        # x_ki_out = self.relu(self.norm2(self.conv2(x_ki_out)))

        # ds_ki_1 = self.decoder1(x_ki_out, x_ki_3)  # b*256*32*32
        # ds_ki_2 = self.decoder2(ds_ki_1, x_ki_2)  # b*128*64*64
        # ds_ki_3 = self.decoder3(ds_ki_2, x_ki_1)  # b*64*128*128
        # ds_ki_4 = self.decoder4(ds_ki_3)  # b*16*256*256
        # x_ki = self.conv_1(ds_ki_4)

        # TODO   unfold 函数   CT **********
        x_16_ct = self.unfold(self.x_16(ct))  # patch_size: 16*16
        x_32_ct = self.unfold(self.x_32(ct))  # patch_size: 32*32
        x_64_ct = self.unfold(self.x_64(ct))  # patch_size: 32*32
        x_128_ct = self.unfold(self.x_128(ct))  # patch_size: 32*32
        x_ct_1 = x_128_ct.view(ct.shape[0], -1, x_128_ct.shape[1], x_128_ct.shape[1])  # b*128*128*128
        x_ct_2 = x_64_ct.view(ct.shape[0], -1, x_64_ct.shape[1], x_64_ct.shape[1])  # b*256*64*64
        x_ct_3 = x_32_ct.view(ct.shape[0], -1, x_32_ct.shape[1], x_32_ct.shape[1])  # b*512*32*32
        x_ct_out = x_16_ct.view(ct.shape[0], -1, x_16_ct.shape[1], x_16_ct.shape[1])  # b*1024*16*16
        x_ct_out = self.vit(x_ct_out)
        x_ct_out = rearrange(x_ct_out, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        # x_16 = torch.cat([x_ct_out, x_suv_out, x_ki_out], dim=1)
        # x_16_s = self.fu_16(x_16)
        # x_16 = self.upsample(self.slu(self.conv_1024(x_16 * x_16_s)))  # b*512*32*32
        #
        # x_32 = torch.cat([x_16, x_ct_3, x_suv_3, x_ki_3], dim=1)
        # x_32_s = self.fu_32(x_32)
        # x_32 = self.upsample(self.slu(self.conv_512(x_32 * x_32_s)))  # b*256*64*64
        #
        # x_64 = torch.cat([x_32, x_ct_2, x_suv_2, x_ki_2], dim=1)
        # x_64_s = self.fu_64(x_64)
        # x_64 = self.upsample(self.slu(self.conv_256(x_64 * x_64_s)))  # b*128*128*128
        #
        # x_128 = torch.cat([x_64, x_ct_1, x_suv_1, x_ki_1], dim=1)
        # x_128_s = self.fu_128(x_128)
        # x_128 = self.upsample(self.slu(self.conv_128(x_128 * x_128_s)))  # b*16*256*256
        # out = self.conv_1(x_128)
        # x_ct_out = self.relu(self.norm2(self.conv2(x_ct_out)))

        x_16 = self.fuse_16(x_ct_out, x_suv_out, x_ki_out)
        x_16 = self.upsample(x_16)
        x_32 = self.fuse_32(x_16, x_ct_3, x_suv_3, x_ki_3)
        x_32 = self.upsample(x_32)
        x_64 = self.fuse_64(x_32, x_ct_2, x_suv_2, x_ki_2)
        x_64 = self.upsample(x_64)
        x_128 = self.fuse_128(x_64, x_ct_1, x_suv_1, x_ki_1)
        x_128 = self.upsample(x_128)
        out = self.conv1x1(x_128)

        # ds_ct_1 = self.decoder1(x_ct_out, x_ct_3)  # b*256*32*32
        # # ds_ct_1 = ds_ct_1 + ds_suv_1 + ds_ki_1
        # ds_ct_1 = ds_ct_1 + ds_suv_1
        # ds_ct_2 = self.decoder2(ds_ct_1, x_ct_2)  # b*128*64*64
        # # ds_ct_2 = ds_ct_2 + ds_suv_2 + ds_ki_2
        # ds_ct_2 = ds_ct_2 + ds_suv_2
        # ds_ct_3 = self.decoder3(ds_ct_2, x_ct_1)  # b*64*128*128
        # # ds_ct_3 = ds_ct_3 + ds_suv_3 + ds_ki_3
        # ds_ct_3 = ds_ct_3 + ds_suv_3
        # ds_ct_4 = self.decoder4(ds_ct_3)  # b*16*256*256
        # # ds_ct_4 = ds_ct_4 + ds_suv_4 + ds_ki_4
        # ds_ct_4 = ds_ct_4 + ds_suv_4
        # x_ct = self.conv1(ds_ct_4)  # b*1*256*256

        return torch.sigmoid(out)


class FUSE(nn.Module):
    def __init__(self, in_channels, m_channels, out_channels, **kwargs):
        super(FUSE, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, m_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(m_channels),
                                   nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.Conv2d(m_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3=None, x4=None):
        x = x1 + x2 + x3
        x_cat = torch.cat([x1, x2, x3], dim=1)
        if x4 is not None:
            x = x + x4
            x_cat = torch.cat([x_cat, x4], dim=1)
        x = self.conv1(x)
        re = x * x_cat
        out = self.conv2(re)
        return out



