import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import warnings
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
from dataset import dualctdataset, RandomGenerator, RandomGenerator1, HECKTORdataset
from sam import samt
import argparse
from sklearn.model_selection import KFold
import os
from torchvision.models import vgg19
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import ex_transforms
from model import define_D, GANLoss, get_scheduler, update_learning_rate, Unet_G, TransUnet_G, Zhao_G, \
    Li_Dense_Unet_G, ResUnet_G, AttUnet_G, Liu_G, TGNet_G
from CaNet import DoubleUnet_G,DoubleUnet2_G,CaUnet_G,DTransUnet_G,MHCAUnet_G,TransformerUnet_G,CaSUnet_G
from natsort import natsorted
import losses
import metrics
from loguru import logger
from tqdm import *
warnings.filterwarnings("ignore")
CUDA_LAUNCH_BLOCKING = 1

curr_time = datetime.datetime.now()

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='testing batch size')
parser.add_argument('--input_nc', type=int, default=2, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=2, help='generator filters in first conv layer')
# parser.add_argument('--ndf', type=int, default=4, help='discriminator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=10, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=1, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb1', type=int, default=0.01, help='weight on L1 term in objective') #0.01
parser.add_argument('--lamb2', type=int, default=0.1, help='weight on L2 term in objective')  #0.1
parser.add_argument('--glr', type=int, default=0.000128, help='initial learning rate for SGD')
parser.add_argument('--trc', type=int, default=1, help='train count')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    logger.info("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer):
    logger.info("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda:0")
    # 预训练模型 要求模型一模一样，每一层的写法和命名都要一样 本质一样都不行
    # 完全一样的模型实现，也可能因为load和当初save时 pytorch版本不同 而导致state_dict中的key不一样
    # 例如 "initial.0.weight" 与 “initial.weight” 的区别
    model.load_state_dict(checkpoint["state_dict"], strict=False)  # 改成strict=False才能编译通过
    optimizer.load_state_dict(checkpoint["optimizer"])


class WGAN_VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(WGAN_VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        out = self.feature_extractor(x)
        return out
logger.add(f"log/JBHI-DATA.log")

def testOne(model):

    if model == 'TGNet':
        net_g = TGNet_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
    elif model == 'CaNet':
        net_g = CaUnet_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'CaSNet':
        net_g = CaSUnet_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'transunet':
        net_g = TransUnet_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
    elif model == 'dtransunet':
        net_g = DTransUnet_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    # elif model == 'unetpp':
    #     net_g = Unetpp_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'unet':
        net_g = Unet_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'resunet':
        net_g = ResUnet_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'attunet':
        net_g = AttUnet_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'MHCAUnet':
        net_g = MHCAUnet_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'transformerunet':
        net_g = TransformerUnet_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'dunet':
        net_g = DoubleUnet_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'dunet2':
        net_g = DoubleUnet2_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'zhao':
        net_g = Zhao_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'DenseUnet':
        net_g = Li_Dense_Unet_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'Liu':
        net_g = Liu_G(init_type='normal', init_gain=0.02, gpu_id='cuda:0')
    elif model == 'Sam':
        net_g = samt().to(device)
    else:
        raise ValueError("input net error")
    
    # 判别器 discriminator
    net_d = define_D(opt.output_nc, opt.ndf, 'pixel', gpu_id=device,use_sigmoid=True)
    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)


    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # optimizer_g = optim.SGD(net_g.parameters(),momentum=0.9,weight_decay=1e-4,lr=opt.glr)

    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)

    best_dice = 0
    best_epoch = 0
    Dice = []
    TDice = []
    Loss = []



    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        # train
        net_g.train()
        train_dice = 0
        train_loss = 0

        for i in range(opt.trc):
            #qbar
            qbar = trange(len(train_set_loader))

            for iteration, batch in enumerate(train_set_loader, 1):
                optimizer_g.zero_grad()
                # forward
                real_a, real_b = batch['input'].to(device), batch['target'].to(device)
  
                fake_b = net_g(real_a)


                if model != 'Liu':
                    #no GAN
                    #loss function
                    loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb1
                    loss_g_l2 = criterionMSE(fake_b, real_b) * opt.lamb2
                    # loss_g = losses.FocalLoss()(fake_b, real_b) + loss_g_l1
                    # loss_g = losses.Dice_and_FocalLoss()(fake_b, real_b) + loss_g_l1
                    loss_g = losses.Dice_and_FocalLoss()(fake_b, real_b) + loss_g_l1 + loss_g_l2
                    # loss_g = criterionL1(fake_b, real_b)

                    loss_g.backward()

                    optimizer_g.step()

                    dice = metrics.dice(ex_transforms.ProbsToLabels()(fake_b.detach().cpu().numpy()), real_b.detach().cpu().numpy())
                    train_dice = train_dice + dice
                    train_loss = train_loss + loss_g.item()

                    qbar.set_postfix(epoch = epoch,loss=loss_g.item(), dice=dice.item())  # 进度条右边显示信息
                    qbar.update(1)
                else:
                    # GAN
                    ######################
                    # (1) Update D network
                    ######################
                    optimizer_d.zero_grad()
                    # train with fake
                    # fake_ab = torch.cat((real_a, fake_b), 1)
                    pred_fake = net_d.forward(fake_b.detach())
                    loss_d_fake = criterionGAN(pred_fake, False)

                    # train with real
                    # real_ab = torch.cat((real_a, real_b), 1)
                    pred_real = net_d.forward(real_b)
                    loss_d_real = criterionGAN(pred_real, True)

                    # Combined D loss
                    loss_d = (loss_d_fake + loss_d_real) * 0.5
                    loss_d.backward()
                    optimizer_d.step()
                    optimizer_g.zero_grad()
                    pred_fake = net_d.forward(fake_b)
                    loss_g_gan = criterionGAN(pred_fake, True)

                    # Second, G(A) = B
                    loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb

                    loss_g = losses.DiceLoss()(fake_b, real_b) + 0.001 * loss_g_gan
                    # loss_g = loss(fake_b, real_b)
                    loss_g.backward()
                    optimizer_g.step()
                    # print("Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} dice: {:.4f}".format(
                    # epoch, iteration, len(train_set_loader), loss_d.item(), loss_g.item(), dice.item()))
                    dice = metrics.dice(ex_transforms.ProbsToLabels()(fake_b.detach().cpu().numpy()), real_b.detach().cpu().numpy())
                    train_dice = train_dice + dice
                    train_loss = train_loss + loss_g.item()
                    qbar.set_postfix(epoch = epoch,lossg=loss_g.item(), lossd=loss_d.item(),dice=dice.item())  # 进度条右边显示信息
                    qbar.update(1)
            
        update_learning_rate(net_g_scheduler, optimizer_g)
        if model == 'Liu':
            update_learning_rate(net_d_scheduler, optimizer_d)
        # test
        tal_dice = 0
        net_g.eval()

        with torch.no_grad():
            num = 1
            for sample in val_set_loader:
                input, target = sample['input'].to(device), sample['target'].to(device)


                prediction = net_g(input)
 

                dice_pred = metrics.dice(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()), target.cpu().numpy())
                tal_dice += dice_pred

                num += 1
            avg_dice = (tal_dice / len(val_set_loader))

            Dice.append(avg_dice)
            Loss.append(train_loss / (len(train_set_loader)*opt.trc))
            TDice.append(train_dice / (len(train_set_loader)*opt.trc))
            logger.info('*' * 100)
            logger.info("train avg_loss: {:.4f} ".format(train_loss / (len(train_set_loader)*opt.trc)))
            logger.info("train avg_dice: {:.4f} ".format(train_dice / (len(train_set_loader)*opt.trc)))
            logger.info("test avg_dice: {:.4f} ".format(tal_dice / len(val_set_loader)))

            # checkpoint
            if best_dice < avg_dice:
                best_dice = avg_dice
                best_epoch = epoch

                a = []
                b = []
                a.append(best_dice)
                b.append(best_epoch)


            logger.info("best dice: {:.4f} \t best epoch: {}".format(best_dice, best_epoch))
            logger.info('*' * 100)

    epoch = range(0, opt.niter + opt.niter_decay)
    curr_time_1 = datetime.datetime.now()
    logger.info(f"计算时间:{curr_time_1 - curr_time}")
    return best_dice


curr_time = datetime.datetime.now()
# pathA = r'D:\NPC_project\data\data_256/'  # npc data_256_15
# pathA = './TG-NET/data/trainDataset'  # hecktor data_144_10
pathA = './data/dataset'
# pathB = 'E:\modal_conversion/dataB/'
# testA = 'E:\modal_conversion/testA/'
# testB = 'E:\modal_conversion/testB/'
save_result_folder = 'D:\\NPC\\results\compare_model'

# model = 'resunet'  # dunet dunet2 unet resunet attunet zhao DenseUnet CaNet CaSNet transunet unetpp transformerunet dtransunet MHCAUnet
mname = 'JBHI-H-dataE'
test_info = "head = 4  block = 4  mlp = 8 mlp droupout = 0.2 pd = 0.1 0.1 atd = 0.0ADam two to single y"



if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)



####################enhance###############
mr_transform = ex_transforms.Compose([
                                        # ex_transforms.Horizontal_Mirroring(p=0.5),
                                        # ex_transforms.Vertical_Mirroring(p=0.5),
                                        # ex_transforms.RandomScale(p=0.5),
                                        # ex_transforms.RandomRotation(p=0.5, angle_range=[-15, 15]),
                                        # ex_transforms.RandomTranslate(p=0.5, range=[-5, 5]),
                                        ex_transforms.NormalizeIntensity(),  # CT [-1, 1], SUV and Ki use mean and std
                                        ex_transforms.ToTensor()
                                        ])

train_transform = ex_transforms.Compose([
                                        # ex_transforms.RandomRotation(p=0.5, angle_range=[0, 45]),
                                        # ex_transforms.Mirroring(p=0.5),
                                        ex_transforms.NormalizeIntensity(),  # CT [-1, 1], SUV and Ki use mean and std
                                        ex_transforms.ToTensor()
                                        ])

val_transform = ex_transforms.Compose([
                                        ex_transforms.NormalizeIntensity(),
                                        ex_transforms.ToTensor()
                                    ])
# train_data = dualctdataset(train_paths[fold], transform= mr_transform)
# val_data = dualctdataset(val_paths[fold], transform=val_transform)

all_dices = []
#dataset size
for dr in [0.2,0.4,0.6,0.8,1.0]:

    train_data = HECKTORdataset(mode='train',transform=mr_transform,data_rate=dr)
    val_data = HECKTORdataset(mode='test',transform=val_transform,data_rate=dr)

    train_set_loader = DataLoader(dataset=train_data, num_workers=opt.threads, batch_size=opt.batch_size,
                                    shuffle=True, drop_last=True)
    val_set_loader = DataLoader(dataset=val_data, num_workers=opt.threads, batch_size=opt.test_batch_size,
                                shuffle=False, drop_last=True)

    logger.info(f"{len(train_set_loader)}, {len(val_set_loader)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('===> Building models')

    # dunet dunet2 unet resunet attunet zhao DenseUnet CaNet CaSNet transunet unetpp transformerunet dtransunet MHCAUnet
    nList = ['unet','resunet','attunet','DenseUnet','zhao','transunet','Liu','CaSNet']
    # nList = ['CaSNet']
    dices = []
    for i in nList:
        dice = testOne(i)
        dices.append(dice)
    all_dices.append(dices)
    print(all_dices)

#dataset enhance
    
# for de in [0,1,2,3,4,5]:

#     train_data = HECKTORdataset(mode='train',transform=mr_transform,data_rate=1.0,enhence_rate=de)
#     val_data = HECKTORdataset(mode='test',transform=val_transform,data_rate=1.0,enhence_rate=de)

#     train_set_loader = DataLoader(dataset=train_data, num_workers=opt.threads, batch_size=opt.batch_size,
#                                     shuffle=True, drop_last=True)
#     val_set_loader = DataLoader(dataset=val_data, num_workers=opt.threads, batch_size=opt.test_batch_size,
#                                 shuffle=False, drop_last=True)

#     logger.info(f"{len(train_set_loader)}, {len(val_set_loader)}")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info('===> Building models')

#     # dunet dunet2 unet resunet attunet zhao DenseUnet CaNet CaSNet transunet unetpp transformerunet dtransunet MHCAUnet
#     # nList = ['unet','resunet','attunet','DenseUnet','zhao','transunet','Liu','CaSNet']
#     nList = ['CaSNet']
#     dices = []
#     for i in nList:
#         dice = testOne(i)
#         dices.append(dice)
#     all_dices.append(dices)
#     print(all_dices)
