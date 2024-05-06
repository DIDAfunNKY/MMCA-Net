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
from medpy.metric.binary import hd,hd95

from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
CUDA_LAUNCH_BLOCKING = 1

curr_time = datetime.datetime.now()

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--batch_size', type=int, default=144, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=144, help='testing batch size')
parser.add_argument('--input_nc', type=int, default=2, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=2, help='generator filters in first conv layer')
# parser.add_argument('--ndf', type=int, default=4, help='discriminator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=20, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=1, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb1', type=int, default=0.001, help='weight on L1 term in objective') #0.01
parser.add_argument('--lamb2', type=int, default=0.01, help='weight on L2 term in objective')  #0.1
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


def load_checkpoint(checkpoint_file, model):
    logger.info("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda:0")
    # 预训练模型 要求模型一模一样，每一层的写法和命名都要一样 本质一样都不行
    # 完全一样的模型实现，也可能因为load和当初save时 pytorch版本不同 而导致state_dict中的key不一样
    # 例如 "initial.0.weight" 与 “initial.weight” 的区别
    model.load_state_dict(checkpoint["state_dict"], strict=False)  # 改成strict=False才能编译通过
    # optimizer.load_state_dict(checkpoint["optimizer"])


class WGAN_VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(WGAN_VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        out = self.feature_extractor(x)
        return out
    

def hausdorff_distance(image0, image1):
    """Code copied from 
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py#L7-L54
    for compatibility reason with python 3.6
    """
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))


if __name__ == '__main__':
    curr_time = datetime.datetime.now()
    # pathA = r'D:\NPC_project\data\data_256/'  # npc data_256_15
    # pathA = './TG-NET/data/trainDataset'  # hecktor data_144_10
    pathA = './data/dataset'
    # pathB = 'E:\modal_conversion/dataB/'
    # testA = 'E:\modal_conversion/testA/'
    # testB = 'E:\modal_conversion/testB/'
    save_result_folder = 'D:\\NPC\\results\compare_model'

    model = 'CaSNet'  # dunet dunet2 unet resunet attunet zhao DenseUnet CaNet CaSNet transunet unetpp transformerunet dtransunet MHCAUnet
    mname = 'JBHI-HS'
    test_info = "head = 4  block = 4  mlp = 8 mlp droupout = 0.2 pd = 0.1 0.1 atd = 0.0ADam two to single y"
    save_model = os.path.join(save_result_folder, mname+"-"+model)
    if not os.path.exists(save_model):
        os.mkdir(save_model)
    fold = 0
    save_folder = os.path.join(save_model, 're_' + str(fold))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    logger.add(f"log/{mname}-{model}-{fold}.log")
    logger.info(test_info)
    logger.info(opt)
    logger.info('===> Loading datasets')

    # ***************** 五折交叉验证 *******************
    all_paths = [os.path.join(pathA, s) for s in natsorted(os.listdir(pathA))]
    folder = KFold(n_splits=5, random_state=42, shuffle=True)
    train_paths = []  # 存放5折的训练集划分
    val_paths = []  # 存放5折的验证集划分

    #导出cases
    case_names = []
    for d in all_paths:
        case_name = d.split('_')[0]
        if case_name not in case_names:
            # print(case_name)
            case_names.append(case_name)

    #划分训练cases 验证cases
    train_cases = []
    val_cases = []
    for k, (Trindex, Tsindex) in enumerate(folder.split(case_names)):
        train_cases.append(np.array(case_names)[Trindex].tolist())
        val_cases.append(np.array(case_names)[Tsindex].tolist())
    #将指定cases的切片放到paths中
    for i in range(5):
        train_paths.append([])
        val_paths.append([])
        for p in all_paths:
            case_name = p.split('_')[0]
            # print(case_name)
            if case_name in train_cases[i]:
                train_paths[i].append(p)
            elif case_name in val_cases[i]:
                if '-' not in p:
                    val_paths[i].append(p)
            else:
                print("error!")

    df = pd.DataFrame(data=train_paths, index=['0', '1', '2', '3', '4'])
    df.to_csv(os.path.join(save_result_folder, 'train.csv'))
    df1 = pd.DataFrame(data=val_paths, index=['0', '1', '2', '3', '4'])
    df1.to_csv(os.path.join(save_result_folder, 'val.csv'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('===> Building models')
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
    
    # load_checkpoint(r'D:\NPC\results\compare_model\JBHI-H-unet\re_0\best_model\net_model.pth',net_g)

    ####################enhance###############
    mr_transform = ex_transforms.Compose([
                                            # ex_transforms.Horizontal_Mirroring(p=0.5),
                                            # ex_transforms.Vertical_Mirroring(p=0.5),
                                            ex_transforms.RandomScale(p=0.5),
                                            ex_transforms.RandomRotation(p=0.5, angle_range=[-15, 15]),
                                            ex_transforms.RandomTranslate(p=0.5, range=[-5, 5]),
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

    train_data = HECKTORdataset(mode='train',transform=train_transform,enhence_rate=5)
    val_data = HECKTORdataset(mode='test',transform=val_transform,enhence_rate=5)

    train_set_loader = DataLoader(dataset=train_data, num_workers=opt.threads, batch_size=opt.batch_size,
                                  shuffle=True, drop_last=True)
    val_set_loader = DataLoader(dataset=val_data, num_workers=opt.threads, batch_size=opt.test_batch_size,
                                shuffle=False, drop_last=True)

    logger.info(f"{len(train_set_loader)}, {len(val_set_loader)}")

   
    
    # 判别器 discriminator
    net_d = define_D(opt.output_nc, opt.ndf, 'pixel', gpu_id=device,use_sigmoid=True)
    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    def compute_MSE(img1, img2):
        return ((img1 - img2) ** 2).mean()

    def compute_RMSE(img1, img2):
        if type(img1) == torch.Tensor:
            return torch.sqrt(compute_MSE(img1, img2)).item()
        else:
            return np.sqrt(compute_MSE(img1, img2))

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # optimizer_g = optim.SGD(net_g.parameters(),momentum=0.9,weight_decay=1e-4,lr=opt.glr)

    net_g_scheduler = get_scheduler(optimizer_g, opt)
    net_d_scheduler = get_scheduler(optimizer_d, opt)
    ganIterations = 0
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
                # ct =real_a.cpu().numpy()[0, 0, :, :]
                # suv = real_a.cpu().numpy()[0, 1, :, :]
                # label = real_b.cpu().numpy()[0, 0, :, :]

                # plt.figure()
                # plt.subplot(131)
                # plt.imshow(ct, cmap='gray')
                # plt.subplot(132)
                # plt.imshow(suv, cmap='gist_yarg')
                # plt.subplot(133)
                # plt.imshow(label, cmap='gray')
                # plt.show()
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
                    #output
                    # if iteration % 10 == 0:
                    #     logger.info("Epoch[{}]({}/{}): Loss_G: {:.4f} dice: {:.4f}".format(
                    #         epoch, len(train_set_loader)*i + iteration, len(train_set_loader)*opt.trc, loss_g.item(), dice.item()))
                    qbar.set_postfix(epoch = epoch,loss=loss_g.item(), dice=dice.item())  # 进度条右边显示信息
                    qbar.update(1)
                else:
                    # GAN
                    fake_b1 = fake_b.detach().cpu().numpy()[0, 0, ...]
                    real_b1 = real_b.detach().cpu().numpy()[0, 0, ...]
                    # print(fake_b1.max(), fake_b1.min())
                    psnr2 = psnr(real_b1, fake_b1)
                    RMSE = compute_RMSE(real_b1, fake_b1)
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
        tal_hd = 0
        net_g.eval()
        inputs = []
        targets = []
        preds = []

        pred_save = os.path.join(save_folder, 'pred')
        if not os.path.exists(pred_save):
            os.mkdir(pred_save)
        pred_save_folder = os.path.join(pred_save, str(epoch))
        if not os.path.exists(pred_save_folder):
            os.mkdir(pred_save_folder)
        with torch.no_grad():
            num = 1
            for sample in val_set_loader:
                input, target = sample['input'].to(device), sample['target'].to(device)

                # ct = input.cpu().numpy()[0, 0, :, :]
                # suv = input.cpu().numpy()[0, 1, :, :]
                # label = target.cpu().numpy()[0, 0, :, :]

                # plt.figure()
                # plt.subplot(131)
                # plt.imshow(ct, cmap='gray')
                # plt.subplot(132)
                # plt.imshow(suv, cmap='gist_yarg')
                # plt.subplot(133)
                # plt.imshow(label, cmap='gray')
                # plt.show()

                id = sample['id']
                # print(id)
                prediction = net_g(input)
                pred = prediction.detach().cpu().numpy()[0, 0, :, :]
                tar = target.detach().cpu().numpy()[0, 0, :, :]
                pred_threshold = ex_transforms.ProbsToLabels()(pred)
                # prediction = torch.from_numpy(pred_threshold).to(device)

                inputs.append(input.cpu().numpy())
                preds.append(prediction.cpu().numpy())
                targets.append(target.cpu().numpy())


                dice_pred = metrics.dice(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()), target.cpu().numpy())
                try:
                    hd = hd95(ex_transforms.ProbsToLabels()(prediction.cpu().numpy()), target.cpu().numpy())
                except:
                    hd = 0
                tal_dice += dice_pred
                tal_hd += hd

                # logger.info("Epoch[{}]({}/{}): dice: {:.4f}".format(epoch, num, len(val_set_loader), dice_pred.item()))
                num += 1
            avg_dice = (tal_dice / len(val_set_loader))
            avg_hd = (tal_hd / len(val_set_loader))
            # avg_dice = avg_dice.detach().cpu().numpy()
            # Dice.append(avg_dice)
            Dice.append(avg_dice)
            Loss.append(train_loss / (len(train_set_loader)*opt.trc))
            TDice.append(train_dice / (len(train_set_loader)*opt.trc))
            logger.info('*' * 100)
            logger.info("train avg_loss: {:.4f} ".format(train_loss / (len(train_set_loader)*opt.trc)))
            logger.info("train avg_dice: {:.4f} ".format(train_dice / (len(train_set_loader)*opt.trc)))
            logger.info("test avg_dice: {:.4f} ".format(avg_dice))
            logger.info("test hd: {:.4f} ".format(avg_hd))
            # if avg_dice > best_dice:
            #     best_dice = avg_dice
            #     best_epoch = epoch
            # logger.info("best dice: {:.4f} \t best epoch: {}".format(best_dice, best_epoch))
            # logger.info('*' * 100)

            # checkpoint
            if best_dice < avg_dice:
                best_dice = avg_dice
                best_epoch = epoch
                model_save = os.path.join(save_folder, 'best_model')
                if not os.path.exists(model_save):
                    os.mkdir(model_save)
                net_g_model_out_path = os.path.join(model_save, 'net_model.pth')
                torch.save(net_g, net_g_model_out_path)
                logger.info("Checkpoint saved to {}".format("checkpoint"))

                filenameg = model_save + '/' + "generator.tar"
                save_checkpoint(net_g, optimizer_g, filename=filenameg)
                a = []
                b = []
                a.append(best_dice)
                b.append(best_epoch)
                summary = pd.DataFrame({
                    'best_dice': a,
                    'best_epoch': b
                })
                file = os.path.join(save_folder, 'summary.csv')
                summary.to_csv(file, index=False, sep=';')
                file = os.path.join(save_folder, 'res')
                np.savez(file, input = inputs, pred=preds, target=targets)

            logger.info("best dice: {:.4f} \t best epoch: {}".format(best_dice, best_epoch))
            logger.info('*' * 100)

    epoch = range(0, opt.niter + opt.niter_decay)
    curr_time_1 = datetime.datetime.now()
    logger.info(f"计算时间:{curr_time_1 - curr_time}")
    # np.savez(os.path.join(save_folder, 'loss_dice.npz'), Loss=Loss.cpu().numpy(), TDice=TDice.cpu().numpy(), Dice=Dice.cpu().numpy())
    plt.figure()
    plt.subplot(131)
    plt.plot(epoch, Loss)
    plt.subplot(132)
    # plt.plot(epoch, TDice)
    # plt.subplot(133)
    plt.plot(epoch, Dice)
    plt.savefig(os.path.join(save_folder, 'loss_dice.png'))
    plt.show()



