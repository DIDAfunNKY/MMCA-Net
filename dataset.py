import os
import random
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from natsort import natsorted
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
from tqdm import *
from skimage.transform import rotate 
import skimage.transform
import cv2


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-40, 40)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=1)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=1)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
        sample = {'image': image, 'label': label}
        return sample


class RandomGenerator1(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=3)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
        sample = {'image': image, 'label': label}
        return sample


class dualctdataset(Dataset):
    def __init__(self, baseA_dir, mode='train', transform=None):
        self.transform = transform  # using transform in torch!
        self.sample_list_A = [s for s in natsorted(baseA_dir)]
        # self.sample_list_B = [(baseB_dir + s) for s in natsorted(os.listdir(baseB_dir))]
        self.mode = mode

    def __len__(self):
        return len(self.sample_list_A)

    def __getitem__(self, idx):
        sample = dict()
        dataA_path = self.sample_list_A[idx]
        # sample['id'] = dataA_path.split('/')[-1].split('.')[0]  # npc
        sample['id'] = dataA_path.split('/')[-1].split('.')[0]  # hecktor
        # dataB_path =self.sample_list_B[idx]
        dataA = np.load(dataA_path)
        # dataB = np.load(dataB_path)
        if self.mode == 'train':
            # ct, suv, ki, label = dataA['ct'], dataA['suv'], dataA['ki'], dataA['label']  # npc
            ct, suv, label = dataA['ct'], dataA['suv'], dataA['label']  # npc
            # ct, suv, label = dataA['ct'], dataA['suv'], dataA['label']  # hecktor
            # input = np.stack([ct, suv, ki], axis=-1)  # ct+suv+ki
            input = np.stack([ct, suv], axis=-1)   # ct + suv
            # input = np.expand_dims(ki, axis=2)    #ki
            # input = np.expand_dims(suv, axis=2)   #suv
            # input = np.expand_dims(ct, axis=2)    #ct
            label = np.expand_dims(label, axis=2)
            sample['input'] = input
            sample['target'] = label
            if self.transform:
                sample = self.transform(sample)
            return sample
        # if self.mode =='test':
        #     image, label = dataA['data'], dataB['data']
        #     label_mean = dataB['mean']
        #     label_std = dataB['std']
        #     label_min = dataB['min_val']
        #     label_max = dataB['max_val']
        #     sample = {'image': image,
        #               'label': label, 'label_mean': label_mean,'label_std': label_std,'label_min':label_min,'label_max':label_max}
        #     return sample




def standerlize(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / (std + 1e-3)  # pet 归一化  

def upsample(input,target):
    newdata = np.zeros((256,256,2))
    newT = np.zeros((256,256))
    newdata[:,:,0] = cv2.resize(input[:,:,0], (256, 256), interpolation=cv2.INTER_CUBIC)
    newdata[:,:,1] = cv2.resize(input[:,:,1], (256, 256), interpolation=cv2.INTER_CUBIC)
    newT = cv2.resize(target, (256, 256), interpolation=cv2.INTER_NEAREST)
    return newdata,newT


class HECKTORdataset(Dataset):
    def __init__(self,  mode='train', transform=None,data_rate = 1.0,enhence_rate = 0):
        self.transform = transform  # using transform in torch!
        self.mode = mode
        self.inputs = []
        self.targets = []
        data = np.load('./HECKTOR.npz')

        np.random.seed(42)  
        random.seed(42)
        n_examples = len(data['input'])
        # n_examples = 1000
        n_train = n_examples * 0.8  
        train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
        test_idx = list(set(range(0,n_examples))-set(train_idx))
        n = int(np.round(len(train_idx)) * data_rate)
        train_idx = train_idx[:n]

        print(n_examples,len(train_idx),len(test_idx))
        if self.mode == 'train':
            inputs_cases = data['input'][train_idx]
            targets_cases = data['target'][train_idx]
            # inputs_cases = data['input'][0:180]
            # targets_cases = data['target'][0:180]
            # print(inputs_cases.shape,targets_cases.shape)
            # plt.imshow(inputs_cases[0,0,:,:,0])
            # plt.show()
            # plt.imshow(inputs_cases[0,:,:,0,0])
            # plt.show()
            qbar = trange(len(inputs_cases))
            for i in range(len(inputs_cases)):
                for j in range(len(inputs_cases[i])):
                    tz = np.sum(targets_cases[i,:,:,j])
                    if tz > 0:
                        #original
                        input = inputs_cases[i,:,:,j,:]
                        target = targets_cases[i,:,:,j]
                        self.inputs.append(input.copy())
                        self.targets.append(target.copy())
                        # input[:, :, 0] = np.clip(standerlize(input[:, :, 0]),0,1)  # ct
                        # input[:, :, 1] = np.clip(standerlize(input[:, :, 1]),0,1)  # pt
                        #
                        # inputS,targetS = upsample(input,target)
                        # self.inputs.append(inputS.copy())
                        # self.targets.append(targetS.copy())

                        if tz > 0:
                            if enhence_rate >= 1 :
                                #flip1
                                input1 = input.copy()
                                target1 = target.copy()
                                input1[:, :, 0] = np.flip(input1[:, :, 0],axis=1) # CT
                                input1[:, :, 1] = np.flip(input1[:, :, 1],axis=1) # pet
                                target1 = np.flip(target1,axis=1) 
                                # o
                                self.inputs.append(input1.copy())
                                self.targets.append(target1.copy())
                                # 256
                                # inputS,targetS = upsample(input1,target1)
                                # self.inputs.append(inputS.copy())
                                # self.targets.append(targetS.copy())
                            if enhence_rate >= 2 :
                                #flip2
                                input2 = input.copy()
                                target2 = target.copy()
                                input2[:, :, 0] = np.flip(input2[:, :, 0],axis=0) # CT
                                input2[:, :, 1] = np.flip(input2[:, :, 1],axis=0) # pet
                                target2 = np.flip(target2,axis=0) 
                                self.inputs.append(input2.copy())
                                self.targets.append(target2.copy())
                                # 256
                                # inputS,targetS = upsample(input2,target2)
                                # self.inputs.append(inputS.copy())
                                # self.targets.append(targetS.copy())
                            if enhence_rate >= 3:
                                #flip12
                                input3 = input1.copy()
                                target3 = target1.copy()
                                input3[:, :, 0] = np.flip(input3[:, :, 0],axis=0) # CT
                                input3[:, :, 1] = np.flip(input3[:, :, 1],axis=0) # pet
                                target3 = np.flip(target3,axis=0) 
                                self.inputs.append(input3.copy())
                                self.targets.append(target3.copy())
                                # 256
                                # inputS,targetS = upsample(input3,target3)
                                # self.inputs.append(inputS.copy())
                                # self.targets.append(targetS.copy())
                            if enhence_rate >= 4:
                                #scale
                                input4 = input.copy()
                                target4 = target.copy()
                                s = random.random()
                                tf = skimage.transform.AffineTransform(scale=1+s/2.0)
                                input4[:, :, 0] = skimage.transform.warp(input4[:, :, 0], inverse_map=tf.inverse)
                                input4[:, :, 1] = skimage.transform.warp(input4[:, :, 1], inverse_map=tf.inverse)
                                target4 = skimage.transform.warp(target4, inverse_map=tf.inverse)
                                self.inputs.append(input4.copy())
                                self.targets.append(target4.copy())
                                # 256
                                # inputS,targetS = upsample(input4,target4)
                                # self.inputs.append(inputS.copy())
                                # self.targets.append(targetS.copy())
                            if enhence_rate >= 5:
                                #rotate
                                input5 = input.copy()
                                target5 = target.copy()
                                angle = random.randrange(-30, 30,1)
                                input5[:, :, 0] = rotate(input5[:, :, 0],angle) # CT
                                input5[:, :, 1] = rotate(input5[:, :, 1],angle) # pet
                                target5 = rotate(target5,angle) # mask
                                self.inputs.append(input5.copy())
                                self.targets.append(target5.copy())
                                # 256
                                # inputS,targetS = upsample(input5,target5)
                                # self.inputs.append(inputS.copy())
                                # self.targets.append(targetS.copy())
                            # #translate
                            # input6 = input.copy()
                            # target6 = target.copy()
                            # x = random.randrange(-20,20)
                            # y = random.randrange(-20,20)
                            # s = random.random()
                            # # tf = skimage.transform.AffineTransform(translation=(x,y))
                            # tf = skimage.transform.AffineTransform(translation=(x,y),scale=1+s/2.0)
                            # input6[:, :, 0] = skimage.transform.warp(input6[:, :, 0], inverse_map=tf.inverse)
                            # input6[:, :, 1] = skimage.transform.warp(input6[:, :, 1], inverse_map=tf.inverse)
                            # target6 = skimage.transform.warp(target6, inverse_map=tf.inverse)
                            # self.inputs.append(input6.copy())
                            # self.targets.append(target6.copy())
                            # #SR
                            # input7 = input4.copy()
                            # target7 = target4.copy()
                            # angle = random.randrange(-45, 45,1)
                            # input7[:, :, 0] = rotate(input7[:, :, 0],angle) # CT
                            # input7[:, :, 1] = rotate(input7[:, :, 1],angle) # pet
                            # target7 = rotate(target7,angle) # mask
                            # self.inputs.append(input7.copy())
                            # self.targets.append(target7.copy())
                qbar.update(1)
        else:
            inputs_cases = data['input'][test_idx]
            targets_cases = data['target'][test_idx]
            # inputs_cases = data['input'][180:]
            # targets_cases = data['target'][180:]
            qbar1 = trange(len(inputs_cases))
            for i in range(len(inputs_cases)):
                for j in range(len(inputs_cases[i])):
                    if np.sum(targets_cases[i,:,:,j]) > 0:
                        input = inputs_cases[i,:,:,j,:]
                        target = targets_cases[i,:,:,j]

                        self.inputs.append(input )
                        self.targets.append(target)
                        # inputS,targetS = upsample(input,target)
                        # self.inputs.append(inputS.copy())
                        # self.targets.append(targetS.copy())
                qbar1.update(1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = dict()

        sample['id'] = idx  # hecktor

        label = np.expand_dims(self.targets[idx], axis=2)
        sample['input'] = self.inputs[idx].transpose([1, 0, 2])
        sample['target'] = label.transpose([1, 0, 2])
        if self.transform:
            sample = self.transform(sample)
        return sample

