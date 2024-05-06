import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import rotate 
import skimage.transform


class Compose:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class ToTensor:
    def __init__(self, mode='train'):
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target']
            img = np.transpose(img, axes=[2, 0, 1])
            mask = np.transpose(mask, axes=[2, 0, 1])
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
            sample['input'], sample['target'] = img, mask

        else:  # if self.mode == 'test'
            img = sample['input']
            img = np.transpose(img, axes=[2, 0, 1])
            img = torch.from_numpy(img).float()
            sample['input'] = img

        return sample


class RandomTranslate:
    # 随机平移
    def __init__(self, p=0.5,range = 10):
        self.p = p
        self.range = range

    def __call__(self, sample):
        if random.random() < self.p:
            # print("horiaontal enhance!")
            img, mask = sample['input'], sample['target']
            # img = img.detach().numpy()
            # mask = mask.detach().numpy()
            # img[:, :, 0] = np.flip(img[:, :, 0],axis=1) # CT
            # img[:, :, 1] = np.flip(img[:, :, 1],axis=1) # pet
            # # img[:, :, 2] = np.flip(img[:, :, 2],axis=1) # ki
            # mask = np.flip(mask,axis=1) # mask

            x = random.randrange(*self.range)
            y = random.randrange(*self.range)
            s = random.random()
            # tf = skimage.transform.AffineTransform(translation=(x,y))
            tf = skimage.transform.AffineTransform(translation=(x,y),scale=1+s/2.0)
            img[:, :, 0] = skimage.transform.warp(img[:, :, 0], inverse_map=tf.inverse)
            img[:, :, 1] = skimage.transform.warp(img[:, :, 1], inverse_map=tf.inverse)
            mask = skimage.transform.warp(mask, inverse_map=tf.inverse)

            # for i in range(img.shape[0]):
            #     img_x = img[i, :, :, :]
            #     mask_x = mask[i, :, :, :]
            #     img[i, :, :, :] = np.flip(img_x, axis=2)
            #     mask[i, :, :, :] = np.flip(mask_x, axis=2)
            # img = torch.from_numpy(img)
            # mask = torch.from_numpy(mask)
            # plt.figure()
            # plt.subplot(1,3,1)
            # plt.imshow(img[:, :, 0])
            # plt.subplot(1,3,2)
            # plt.imshow(img[:, :, 1])
            # plt.subplot(1,3,3)
            # plt.imshow(mask)
            # plt.show()

            sample['input'], sample['target'] = img.copy(), mask.copy()
        return sample
    
class RandomScale:
    # 随机缩放
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:

            img, mask = sample['input'], sample['target']


            s = random.random()
            tf = skimage.transform.AffineTransform(scale=1+s/4.0)
            img[:, :, 0] = skimage.transform.warp(img[:, :, 0], inverse_map=tf.inverse)
            img[:, :, 1] = skimage.transform.warp(img[:, :, 1], inverse_map=tf.inverse)
            mask = skimage.transform.warp(mask, inverse_map=tf.inverse)

            sample['input'], sample['target'] = img.copy(), mask.copy()
        return sample

class Horizontal_Mirroring:
    # 水平翻转
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            # print("horiaontal enhance!")
            img, mask = sample['input'], sample['target']
            # img = img.detach().numpy()
            # mask = mask.detach().numpy()
            img[:, :, 0] = np.flip(img[:, :, 0],axis=1) # CT
            img[:, :, 1] = np.flip(img[:, :, 1],axis=1) # pet
            # img[:, :, 2] = np.flip(img[:, :, 2],axis=1) # ki
            mask = np.flip(mask,axis=1) # mask

            # for i in range(img.shape[0]):
            #     img_x = img[i, :, :, :]
            #     mask_x = mask[i, :, :, :]
            #     img[i, :, :, :] = np.flip(img_x, axis=2)
            #     mask[i, :, :, :] = np.flip(mask_x, axis=2)
            # img = torch.from_numpy(img)
            # mask = torch.from_numpy(mask)

            sample['input'], sample['target'] = img.copy(), mask.copy()
        return sample


class Vertical_Mirroring:
    # 垂直翻转
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            # print("veritical enhance!")
            img, mask = sample['input'], sample['target']
            # img = img.detach().numpy()
            # mask = mask.detach().numpy()
            # for i in range(img.shape[0]):
            #     img_x = img[i, :, :, :]
            #     mask_x = mask[i, :, :, :]
            #     img[i, :, :, :] = np.flip(img_x, axis=1)
            #     mask[i, :, :, :] = np.flip(mask_x, axis=1)
            # img = torch.from_numpy(img)
            # mask = torch.from_numpy(mask)
            img[:, :, 0] = np.flip(img[:, :, 0],axis=0) # CT
            img[:, :, 1] = np.flip(img[:, :, 1],axis=0) # pet
            # img[:, :, 2] = np.flip(img[:, :, 2],axis=0) # ki
            mask = np.flip(mask,axis=0) # mask

            # plt.figure()
            # plt.subplot(1,3,1)
            # plt.imshow(img[:, :, 0])
            # plt.subplot(1,3,2)
            # plt.imshow(img[:, :, 1])
            # plt.subplot(1,3,3)
            # plt.imshow(mask)
            # plt.show()

            sample['input'], sample['target'] = img.copy(), mask.copy()
        return sample


class NormalizeIntensity:

    def __call__(self, sample):
        img = sample['input']

        img[:, :, 0] = self.normalize_ct(img[:, :, 0])  # ct
        img[:, :, 0] = self.standerlize(img[:, :, 0])  # ct
        img[:, :, 1] = self.standerlize(img[:, :, 1])  # pt
        # img[:, :, 2] = self.normalize_ki(img[:, :, 2])  # ki
        # img = self.normalize_suv(img)

        sample['input'] = img

        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(img[:, :, 0])
        # plt.subplot(1,3,2)
        # plt.imshow(img[:, :, 1])
        # plt.subplot(1,3,3)
        # plt.imshow(sample['target'])
        # plt.show()
        return sample

    # @staticmethod
    # def normalize_ct(img):
    #     Max = np.max(img)
    #     Min = np.min(img)
    #     M = max(abs(Max), abs(Min))

    #     norm_img = np.clip(img, Min, Max) / M  # ct 归一化[-1, 1]

    #     return norm_img

    @staticmethod
    def normalize_ct(img):
        norm_img = np.clip(img, -200, 200)  # ct 归一化[-1, 1]   hecktor
        # norm_img = np.clip(img, -1000, 1000) / 1000  # ct 归一化[-1, 1]  sts
        return norm_img  

    @staticmethod
    def standerlize(img):
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-3)  # pet 归一化  

    @staticmethod
    def normalize_pet(img):
        Max = np.max(img)
        Min = np.min(img)
        return (img - Min) / ((Max - Min) + 1e-4)  #归一化

    @staticmethod
    def normalize_suv(img):
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-4)  # pet 归一化

    @staticmethod
    def normalize_sPET(img):
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-3)  # pet 归一化
        # Max = np.max(img)
        # Min = np.min(img)
        # return (img - Min) / Max

    @staticmethod
    def normalize_ki(img):
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-3)  # pet 归一化
        # Max = np.max(img)
        # Min = np.min(img)
        # return (img - Min) / Max


class RandomRotation:
    def __init__(self, p=0.5, angle_range=[5, 15]):
        self.p = p
        self.angle_range = angle_range

    def __call__(self, sample):
        if random.random() < self.p:

            img, mask = sample['input'], sample['target']
            angle = random.randrange(*self.angle_range)  #

            # img = img.detach().numpy()
            # mask = mask.detach().numpy()
            # for i in range(img.shape[0]):
            #     img_x = img[i, :, :, :]
            #     mask_x = mask[i, :, :, :]
            #     # plt.figure()
            #     # plt.subplot(131)
            #     # plt.imshow(img[i, 0, :, :], cmap='gray')
            #     # plt.subplot(132)
            #     # plt.imshow(img[i, 1, :, :], cmap='gray')
            #     # plt.subplot(133)
            #     # plt.imshow(mask[i, 0, :, :], cmap='gray')
            #     # plt.show()
            #     for k in range(img.shape[1]):
            #         img[i, k, :, :] = rotate(img_x[k, :, :], angle)
            #     mask[i, 0, :, :] = rotate(mask_x[0, :, :], angle)
            #     # plt.figure()
            #     # plt.subplot(131)
            #     # plt.imshow(img[i, 0, :, :], cmap='gray')
            #     # plt.subplot(132)
            #     # plt.imshow(img[i, 1, :, :], cmap='gray')
            #     # plt.subplot(133)
            #     # plt.imshow(mask[i, 0, :, :], cmap='gray')
            #     # plt.show()
            # img = torch.from_numpy(img)
            # mask = torch.from_numpy(mask)
            img[:, :, 0] = rotate(img[:, :, 0],angle) # CT
            img[:, :, 1] = rotate(img[:, :, 1],angle) # pet
            # img[:, :, 2] = rotate(img[:, :, 2],angle) # ki
            mask = rotate(mask,angle) # mask

            # plt.figure()
            # plt.subplot(1,3,1)
            # plt.imshow(img[:, :, 0],cmap='gray')
            # plt.subplot(1,3,2)
            # plt.imshow(img[:, :, 1], cmap='gist_yarg')
            # plt.subplot(1,3,3)
            # plt.imshow(mask,cmap='gray')
            # plt.show()
        
            sample['input'], sample['target'] = img, mask
        return sample

    @staticmethod
    def rotate_3d_along_axis(img, angle, axis, order):

        if axis == 0:
            rot_img = rotate(img, angle, order=order, preserve_range=True)

        if axis == 1:
            rot_img = np.transpose(img, axes=(1, 2, 0))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(2, 0, 1))

        if axis == 2:
            rot_img = np.transpose(img, axes=(2, 0, 1))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(1, 2, 0))

        return rot_img


class ZeroPadding:

    def __init__(self, target_shape, mode='train'):
        self.target_shape = np.array(target_shape)  # without channel dimension
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train':
            img, mask = sample['input'], sample['target']

            input_shape = np.array(img.shape[:-1])  # last (channel) dimension is ignored
            d_x, d_y, d_z = self.target_shape - input_shape
            d_x, d_y, d_z = int(d_x), int(d_y), int(d_z)

            if not all(i == 0 for i in (d_x, d_y, d_z)):
                positive = [i if i > 0 else 0 for i in (d_x, d_y, d_z)]
                negative = [i if i < 0 else None for i in (d_x, d_y, d_z)]

                # padding for positive values:
                img = np.pad(img, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))
                mask = np.pad(mask, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))

                # cropping for negative values:
                img = img[: negative[0], : negative[1], : negative[2], :].copy()
                mask = mask[: negative[0], : negative[1], : negative[2], :].copy()

                assert img.shape[:-1] == mask.shape[:-1], f'Shape mismatch for the image {img.shape[:-1]} and mask {mask.shape[:-1]}'

                sample['input'], sample['target'] = img, mask

            return sample

        else:  # if self.mode == 'test'
            img = sample['input']

            input_shape = np.array(img.shape[:-1])  # last (channel) dimension is ignored
            d_x, d_y, d_z = self.target_shape - input_shape
            d_x, d_y, d_z = int(d_x), int(d_y), int(d_z)

            if not all(i == 0 for i in (d_x, d_y, d_z)):
                positive = [i if i > 0 else 0 for i in (d_x, d_y, d_z)]
                negative = [i if i < 0 else None for i in (d_x, d_y, d_z)]

                # padding for positive values:
                img = np.pad(img, ((0, positive[0]), (0, positive[1]), (0, positive[2]), (0, 0)), 'constant', constant_values=(0, 0))

                # cropping for negative values:
                img = img[: negative[0], : negative[1], : negative[2], :].copy()

                sample['input'] = img

            return sample


class ExtractPatch:
    """Extracts a patch of a given size from an image (4D numpy array)."""

    def __init__(self, patch_size, p_tumor=0.5):
        self.patch_size = patch_size  # without channel dimension!
        self.p_tumor = p_tumor  # probs to extract a patch with a tumor

    def __call__(self, sample):
        img = sample['input']
        mask = sample['target']

        assert all(x <= y for x, y in zip(self.patch_size, img.shape[:-1])), \
            f"Cannot extract the patch with the shape {self.patch_size} from  " \
                f"the image with the shape {img.shape}."

        # patch_size components:
        ps_x, ps_y, ps_z = self.patch_size

        if random.random() < self.p_tumor:
            # coordinates of the tumor's center:
            xs, ys, zs, _ = np.where(mask != 0)
            tumor_center_x = np.min(xs) + (np.max(xs) - np.min(xs)) // 2
            tumor_center_y = np.min(ys) + (np.max(ys) - np.min(ys)) // 2
            tumor_center_z = np.min(zs) + (np.max(zs) - np.min(zs)) // 2

            # compute the origin of the patch:
            patch_org_x = random.randint(tumor_center_x - ps_x, tumor_center_x)
            patch_org_x = np.clip(patch_org_x, 0, img.shape[0] - ps_x)

            patch_org_y = random.randint(tumor_center_y - ps_y, tumor_center_y)
            patch_org_y = np.clip(patch_org_y, 0, img.shape[1] - ps_y)

            patch_org_z = random.randint(tumor_center_z - ps_z, tumor_center_z)
            patch_org_z = np.clip(patch_org_z, 0, img.shape[2] - ps_z)
        else:
            patch_org_x = random.randint(0, img.shape[0] - ps_x)
            patch_org_y = random.randint(0, img.shape[1] - ps_y)
            patch_org_z = random.randint(0, img.shape[2] - ps_z)

        # extract the patch:
        patch_img = img[patch_org_x: patch_org_x + ps_x,
                        patch_org_y: patch_org_y + ps_y,
                        patch_org_z: patch_org_z + ps_z, :].copy()

        patch_mask = mask[patch_org_x: patch_org_x + ps_x,
                          patch_org_y: patch_org_y + ps_y,
                          patch_org_z: patch_org_z + ps_z, :].copy()

        assert patch_img.shape[:-1] == self.patch_size, \
            f"Shape mismatch for the patch with the shape {patch_img.shape[:-1]}, " \
                f"whereas the required shape is {self.patch_size}."

        sample['input'] = patch_img
        sample['target'] = patch_mask

        return sample


class InverseToTensor:
    def __call__(self, sample):
        output = sample['output']

        output = torch.squeeze(output)  # squeeze the batch and channel dimensions
        output = output.numpy()

        sample['output'] = output
        return sample


class CheckOutputShape:
    def __init__(self, shape=(144, 144, 144)):
        self.shape = shape

    def __call__(self, sample):
        output = sample['output']
        assert output.shape == self.shape, \
            f'Received wrong output shape. Must be {self.shape}, but received {output.shape}.'
        return sample


class ProbsToLabels:
    def __call__(self, sample):
        # output = sample['output']
        output = sample
        output = (output > 0.5).astype(int)  # get binary label
        # sample['output'] = output
        sample = output
        return sample

