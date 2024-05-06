import numpy as np

import matplotlib.pyplot as plt

from surface_distance import compute_surface_distances, compute_robust_hausdorff

from scipy.spatial import cKDTree

from medpy.metric.binary import hd,hd95

import cv2

def robust_hausdorff(image0, image1, spacing, percent=95.0):
    # image0 = (image0 / 255).astype(np.uint8)
    surface_distances = compute_surface_distances(image0 != 0, image1 != 0,
                                                  spacing)
    return compute_robust_hausdorff(surface_distances, percent)


def VOE_RVD_Recall_Precision(mask_gt, mask_seg):
    tp = np.sum(np.logical_and(mask_gt, mask_seg))
    fn = np.sum(mask_gt) - tp
    fp = np.sum(mask_seg) - tp
    Recall = tp / (tp + fn)
    if tp + fp == 0:
        fp = 0.0001
        tp = 0.000001
    Precision = tp / (tp + fp)
    VOE = 1 - 2 * (np.sum(np.logical_and(mask_gt, mask_seg)) / (np.sum(mask_gt) + np.sum(mask_seg)))
    RVD = np.absolute((np.sum(mask_seg) / np.sum(mask_gt)) - 1)
    if np.linalg.norm(mask_seg) == 0:
        SCOS = 0.00
    else:
        SCOS = np.sum(np.dot(mask_gt,mask_seg) / (np.linalg.norm(mask_gt) * np.linalg.norm(mask_seg)))

    return Recall, Precision, VOE, RVD


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
        return 0 if len(b_points) == 0 else 50
    elif len(b_points) == 0:
        return 50

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))


def dice(mask_gt, mask_seg):
    # print(mask_gt,mask_seg)
    # print(np.logical_and(mask_gt, mask_seg))
    return 2 * np.sum(np.logical_and(
        mask_gt, mask_seg)) / (np.sum(mask_gt) + np.sum(mask_seg)+1e-8)

def load_pred(path):
    data = np.load(path)['pred']

    preds = []

    for i in range(len(data)):
        for j in range(len(data[i])):
            preds.append(data[i][j])

    return preds


path = r'D:\NPC\results\compare_model\JBHI-H-CaSNet\re_0\res.npz'




data = np.load(path)

inputx = data['input']
predx = data['pred']
targetx = data['target']

inputs = []
preds = []
targets = []

predA = load_pred(r'D:\NPC\results\compare_model\JBHI-H-sam-med-B-Sam\re_0\res.npz')
predB = load_pred(r'D:\NPC\results\compare_model\JBHI-H-sam-L-Sam\re_0\res.npz')
predC = load_pred(r'D:\NPC\results\compare_model\JBHI-H-sam-B-Sam\re_0\res.npz')

# print(inputs.shape,preds.shape,targets.shape)

for i in range(len(inputx)):
    for j in range(len(inputx[i])):
        inputs.append(inputx[i][j])
        preds.append(predx[i][j])
        targets.append(targetx[i][j])
    print(i)

total_dice = 0
total_hd = 0
total_voe = 0
total_rvd = 0
total_recall = 0
for i in range(len(inputs)):


    cur_dice = dice(preds[i] > 0.5,targets[i])
    cur_diceA = dice(predA[i] > 0.5,targets[i])
    cur_diceB = dice(predB[i] > 0.5,targets[i])
    cur_diceC = dice(predC[i] > 0.5,targets[i])

    print(cur_dice,cur_diceA,cur_diceB,cur_diceC)
    if cur_dice > 0.0:

        # [recall, precision, voe, rvd] = VOE_RVD_Recall_Precision(targets[i],predi)
        # total_dice += cur_dice
        # total_hd += cur_hd
        # total_voe += voe
        # total_rvd += rvd
        # total_recall += recall
        plt.subplot(1,7,1)
        plt.imshow(inputs[i][0],cmap='gray')
        plt.axis('off')
        plt.subplot(1,7,2)
        plt.imshow(inputs[i][1],cmap='binary')
        plt.axis('off')
        plt.subplot(1,7,3)
        plt.imshow(targets[i][0],cmap='gray')
        plt.axis('off')
        # plt.imshow(preds[i][0][0])
        plt.subplot(1,7,4)
        pred = (preds[i][0] > 0.5).astype(int)
        plt.imshow(pred,cmap='gray')
        plt.axis('off')
        plt.subplot(1,7,5)
        pred = (predA[i][0] > 0.5).astype(int)
        plt.imshow(pred,cmap='gray')
        plt.axis('off')
        plt.subplot(1,7,6)
        pred = (predB[i][0] > 0.5).astype(int)
        plt.imshow(pred,cmap='gray')
        plt.axis('off')
        plt.subplot(1,7,7)
        pred = (predC[i][0] > 0.5).astype(int)
        plt.imshow(pred,cmap='gray')
        plt.axis('off')
        plt.show()


total_dice = total_dice / len(inputs)
total_hd = total_hd  / len(inputs)
# total_recall = total_recall / len(inputs)
total_voe = total_voe / len(inputs)
total_rvd = total_rvd / len(inputs)
print(total_dice,total_hd,total_voe,total_rvd)

# count = 0
# total_dice = 0
# total_hd = 0
# for i in range(len(inputs)):
#     for j in range(len(targets[i])):

#         pred = (preds> 0.5).astype(int)
#         # print(np.sum(targets[i][j][0]),np.sum(pred[i][j]))
#         cur_dice = dice(targets[i][j][0],pred[i][j][0])
#         cur_hd = hd95(pred[i][j][0],targets[i][j][0])
#         total_dice += cur_dice
#         total_hd += cur_hd
#         count = count + 1

# total_dice = total_dice / count
# total_hd = total_hd / count
# print(total_dice,total_hd)