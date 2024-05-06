import numpy as np
import matplotlib.pyplot as plt
import cv2

def dice(mask_gt, mask_seg):
    # print(mask_gt,mask_seg)
    # print(np.logical_and(mask_gt, mask_seg))
    return 2 * np.sum(np.logical_and(
        mask_gt, mask_seg)) / (np.sum(mask_gt) + np.sum(mask_seg)+1e-8)


data1 = np.load(r'D:\NPC\results\compare_model\JBHI-H-CasNet\re_0\res.npz')
data2 = np.load(r'D:\NPC\results\compare_model\JBHI-H-attunet\re_0\res.npz')
data3 = np.load(r'D:\NPC\results\compare_model\JBHI-H-resunet\re_0\res.npz')
data4 = np.load(r'D:\NPC\results\compare_model\JBHI-H-unet\re_0\res.npz')
data5 = np.load(r'D:\NPC\results\compare_model\JBHI-H-transunet\re_0\res.npz')
data6 = np.load(r'D:\NPC\results\compare_model\JBHI-H-liu\re_0\res.npz')
data7 = np.load(r'D:\NPC\results\compare_model\JBHI-H-zhao\re_0\res.npz')
data8 = np.load(r'D:\NPC\results\compare_model\JBHI-H-denseUnet\re_0\res.npz')
data9 = np.load(r'D:\NPC\results\compare_model\JBHI-H-sam-med-B-Sam\re_0\res.npz')


preds= [[],[],[],[],[],[],[],[],[]]
ct = []
pet = []
gt = []

dataList = [data1,data2,data3,data4,data5,data7,data7,data8,data9]
for i in range(len(dataList)):
    pred = dataList[i]['pred']
    print(pred.shape)
    for j in range(pred.shape[0]):
        for k in range(pred.shape[1]):
            preds[i].append(pred[j,k,0])

dt = data1['target']
di = data1['input']
for i in range(11):
    for j in range(144):
        gt.append(dt[i,j,0])
        ct.append(di[i,j,0])
        pet.append(di[i,j,1])
print('done')
dices = [0,0,0,0,0,0,0,0,0]

re_i = [277,280,323,330,338,518,552,675,685,953,1228,1326,1513]
re_ii = [323,675,953,1228,1513]
# for i in range(25*64):
#     for j in range(len(preds)):
#         dices[j] = dice(gt[i],preds[j][i] > 0.5)
#     if np.argmax(dices,axis=-1) == 0 and dices[0] > 0.8:
#         print(i,dices)
#         plt.figure(figsize=(16,8))
#         plt.subplot(1,11,1)
#         plt.imshow(ct[i],cmap='gray')
#         plt.axis('off')
#         plt.subplot(1,11,2)
#         plt.imshow(pet[i],cmap='binary')
#         plt.axis('off')
#         plt.subplot(1,11,3)
#         plt.imshow(gt[i],cmap='gray')
#         plt.axis('off')
#         for k in range(8):
#             plt.subplot(1,11,4+k)
#             plt.imshow(preds[k][i] > 0.5,cmap='gray')
#             plt.text(0,140,'DSC='+str('%.3f' % dices[k]),fontsize=12, color = "r",family='Times New Roman')
#             plt.axis('off')
#         plt.show()

plt.figure(figsize=(16,8))
count = 0
for i in re_ii:
    for j in range(len(preds)):
        dices[j] = dice(gt[i],cv2.resize(preds[j][i],(144,144), interpolation=cv2.INTER_NEAREST) > 0.5)
    if np.argmax(dices,axis=-1) == 0 and dices[0] > 0.8:
        print(i,dices)
        plt.subplot(5,12,1+count*12)
        plt.imshow(ct[i],cmap='gray')
        plt.axis('off')
        plt.subplot(5,12,2+count*12)
        plt.imshow(pet[i],cmap='binary')
        plt.axis('off')
        plt.subplot(5,12,3+count*12)
        plt.imshow(gt[i],cmap='gray')
        plt.axis('off')
        for k in range(9):
            plt.subplot(5,12,4+k+count*12)
            plt.imshow(cv2.resize(preds[k][i],(144,144), interpolation=cv2.INTER_NEAREST) > 0.5,cmap='gray')
            plt.text(0,140,'DSC='+str('%.3f' % dices[k]),fontsize=12, color = "r",family='Times New Roman')
            plt.axis('off')
    count = count + 1
plt.show()


    


