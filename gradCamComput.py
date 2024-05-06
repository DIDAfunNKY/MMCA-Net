from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget 
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
from efficientnet_pytorch import EfficientNet
import numpy as np
import matplotlib.pyplot as plt

def dice(mask_gt, mask_seg):
    # print(mask_gt,mask_seg)
    # print(np.logical_and(mask_gt, mask_seg))
    return 2 * np.sum(np.logical_and(
        mask_gt, mask_seg)) / (np.sum(mask_gt) + np.sum(mask_seg)+1e-8)


def normalize_max_min(img):
    Max = np.max(img)
    Min = np.min(img)
    return (img - Min) / ((Max - Min) + 1e-4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = resnet50(pretrained=True)
model = torch.load(r'D:\NPC\results\compare_model\JBHI-H-unet\re_0\best_model\net_model.pth')


data = np.load(r'D:\NPC\results\compare_model\JBHI-H-unet\re_0\res.npz')


# select = []
# select = [93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 110, 112, 113, 126, 127, 130]
for ci in range(len(data['pred'])):
    preds = data['pred'][ci]
    targets = data['target'][ci]

    # for i in range(len(preds)):
    #     predi = preds[i] > 0.5

    #     cur_dice = dice(predi,targets[i])
    #     if cur_dice > 0.9:
    #         select.append(i)

    # print(select)


    inputs = data['input'][ci]
    # for i in range(len(inputs)):
    #     plt.imshow(inputs[i][1])
    #     plt.show()

    input_tensor = torch.tensor(inputs)# Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    target_layers = [model.block_4_2_right]
    cam = EigenCAM(model=model, target_layers=target_layers)


    target = SemanticSegmentationTarget(category=0,mask=targets)

    grayscale_cam = cam(input_tensor=input_tensor, targets=target)

    model_outputs = cam.outputs

    print(model_outputs)

    print(len(grayscale_cam))

    np.save('./gradCam/unet1'+str(ci)+'.npy',grayscale_cam)


    target_layers = [model.conv1x1]
    cam = EigenCAM(model=model, target_layers=target_layers)


    target = SemanticSegmentationTarget(category=0,mask=targets)

    grayscale_cam = cam(input_tensor=input_tensor, targets=target)

    model_outputs = cam.outputs

    print(model_outputs)

    print(len(grayscale_cam))

    np.save('./gradCam/unet2'+str(ci)+'.npy',grayscale_cam)
# for i in range(len(select)):

#     image = inputs[i][1]
#     image = np.expand_dims(image,2).repeat(3,axis=2)
#     image = normalize_max_min(image)

#     visualization = show_cam_on_image(image, grayscale_cam[i], use_rgb=True,image_weight=0.75)

#     plt.subplot(2,len(select),i+1)
#     plt.imshow(visualization)
#     plt.axis('off')

#     image = inputs[i][0]
#     image = np.expand_dims(image,2).repeat(3,axis=2)
#     image = normalize_max_min(image)

#     visualization = show_cam_on_image(image, grayscale_cam[i], use_rgb=True,image_weight=0.75)

#     plt.subplot(2,len(select),i+1+len(select))
#     plt.imshow(visualization)
#     plt.axis('off')
# plt.show()