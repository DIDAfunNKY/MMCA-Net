from argparse import Namespace
import torch
import numpy as np
import cv2
from predictor_sammed import SammedPredictor
from build_sam import sam_model_registry
import torch.nn as nn
import torch.nn.functional as F


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



class samt(nn.Module):
    def __init__(self):
        super().__init__()
        args = Namespace()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.image_size = 256
        args.encoder_adapter = True
        # args.sam_checkpoint = "F:\sam-med2d_b.pth"
        args.sam_checkpoint = None
        self.model = sam_model_registry["vit_l"](args)


    def forward(self, x):

        


        x = x[:,1:2,:,:]
        x = x.repeat(1,3,1,1)

        inputs = {'image':x,'original_size':256}
        outputs = self.model(inputs,False)
        result = outputs["masks"]


        return result
