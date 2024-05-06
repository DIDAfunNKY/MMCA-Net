import numpy as np
import torch.nn as nn
from surface_distance import compute_surface_distances, compute_robust_hausdorff
import numpy as np
from scipy.spatial import cKDTree

# def Dice(input, target):
#     axes = tuple(range(1, input.dim()))
#     bin_input = (input > 0.5).float()
#     intersect = (bin_input * target).sum(dim=axes)
#     union = bin_input.sum(dim=axes) + target.sum(dim=axes)
#     score = 2 * intersect / (union + 1e-3)
#     return score.mean()

class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        self.smooth = 0.001

    def forward(self, input, target):
        c = input.dim()
        axes = tuple(range(0, input.dim()))
        intersect = (input * target).sum(dim=axes)
        # union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        union = input.sum(dim=axes) + target.sum(dim=axes)
        dic = 2 * intersect / (union + self.smooth)
        return dic.mean()

def dice(mask_gt, mask_seg):
    # print(mask_gt,mask_seg)
    # print(np.logical_and(mask_gt, mask_seg))
    return 2 * np.sum(np.logical_and(
        mask_gt, mask_seg)) / (np.sum(mask_gt) + np.sum(mask_seg)+1e-8)


def dice_1(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()
    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)
    return score.mean()


def dice_2(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.6).float()
    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)
    return score.mean()


def recall(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positives = target.sum(dim=axes)
    recall = true_positives / all_positives

    return recall.mean()


def precision(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positive_calls = binary_input.sum(dim=axes)
    precision = true_positives / all_positive_calls

    return precision.mean()


def robust_hausdorff(image0, image1, spacing, percent=95.0):
    # image0 = (image0 / 255).astype(np.uint8)
    surface_distances = compute_surface_distances(image0 != 0, image1 != 0,
                                                  spacing)
    return compute_robust_hausdorff(surface_distances, percent)


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
