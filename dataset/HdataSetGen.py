import numpy as np

import utils
import nibabel as nib

import matplotlib.pyplot as plt



def read_data(path_to_nifti, return_numpy=True):
    """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
    if return_numpy:
        return nib.load(str(path_to_nifti)).get_fdata()
    return nib.load(str(path_to_nifti))

paths = utils.get_paths_to_patient_files(r'C:\Users\91694\Desktop\MMCANET\TN-HECKTOR\HECKTOR_224')

cts=[]
inputs = []
targets = []

print(len(paths))
for i in range(len(paths)):
    ct = read_data(paths[i][0])
    pt = read_data(paths[i][1])
    mask = read_data(paths[i][2])

    input = np.stack([ct, pt], axis=-1)

    inputs.append(input)
    targets.append(mask)

    # print(data.shape)
np.savez("./HECKTOR.npz",input=inputs,target=targets)
# plt.imshow(data[120])
# plt.show()

