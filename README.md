# MMCA-NET: A Multimodal Cross Attention Transformer Network for Nasopharyngeal Carcinoma Tumor Segmentation Based on a Total-Body PET/CT System
[[`Paper`](https://doi.org/10.1109/JBHI.2024.3405993)]

Official code implement of MMCA-Net.
# caNet.py is the implmentation of MMCA-Net and its sub-branch models.
# Run train-Batch.py, to reproduce the whole experiment, all methods in ablation experiments and comparison experiments are inclued in train-Batch.py.
(You may need to modifed dataloader to your own path, and apply for the public dataset HECKTOR, more detials about the dataset can be obtained with Google. Thanks)

## Results
| **Dataset**      | **Model**                         | **DICE$\uparrow$** | **HD95$\downarrow$** | **VOE$\downarrow$** | **RVD$\downarrow$** |
|------------------|-----------------------------------|--------------------|----------------------|---------------------|---------------------|
| **~**            | Sam-B \cite{SAM}                  | 0.6450             | 14.7267              | 0.3717              | 0.8935              |
| **~**            | Sam-L \cite{SAM}                  | 0.6294             | 14.3043              | 0.4078              | 1.0333              |
| **~**            | Sam-Med-B \cite{sammed}           | 0.7116             | 8.7302               | 0.2767              | 0.3136              |
| **~**            | U-Net\cite{Unet}                  | 0.7887             | 4.2196               | 0.2113              | 0.1528              |
| **~**            | DAUNet\cite{DAUnet}               | 0.7840             | 5.1349               | 0.2160              | 0.1339              |
| **HECKTOR**      | TransUNet\cite{chen2021transUnet} | 0.7799             | 5.7205               | 0.2201              | 0.1664              |
| **~**            | Zhao\cite{Zhao}                   | 0.7837             | 4.5817               | 0.2163              | 0.1923              |
| **~**            | AttUNet\cite{attUnet}             | 0.7794             | 8.1697               | 0.2206              | 0.2032              |
| **~**            | DenseNet\cite{DCNet}              | 0.7817             | 4.5893               | 0.2183              | 0.1755              |
| **~**            | ResUNet++\cite{ResUNet++}         | 0.7839             | 4.0542               | 0.2161              | 0.1841              |
| **~**            | Proposed                          | \textbf{0.7944}    | \textbf{3.9450}      | \textbf{0.2056}     | \textbf{0.1113}     |
| **~**            | Sam-B \cite{SAM}                  | 0.7468             | 12.7546              | 0.5250              | 0.9715              |
| **~**            | Sam-L \cite{SAM}                  | 0.5540             | 21.1386              | 0.4953              | 0.8624              |
| **~**            | Sam-Med-B \cite{sammed}           | 0.7828             | 11.2839              | 0.4788              | 0.9684              |
| **~**            | U-Net\cite{Unet}                  | 0.7981             | 10.9227              | 0.2019              | 0.5484              |
| **~**            | DAUNet\cite{DAUnet}               | 0.7881             | 12.2515              | 0.2119              | 0.5756              |
| **Self-Collect** | TransUNet\cite{chen2021transUnet} | 0.797              | 11.945               | 0.2034              | 0.5190              |
| **~**            | Zhao\cite{Zhao}                   | 0.7970             | 11.0422              | 0.2029              | 0.5402              |
| **~**            | AttUNet\cite{attUnet}             | 0.7972             | 10.7563              | 0.2028              | 0.5449              |
| **~**            | DenseNet\cite{DCNet}              | 0.8076             | 12.2831              | 0.1924              | 0.5403              |
| **~**            | ResUNet++\cite{ResUNet++}         | 0.8061             | 12.1215              | 0.1939              | 0.6771              |
| **~**            | Proposed                          | \textbf{0.8153}    | \textbf{9.9860}      | \textbf{0.1847}     | \textbf{0.4998}     |


## Citing

```
@ARTICLE{10540111,
  author={Zhao, Wenjie and Huang, Zhenxing and Tang, Si and Li, Wenbo and Gao, Yunlong and Hu, Yingying and Fan, Wei and Cheng, Chuanli and Yang, Yongfeng and Zheng, Hairong and Liang, Dong and Hu, Zhanli},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={MMCA-NET: A Multimodal Cross Attention Transformer Network for Nasopharyngeal Carcinoma Tumor Segmentation Based on a Total-Body PET/CT System}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Feature extraction;Image segmentation;Computed tomography;Transformers;Decoding;Cancer;Deep learning;Nasopharyngeal carcinoma segmentation;Multimodal PET/CT;Transformer;Cross attention},
  doi={10.1109/JBHI.2024.3405993}}

```
