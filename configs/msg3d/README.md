# MSG3D

## Abstract

Spatial-temporal graphs have been widely used by skeleton-based action recognition algorithms to model human action dynamics. To capture robust movement patterns from these graphs, long-range and multi-scale context aggregation and spatial-temporal dependency modeling are critical aspects of a powerful feature extractor. However, existing methods have limitations in achieving (1) unbiased long-range joint relationship modeling under multi-scale operators and (2) unobstructed cross-spacetime information flow for capturing complex spatial-temporal dependencies. In this work, we present (1) a simple method to disentangle multi-scale graph convolutions and (2) a unified spatial-temporal graph convolutional operator named G3D. The proposed multi-scale aggregation scheme disentangles the importance of nodes in different neighborhoods for effective long-range modeling. The proposed G3D module leverages dense cross-spacetime edges as skip connections for direct information propagation across the spatial-temporal graph. By coupling these proposals, we develop a powerful feature extractor named MS-G3D based on which our model outperforms previous state-of-the-art methods on three large-scale datasets: NTU RGB+D 60, NTU RGB+D 120, and Kinetics Skeleton 400.

## Citation

```BibTeX
@inproceedings{liu2020disentangling,
  title={Disentangling and unifying graph convolutions for skeleton-based action recognition},
  author={Liu, Ziyu and Zhang, Hongwen and Chen, Zhenghao and Wang, Zhiyong and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={143--152},
  year={2020}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/msg3d/ntu60_xsub_LT_msg3d/j.py): [69.8](https://drive.google.com/drive/folders/11EwlgF88jADIinH446ARGV8GLzm4wedh?usp=share_link) | [bone_config](/configs/msg3d/ntu60_xsub_LT_msg3d/b.py): [65.7](https://drive.google.com/drive/folders/11EwlgF88jADIinH446ARGV8GLzm4wedh?usp=share_link) | [joint_motion_config](/configs/msg3d/ntu60_xsub_LT_msg3d/jm.py): [66.3](https://drive.google.com/drive/folders/11EwlgF88jADIinH446ARGV8GLzm4wedh?usp=share_link) | [bone_motion_config](/configs/msg3d/ntu60_xsub_LT_msg3d/bm.py): [65.5](https://drive.google.com/drive/folders/11EwlgF88jADIinH446ARGV8GLzm4wedh?usp=share_link) | 72.1 | 74.8 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/msg3d/ntu60_xview_LT_msg3d/j.py): [77.2](https://drive.google.com/drive/folders/1qUZJvNwWYzXzowR0rbydxTEakHDLczLL?usp=share_link) | [bone_config](/configs/msg3d/ntu60_xview_LT_msg3d/b.py): [72.0](https://drive.google.com/drive/folders/1qUZJvNwWYzXzowR0rbydxTEakHDLczLL?usp=share_link) | [joint_motion_config](/configs/msg3d/ntu60_xview_LT_msg3d/jm.py): [72.4](https://drive.google.com/drive/folders/1qUZJvNwWYzXzowR0rbydxTEakHDLczLL?usp=share_link) | [bone_motion_config](/configs/msg3d/ntu60_xview_LT_msg3d/bm.py): [68.3](https://drive.google.com/drive/folders/1qUZJvNwWYzXzowR0rbydxTEakHDLczLL?usp=share_link) | 78.4 | 80.9 |
| NTURGB+D 120 XSub | Official 3D Skeleton | [joint_config](/configs/msg3d/ntu120_xsub_LT_msg3d/j.py): [59.4](https://drive.google.com/drive/folders/1qUZJvNwWYzXzowR0rbydxTEakHDLczLL?usp=share_linkh) | [bone_config](/configs/msg3d/ntu120_xsub_LT_msg3d/b.py): [56.7](https://drive.google.com/drive/folders/1qUZJvNwWYzXzowR0rbydxTEakHDLczLL?usp=share_link) | [joint_motion_config](/configs/msg3d/ntu120_xsub_LT_msg3d/jm.py): [56.6](https://drive.google.com/drive/folders/1qUZJvNwWYzXzowR0rbydxTEakHDLczLL?usp=share_link) | [bone_motion_config](/configs/msg3d/ntu120_xsub_LT_msg3d/bm.py): [52.3](https://drive.google.com/drive/folders/1qUZJvNwWYzXzowR0rbydxTEakHDLczLL?usp=share_link) | 61.0 | 62.7 |
| NTURGB+D 120 XSet | Official 3D Skeleton | [joint_config](/configs/msg3d/ntu120_xset_LT_msg3d/j.py): [61.0](https://drive.google.com/drive/folders/16ewnOb3QVen9CJuLzMdupzOyM6PIFlKV?usp=share_link) | [bone_config](/configs/msg3d/ntu120_xset_LT_msg3d/b.py): [57.8](https://drive.google.com/drive/folders/16ewnOb3QVen9CJuLzMdupzOyM6PIFlKV?usp=share_link) | [joint_motion_config](/configs/msg3d/ntu120_xset_LT_msg3d/jm.py): [59.4](https://drive.google.com/drive/folders/16ewnOb3QVen9CJuLzMdupzOyM6PIFlKV?usp=share_link) | [bone_motion_config](/configs/msg3d/ntu120_xset_LT_msg3d/bm.py): [54.6](https://drive.google.com/drive/folders/16ewnOb3QVen9CJuLzMdupzOyM6PIFlKV?usp=share_link) | 62.2 | 64.6 |

We also provide numerous checkpoints trained with BFL (Balanced Representation Learning) on NTURGB+D. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Skip Top1 | Joint Motion Top1 | Bone Motion Top1 | Skip Motion Top1 | Two-Stream Top1 | Four Stream Top1 | Six Stream Top1|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/msg3d/msg3d_BFL_ntu60_xsub/j.py): [78.0](https://drive.google.com/drive/folders/1wBHSQ15UjUOgGTSnOz9033_U25scN2Fr?usp=share_link) | [bone_config](/configs/msg3d/msg3d_BFL_ntu60_xsub/b.py): [77.0](https://drive.google.com/drive/folders/1wBHSQ15UjUOgGTSnOz9033_U25scN2Fr?usp=share_link) | [skip_config](/configs/msg3d/msg3d_BFL_ntu60_xsub/k.py): [78.9](https://drive.google.com/drive/folders/1wBHSQ15UjUOgGTSnOz9033_U25scN2Fr?usp=share_link) | [joint_motion_config](/configs/msg3d/msg3d_BFL_ntu60_xsub/jm.py): [75.4](https://drive.google.com/drive/folders/1wBHSQ15UjUOgGTSnOz9033_U25scN2Fr?usp=share_link) | [bone_motion_config](/configs/msg3d/msg3d_BFL_ntu60_xsub/bm.py): [73.9](https://drive.google.com/drive/folders/1wBHSQ15UjUOgGTSnOz9033_U25scN2Fr?usp=share_link) | [skip_motion_config](/configs/msg3d/msg3d_BFL_ntu60_xsub/km.py): [73.8](https://drive.google.com/drive/folders/1wBHSQ15UjUOgGTSnOz9033_U25scN2Fr?usp=share_link) | 0 | 81.8 | 82.4 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/msg3d/msg3d_BFL_ntu60_xview/j.py): [82.4](https://drive.google.com/drive/folders/1ZmB7cTharS5YpNBk50XRnfhbV__CHU8S?usp=share_link) | [bone_config](/configs/msg3d/msg3d_BFL_ntu60_xview/b.py): [81.0](https://drive.google.com/drive/folders/1ZmB7cTharS5YpNBk50XRnfhbV__CHU8S?usp=share_link) | [skip_config](/configs/msg3d/msg3d_BFL_ntu60_xview/k.py): [81.6](https://drive.google.com/drive/folders/1ZmB7cTharS5YpNBk50XRnfhbV__CHU8S?usp=share_link) | [joint_motion_config](/configs/msg3d/msg3d_BFL_ntu60_xview/jm.py): [79.4](https://drive.google.com/drive/folders/1ZmB7cTharS5YpNBk50XRnfhbV__CHU8S?usp=share_link) | [bone_motion_config](/configs/msg3d/msg3d_BFL_ntu60_xview/bm.py): [77.1](https://drive.google.com/drive/folders/1ZmB7cTharS5YpNBk50XRnfhbV__CHU8S?usp=share_link) | [skip_motion_config](/configs/msg3d/msg3d_BFL_ntu60_xview/km.py): [76.9](https://drive.google.com/drive/folders/1ZmB7cTharS5YpNBk50XRnfhbV__CHU8S?usp=share_link) | 0 | 85.3 | 85.7 |


**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion. For Six-Stream results, we adopt the **2 (Joint):2 (Bone):2 (Skip):1 (Joint Motion):1 (Bone Motion):1 (Skip Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train MSG3D on NTURGB+D XSub (Joint Modality) with one GPU, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/msg3d/ntu60_xsub_LT_msg3d/j.py 1 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test MSG3D on NTURGB+D XSub (Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/msg3d/ntu60_xsub_LT_msg3d/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

You can use the following command to ensemble the results of different modalities.
```
cd ./tools
python ensemble.py
```
