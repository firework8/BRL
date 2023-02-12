# BRL

## Introduction

This directory includes configs for training BRL. We provide BRL trained on NTURGB+D 60 and NTURGB+D 120 in the **long-tailed** training setting. We provide checkpoints for six modalities: Joint, Bone, Skip, Joint Motion, Bone Motion, and Skip Motion. The accuracy of each modality links to the weight file.

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Skip Top1 | Joint Motion Top1 | Bone Motion Top1 | Skip Motion Top1 | Two-Stream Top1 | Four Stream Top1 | Six Stream Top1|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/BRL/ntu60_xsub_LT/j.py): [90.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/j.pth) | [bone_config](/configs/BRL/ntu60_xsub_LT/b.py): [90.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/b.pth) | [skip_config](/configs/BRL/ntu60_xsub_LT/k.py): [90.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/BRL/ntu60_xsub_LT/jm.py): [88.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/BRL/ntu60_xsub_LT/bm.py): [87.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/bm.pth) | [skip_motion_config](/configs/BRL/ntu60_xsub_LT/km.py): [90.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/b.pth) | 92.2 | 92.6 | 92.6 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/BRL/ntu60_xview_LT/j.py): [96.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xview_3dkp/j.pth) | [bone_config](/configs/BRL/ntu60_xview_LT/b.py): [95.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xview_3dkp/b.pth) | [skip_config](/configs/BRL/ntu60_xview_LT/k.py): [90.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/BRL/ntu60_xview_LT/jm.py): [95.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xview_3dkp/jm.pth) | [bone_motion_config](/configs/BRL/ntu60_xview_LT/bm.py): [93.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xview_3dkp/bm.pth) | [skip_motion_config](/configs/BRL/ntu60_xview_LT/km.py): [90.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/b.pth) | 97.1 | 97.4 | 92.6 |
| NTURGB+D 120 XSub | Official 3D Skeleton | [joint_config](/configs/BRL/ntu120_xsub_LT/j.py): [84.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xsub_3dkp/j.pth) | [bone_config](/configs/BRL/ntu120_xsub_LT/b.py): [87.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xsub_3dkp/b.pth) | [skip_config](/configs/BRL/ntu120_xsub_LT/k.py): [90.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/BRL/ntu120_xsub_LT/jm.py): [82.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/BRL/ntu120_xsub_LT/bm.py): [81.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xsub_3dkp/bm.pth) | [skip_motion_config](/configs/BRL/ntu120_xsub_LT/km.py): [90.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/b.pth) | 88.2 | 88.6 | 92.6 |
| NTURGB+D 120 XSet | Official 3D Skeleton | [joint_config](/configs/BRL/ntu120_xset_LT/j.py): [86.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xset_3dkp/j.pth) | [bone_config](/configs/BRL/ntu120_xset_LT/b.py): [88.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xset_3dkp/b.pth) | [skip_config](/configs/BRL/ntu120_xset_LT/k.py): [90.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/BRL/ntu120_xset_LT/jm.py): [85.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xset_3dkp/jm.pth) | [bone_motion_config](/configs/BRL/ntu120_xset_LT/bm.py): [84.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xset_3dkp/bm.pth) | [skip_motion_config](/configs/BRL/ntu120_xset_LT/km.py): [90.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/b.pth) | 90.1 | 90.8 | 92.6 |

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion. For Six-Stream results, we adopt the **2 (Joint):2 (Bone):2 (Skip):1 (Joint Motion):1 (Bone Motion):1 (Skip Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train BRL on NTURGB+D XSub (Joint Modality) with one GPU, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/BRL/ntu60_xsub_LT/j.py 1 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test BRL on NTURGB+D XSub (Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/BRL/ntu60_xsub_LT/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```
