# STGCN

## Introduction

STGCN is one of the first algorithms that adopt Graph Convolution Neural Networks for skeleton processing. We provide STGCN trained on NTURGB+D 60 and NTURGB+D 120 in the **long-tailed** training setting. We provide checkpoints for four modalities: Joint, Bone, Joint Motion, and Bone Motion. The accuracy of each modality links to the weight file.

## Citation

```BibTeX
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Thirty-second AAAI conference on artificial intelligence},
  year={2018}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone-Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/stgcn/ntu60_xsub_LT_stgcn/j.py): [91.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xsub_3dkp/j.pth) | [bone_config](/configs/stgcn/ntu60_xsub_LT_stgcn/b.py): [91.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/stgcn/ntu60_xsub_LT_stgcn/jm.py): [88.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn/ntu60_xsub_LT_stgcn/bm.py): [88.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xsub_3dkp/bm.pth) | 92.9 | 93.2 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/stgcn/ntu60_xview_LT_stgcn/j.py): [96.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xview_3dkp/j.pth) | [bone_config](/configs/stgcn/ntu60_xview_LT_stgcn/b.py): [96.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xview_3dkp/b.pth) | [joint_motion_config](/configs/stgcn/ntu60_xview_LT_stgcn/jm.py): [95.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xview_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn/ntu60_xview_LT_stgcn/bm.py): [94.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xview_3dkp/bm.pth) | 97.4 | 97.5 |
| NTURGB+D 120 XSub | Official 3D Skeleton | [joint_config](/configs/stgcn/ntu120_xsub_LT_stgcn/j.py): [85.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xsub_3dkp/j.pth) | [bone_config](/configs/stgcn/ntu120_xsub_LT_stgcn/b.py): [88.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xsub_3dkp/b.pth) | [joint_motion_config](/configs/stgcn/ntu120_xsub_LT_stgcn/jm.py): [82.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn/ntu120_xsub_LT_stgcn/bm.py): [83.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xsub_3dkp/bm.pth) | 89.3 | 89.6 |
| NTURGB+D 120 XSet | Official 3D Skeleton | [joint_config](/configs/stgcn/ntu120_xset_LT_stgcn/j.py): [87.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xset_3dkp/j.pth) | [bone_config](/configs/stgcn/ntu120_xset_LT_stgcn/b.py): [89.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xset_3dkp/b.pth) | [joint_motion_config](/configs/stgcn/ntu120_xset_LT_stgcn/jm.py): [85.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xset_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn/ntu120_xset_LT_stgcn/bm.py): [85.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xset_3dkp/bm.pth) | 91.2 | 91.3 |

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train STGCN on NTURGB+D XSub (Joint Modality) with one GPU, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/stgcn/ntu60_xsub_LT_stgcn/j.py 1 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test STGCN on NTURGB+D XSub (Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/stgcn/ntu60_xsub_LT_stgcn/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```
