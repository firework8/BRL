# BRL

## Introduction

This directory includes configs for training BRL. We provide BRL trained on NTURGB+D 60 and NTURGB+D 120 in the **long-tailed** training setting. We provide checkpoints for six modalities: Joint, Bone, Skip, Joint Motion, Bone Motion, and Skip Motion. The accuracy of each modality links to the weight file.

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Skip Top1 | Joint Motion Top1 | Bone Motion Top1 | Skip Motion Top1 | Two-Stream Top1 | Four Stream Top1 | Six Stream Top1|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/BRL/ntu60_xsub_LT/j.py): [76.7](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [bone_config](/configs/BRL/ntu60_xsub_LT/b.py): [76.1](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [skip_config](/configs/BRL/ntu60_xsub_LT/k.py): [77.7](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [joint_motion_config](/configs/BRL/ntu60_xsub_LT/jm.py): [75.0](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [bone_motion_config](/configs/BRL/ntu60_xsub_LT/bm.py): [72.8](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [skip_motion_config](/configs/BRL/ntu60_xsub_LT/km.py): [73.4](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | 79.6 | 81.0 | 81.8 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/BRL/ntu60_xview_LT/j.py): [81.4](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [bone_config](/configs/BRL/ntu60_xview_LT/b.py): [80.3](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [skip_config](/configs/BRL/ntu60_xview_LT/k.py): [81.1](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [joint_motion_config](/configs/BRL/ntu60_xview_LT/jm.py): [78.5](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [bone_motion_config](/configs/BRL/ntu60_xview_LT/bm.py): [76.2](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [skip_motion_config](/configs/BRL/ntu60_xview_LT/km.py): [77.2](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | 84.0 | 84.9 | 85.4 |
| NTURGB+D 120 XSub | Official 3D Skeleton | [joint_config](/configs/BRL/ntu120_xsub_LT/j.py): [65.3](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [bone_config](/configs/BRL/ntu120_xsub_LT/b.py): [65.3](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [skip_config](/configs/BRL/ntu120_xsub_LT/k.py): [64.2](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [joint_motion_config](/configs/BRL/ntu120_xsub_LT/jm.py): [59.7](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [bone_motion_config](/configs/BRL/ntu120_xsub_LT/bm.py): [59.8](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [skip_motion_config](/configs/BRL/ntu120_xsub_LT/km.py): [59.6](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | 68.7 | 69.4 | 69.7 |
| NTURGB+D 120 XSet | Official 3D Skeleton | [joint_config](/configs/BRL/ntu120_xset_LT/j.py): [66.8](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [bone_config](/configs/BRL/ntu120_xset_LT/b.py): [66.6](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [skip_config](/configs/BRL/ntu120_xset_LT/k.py): [65.9](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [joint_motion_config](/configs/BRL/ntu120_xset_LT/jm.py): [63.5](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [bone_motion_config](/configs/BRL/ntu120_xset_LT/bm.py): [62.2](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [skip_motion_config](/configs/BRL/ntu120_xset_LT/km.py): [61.6](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | 69.7 | 71.0 | 71.3 |

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

You can use the following command to ensemble the results of different modalities.
```
cd ./tools
python ensemble.py
```
