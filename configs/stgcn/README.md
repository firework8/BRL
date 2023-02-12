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

| Dataset | Annotation | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/stgcn/ntu60_xsub_LT_stgcn/j.py): [67.9](https://drive.google.com/drive/folders/1orZoEDEDF1pkyN94LgpIR8ApCbY-jUX5?usp=share_link) | [bone_config](/configs/stgcn/ntu60_xsub_LT_stgcn/b.py): [65.0](https://drive.google.com/drive/folders/1orZoEDEDF1pkyN94LgpIR8ApCbY-jUX5?usp=share_link) | [joint_motion_config](/configs/stgcn/ntu60_xsub_LT_stgcn/jm.py): [64.7](hhttps://drive.google.com/drive/folders/1orZoEDEDF1pkyN94LgpIR8ApCbY-jUX5?usp=share_link) | [bone_motion_config](/configs/stgcn/ntu60_xsub_LT_stgcn/bm.py): [64.2](https://drive.google.com/drive/folders/1orZoEDEDF1pkyN94LgpIR8ApCbY-jUX5?usp=share_link) | 70.6 | 73.3 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/stgcn/ntu60_xview_LT_stgcn/j.py): [74.5](https://drive.google.com/drive/folders/19Z_pZ99UxfCyYYizKtZu3r2_jNBdxXt5?usp=share_link) | [bone_config](/configs/stgcn/ntu60_xview_LT_stgcn/b.py): [71.3](https://drive.google.com/drive/folders/19Z_pZ99UxfCyYYizKtZu3r2_jNBdxXt5?usp=share_link) | [joint_motion_config](/configs/stgcn/ntu60_xview_LT_stgcn/jm.py): [71.3](https://drive.google.com/drive/folders/19Z_pZ99UxfCyYYizKtZu3r2_jNBdxXt5?usp=share_link) | [bone_motion_config](/configs/stgcn/ntu60_xview_LT_stgcn/bm.py): [70.4](https://drive.google.com/drive/folders/19Z_pZ99UxfCyYYizKtZu3r2_jNBdxXt5?usp=share_link) | 77.1 | 79.4 |
| NTURGB+D 120 XSub | Official 3D Skeleton | [joint_config](/configs/stgcn/ntu120_xsub_LT_stgcn/j.py): [54.9](https://drive.google.com/drive/folders/1kC1aRuY-HZhvSGmex6lLWB7kO6frds_W?usp=share_linkh) | [bone_config](/configs/stgcn/ntu120_xsub_LT_stgcn/b.py): [54.8](https://drive.google.com/drive/folders/1kC1aRuY-HZhvSGmex6lLWB7kO6frds_W?usp=share_link) | [joint_motion_config](/configs/stgcn/ntu120_xsub_LT_stgcn/jm.py): [49.4](https://drive.google.com/drive/folders/1kC1aRuY-HZhvSGmex6lLWB7kO6frds_W?usp=share_link) | [bone_motion_config](/configs/stgcn/ntu120_xsub_LT_stgcn/bm.py): [50.4](https://drive.google.com/drive/folders/1kC1aRuY-HZhvSGmex6lLWB7kO6frds_W?usp=share_link) | 57.7 | 59.1 |
| NTURGB+D 120 XSet | Official 3D Skeleton | [joint_config](/configs/stgcn/ntu120_xset_LT_stgcn/j.py): [57.6](https://drive.google.com/drive/folders/1iiaXW9pf7MtmC_FyUUDFmSF70H0ngQ2d?usp=share_link) | [bone_config](/configs/stgcn/ntu120_xset_LT_stgcn/b.py): [56.7](https://drive.google.com/drive/folders/1iiaXW9pf7MtmC_FyUUDFmSF70H0ngQ2d?usp=share_link) | [joint_motion_config](/configs/stgcn/ntu120_xset_LT_stgcn/jm.py): [55.6](https://drive.google.com/drive/folders/1iiaXW9pf7MtmC_FyUUDFmSF70H0ngQ2d?usp=share_link) | [bone_motion_config](/configs/stgcn/ntu120_xset_LT_stgcn/bm.py): [55.2](https://drive.google.com/drive/folders/1iiaXW9pf7MtmC_FyUUDFmSF70H0ngQ2d?usp=share_link) | 60.3 | 62.3 |

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

You can use the following command to ensemble the results of different modalities.
```
cd ./tools
python ensemble.py
```