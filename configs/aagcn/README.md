# AAGCN

## Abstract

Graph convolutional networks (GCNs), which generalize CNNs to more generic non-Euclidean structures, have achieved remarkable performance for skeleton-based action recognition. However, there still exist several issues in the previous GCN-based models. First, the topology of the graph is set heuristically and fixed over all the model layers and input data. This may not be suitable for the hierarchy of the GCN model and the diversity of the data in action recognition tasks. Second, the second-order information of the skeleton data, i.e., the length and orientation of the bones, is rarely investigated, which is naturally more informative and discriminative for the human action recognition. In this work, we propose a novel multi-stream attention-enhanced adaptive graph convolutional neural network (MS-AAGCN) for skeleton-based action recognition. The graph topology in our model can be either uniformly or individually learned based on the input data in an end-to-end manner. This data-driven approach increases the flexibility of the model for graph construction and brings more generality to adapt to various data samples. Besides, the proposed adaptive graph convolutional layer is further enhanced by a spatial-temporal-channel attention module, which helps the model pay more attention to important joints, frames and features. Moreover, the information of both the joints and bones, together with their motion information, are simultaneously modeled in a multi-stream framework, which shows notable improvement for the recognition accuracy. Extensive experiments on the two large-scale datasets, NTU-RGBD and Kinetics-Skeleton, demonstrate that the performance of our model exceeds the state-of-the-art with a significant margin.

## Citation

```BibTeX
@article{shi2020skeleton,
  title={Skeleton-based action recognition with multi-stream adaptive graph convolutional networks},
  author={Shi, Lei and Zhang, Yifan and Cheng, Jian and Lu, Hanqing},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={9532--9545},
  year={2020},
  publisher={IEEE}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/aagcn/ntu60_xsub_LT_aagcn/j.py): [69.3](https://drive.google.com/drive/folders/1myLXG2DS02DE9BYCpPMU5-eNv_HDGRJQ?usp=share_link) | [bone_config](/configs/aagcn/ntu60_xsub_LT_aagcn/b.py): [65.4](https://drive.google.com/drive/folders/1myLXG2DS02DE9BYCpPMU5-eNv_HDGRJQ?usp=share_link) | [joint_motion_config](/configs/aagcn/ntu60_xsub_LT_aagcn/jm.py): [69.8](https://drive.google.com/drive/folders/1myLXG2DS02DE9BYCpPMU5-eNv_HDGRJQ?usp=share_link) | [bone_motion_config](/configs/aagcn/ntu60_xsub_LT_aagcn/bm.py): [64.2](https://drive.google.com/drive/folders/1myLXG2DS02DE9BYCpPMU5-eNv_HDGRJQ?usp=share_link) | 71.7 | 74.0 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/aagcn/ntu60_xview_LT_aagcn/j.py): [75.2](https://drive.google.com/drive/folders/1RH40djJcEBghHrHYHWFy6n-HWbWHRL8L?usp=share_link) | [bone_config](/configs/aagcn/ntu60_xview_LT_aagcn/b.py): [72.0](https://drive.google.com/drive/folders/1RH40djJcEBghHrHYHWFy6n-HWbWHRL8L?usp=share_link) | [joint_motion_config](/configs/aagcn/ntu60_xview_LT_aagcn/jm.py): [71.9](https://drive.google.com/drive/folders/1RH40djJcEBghHrHYHWFy6n-HWbWHRL8L?usp=share_link) | [bone_motion_config](/configs/aagcn/ntu60_xview_LT_aagcn/bm.py): [69.6](https://drive.google.com/drive/folders/1RH40djJcEBghHrHYHWFy6n-HWbWHRL8L?usp=share_link) | 76.7 | 78.9 |
| NTURGB+D 120 XSub | Official 3D Skeleton | [joint_config](/configs/aagcn/ntu120_xsub_LT_aagcn/j.py): [56.7](https://drive.google.com/drive/folders/1USpJW5m9AJBMONTZbwOdHbxMp1a2qxIT?usp=share_link) | [bone_config](/configs/aagcn/ntu120_xsub_LT_aagcn/b.py): [57.0](https://drive.google.com/drive/folders/1USpJW5m9AJBMONTZbwOdHbxMp1a2qxIT?usp=share_link) | [joint_motion_config](/configs/aagcn/ntu120_xsub_LT_aagcn/jm.py): [49.4](https://drive.google.com/drive/folders/1USpJW5m9AJBMONTZbwOdHbxMp1a2qxIT?usp=share_link) | [bone_motion_config](/configs/aagcn/ntu120_xsub_LT_aagcn/bm.py): [50.9](https://drive.google.com/drive/folders/1USpJW5m9AJBMONTZbwOdHbxMp1a2qxIT?usp=share_link) | 60.0 | 61.1 |
| NTURGB+D 120 XSet | Official 3D Skeleton | [joint_config](/configs/aagcn/ntu120_xset_LT_aagcn/j.py): [58.7](https://drive.google.com/drive/folders/1ha2JTR2LUWEn6zjfsgRkVyKyqxqrW-Sh?usp=share_linkh) | [bone_config](/configs/aagcn/ntu120_xset_LT_aagcn/b.py): [57.6](https://drive.google.com/drive/folders/1ha2JTR2LUWEn6zjfsgRkVyKyqxqrW-Sh?usp=share_link) | [joint_motion_config](/configs/aagcn/ntu120_xset_LT_aagcn/jm.py): [56.6](https://drive.google.com/drive/folders/1ha2JTR2LUWEn6zjfsgRkVyKyqxqrW-Sh?usp=share_link) | [bone_motion_config](/configs/aagcn/ntu120_xset_LT_aagcn/bm.py): [55.1](https://drive.google.com/drive/folders/1ha2JTR2LUWEn6zjfsgRkVyKyqxqrW-Sh?usp=share_link) | 61.3 | 63.2 |

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train AAGCN on NTURGB+D XSub (Joint Modality) with one GPU, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/aagcn/ntu60_xsub_LT_aagcn/j.py 1 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test AAGCN on NTURGB+D XSub (Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/aagcn/ntu60_xsub_LT_aagcn/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

You can use the following command to ensemble the results of different modalities.
```
cd ./tools
python ensemble.py
```
