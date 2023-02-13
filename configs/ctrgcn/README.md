# CTRGCN

## Abstract

Graph convolutional networks (GCNs) have been widely used and achieved remarkable results in skeleton-based action recognition. In GCNs, graph topology dominates feature aggregation and therefore is the key to extracting representative features. In this work, we propose a novel Channel-wise Topology Refinement Graph Convolution (CTR-GC) to dynamically learn different topologies and effectively aggregate joint features in different channels for skeleton-based action recognition. The proposed CTR-GC models channel-wise topologies through learning a shared topology as a generic prior for all channels and refining it with channel-specific correlations for each channel. Our refinement method introduces few extra parameters and significantly reduces the difficulty of modeling channel-wise topologies. Furthermore, via reformulating graph convolutions into a unified form, we find that CTR-GC relaxes strict constraints of graph convolutions, leading to stronger representation capability. Combining CTR-GC with temporal modeling modules, we develop a powerful graph convolutional network named CTR-GCN which notably outperforms state-of-the-art methods on the NTU RGB+D, NTU RGB+D 120, and NW-UCLA datasets.

## Citation

```BibTeX
@inproceedings{chen2021channel,
  title={Channel-wise topology refinement graph convolution for skeleton-based action recognition},
  author={Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13359--13368},
  year={2021}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/ctrgcn/ntu60_xsub_LT_ctrgcn/j.py): [69.3](https://drive.google.com/drive/folders/1noLdMbJdcZ_yJf0cCh6OeE87I5-bVs23?usp=share_link) | [bone_config](/configs/ctrgcn/ntu60_xsub_LT_ctrgcn/b.py): [63.5](https://drive.google.com/drive/folders/1noLdMbJdcZ_yJf0cCh6OeE87I5-bVs23?usp=share_link) | [joint_motion_config](/configs/ctrgcn/ntu60_xsub_LT_ctrgcn/jm.py): [64.3](https://drive.google.com/drive/folders/1noLdMbJdcZ_yJf0cCh6OeE87I5-bVs23?usp=share_link) | [bone_motion_config](/configs/ctrgcn/ntu60_xsub_LT_ctrgcn/bm.py): [65.8](https://drive.google.com/drive/folders/1noLdMbJdcZ_yJf0cCh6OeE87I5-bVs23?usp=share_link) | 71.2 | 74.1 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/ctrgcn/ntu60_xview_LT_ctrgcn/j.py): [75.6](https://drive.google.com/drive/folders/1_D6mS-SlTdIDqT9gpxfdXyDTDa23t6nJ?usp=share_link) | [bone_config](/configs/ctrgcn/ntu60_xview_LT_ctrgcn/b.py): [72.6](https://drive.google.com/drive/folders/1_D6mS-SlTdIDqT9gpxfdXyDTDa23t6nJ?usp=share_link) | [joint_motion_config](/configs/ctrgcn/ntu60_xview_LT_ctrgcn/jm.py): [73.0](https://drive.google.com/drive/folders/1_D6mS-SlTdIDqT9gpxfdXyDTDa23t6nJ?usp=share_link) | [bone_motion_config](/configs/ctrgcn/ntu60_xview_LT_ctrgcn/bm.py): [71.9](https://drive.google.com/drive/folders/1_D6mS-SlTdIDqT9gpxfdXyDTDa23t6nJ?usp=share_link) | 77.5 | 80.4 |
| NTURGB+D 120 XSub | Official 3D Skeleton | [joint_config](/configs/ctrgcn/ntu120_xsub_LT_ctrgcn/j.py): [57.7](https://drive.google.com/drive/folders/1FJBnNiNdzgcJpcmrvDXeysuvcoQeVUZH?usp=share_link) | [bone_config](/configs/ctrgcn/ntu120_xsub_LT_ctrgcn/b.py): [58.7](https://drive.google.com/drive/folders/1FJBnNiNdzgcJpcmrvDXeysuvcoQeVUZH?usp=share_link) | [joint_motion_config](/configs/ctrgcn/ntu120_xsub_LT_ctrgcn/jm.py): [54.8](https://drive.google.com/drive/folders/1FJBnNiNdzgcJpcmrvDXeysuvcoQeVUZH?usp=share_link) | [bone_motion_config](/configs/ctrgcn/ntu120_xsub_LT_ctrgcn/bm.py): [54.8](https://drive.google.com/drive/folders/1FJBnNiNdzgcJpcmrvDXeysuvcoQeVUZH?usp=share_link) | 61.6 | 63.2 |
| NTURGB+D 120 XSet | Official 3D Skeleton | [joint_config](/configs/ctrgcn/ntu120_xset_LT_ctrgcn/j.py): [61.6](https://drive.google.com/drive/folders/1O8mxsRlOpO-JmEz3A1LfJ_vG5NK1td3b?usp=share_linkh) | [bone_config](/configs/ctrgcn/ntu120_xset_LT_ctrgcn/b.py): [60.2](https://drive.google.com/drive/folders/1O8mxsRlOpO-JmEz3A1LfJ_vG5NK1td3b?usp=share_link) | [joint_motion_config](/configs/ctrgcn/ntu120_xset_LT_ctrgcn/jm.py): [58.2](https://drive.google.com/drive/folders/1O8mxsRlOpO-JmEz3A1LfJ_vG5NK1td3b?usp=share_linkh) | [bone_motion_config](/configs/ctrgcn/ntu120_xset_LT_ctrgcn/bm.py): [56.2](https://drive.google.com/drive/folders/1O8mxsRlOpO-JmEz3A1LfJ_vG5NK1td3b?usp=share_link) | 64.3 | 66.0 |

We also provide numerous checkpoints trained with BFL (Balanced Representation Learning) on NTURGB+D. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Skip Top1 | Joint Motion Top1 | Bone Motion Top1 | Skip Motion Top1 | Two-Stream Top1 | Four Stream Top1 | Six Stream Top1|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xsub/j.py): [76.9](https://drive.google.com/drive/folders/1lWgduccLI_Oc1JuOJQXhU6I9IFfFlbxc?usp=share_link) | [bone_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xsub/b.py): [77.3](https://drive.google.com/drive/folders/1lWgduccLI_Oc1JuOJQXhU6I9IFfFlbxc?usp=share_link) | [skip_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xsub/k.py): [76.7](https://drive.google.com/drive/folders/1lWgduccLI_Oc1JuOJQXhU6I9IFfFlbxc?usp=share_link) | [joint_motion_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xsub/jm.py): [73.0](https://drive.google.com/drive/folders/1lWgduccLI_Oc1JuOJQXhU6I9IFfFlbxc?usp=share_link) | [bone_motion_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xsub/bm.py): [72.9](https://drive.google.com/drive/folders/1lWgduccLI_Oc1JuOJQXhU6I9IFfFlbxc?usp=share_link) | [skip_motion_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xsub/km.py): [73.3](https://drive.google.com/drive/folders/1lWgduccLI_Oc1JuOJQXhU6I9IFfFlbxc?usp=share_link) | 80.3 | 81.2 | 81.8 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xview/j.py): [80.7](https://drive.google.com/drive/folders/1CxaWXRbJ0_E3Sgj3Fan6rwDCjzIpNXfg?usp=share_link) | [bone_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xview/b.py): [79.8](https://drive.google.com/drive/folders/1CxaWXRbJ0_E3Sgj3Fan6rwDCjzIpNXfg?usp=share_link) | [skip_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xview/k.py): [79.9](https://drive.google.com/drive/folders/1CxaWXRbJ0_E3Sgj3Fan6rwDCjzIpNXfg?usp=share_linkh) | [joint_motion_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xview/jm.py): [78.9](https://drive.google.com/drive/folders/1CxaWXRbJ0_E3Sgj3Fan6rwDCjzIpNXfg?usp=share_link) | [bone_motion_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xview/bm.py): [75.9](https://drive.google.com/drive/folders/1CxaWXRbJ0_E3Sgj3Fan6rwDCjzIpNXfg?usp=share_linkh) | [skip_motion_config](/configs/ctrgcn/ctrgcn_BFL_ntu60_xview/km.py): [77.2](https://drive.google.com/drive/folders/1CxaWXRbJ0_E3Sgj3Fan6rwDCjzIpNXfg?usp=share_link) | 83.1 | 84.7 | 85.0 |


**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion. For Six-Stream results, we adopt the **2 (Joint):2 (Bone):2 (Skip):1 (Joint Motion):1 (Bone Motion):1 (Skip Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train CTRGCN on NTURGB+D XSub (Joint Modality) with one GPU, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/ctrgcn/ntu60_xsub_LT_ctrgcn/j.py 1 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test CTRGCN on NTURGB+D XSub (Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/ctrgcn/ntu60_xsub_LT_ctrgcn/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

You can use the following command to ensemble the results of different modalities.
```
cd ./tools
python ensemble.py
```
