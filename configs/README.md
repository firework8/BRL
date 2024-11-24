# Configs

## Introduction

This directory includes configs for different recognition algorithms. We provide the results trained on NTU RGB+D 60, NTU RGB+D 120 and Kinetics-400 in the long-tailed training setting. We also provide checkpoints for six modalities: Joint, Bone, Skip, Joint Motion, Bone Motion, and Skip Motion. The accuracy of each modality links to the weight file.

## BRL Model Zoo

We release numerous checkpoints for BRL trained with various modalities, annotations on NTU RGB+D 60, NTU RGB+D 120 and Kinetics-400. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Skip Top1 | Joint Motion Top1 | Bone Motion Top1 | Skip Motion Top1 | Two-Stream Top1 | Four Stream Top1 | Six Stream Top1|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/BRL/ntu60_xsub_LT/j.py): [76.7](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [bone_config](/configs/BRL/ntu60_xsub_LT/b.py): [76.1](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [skip_config](/configs/BRL/ntu60_xsub_LT/k.py): [77.7](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [joint_motion_config](/configs/BRL/ntu60_xsub_LT/jm.py): [75.0](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [bone_motion_config](/configs/BRL/ntu60_xsub_LT/bm.py): [72.8](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [skip_motion_config](/configs/BRL/ntu60_xsub_LT/km.py): [73.4](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | 79.6 | 81.0 | 81.8 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/BRL/ntu60_xview_LT/j.py): [81.4](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [bone_config](/configs/BRL/ntu60_xview_LT/b.py): [80.3](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [skip_config](/configs/BRL/ntu60_xview_LT/k.py): [81.1](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [joint_motion_config](/configs/BRL/ntu60_xview_LT/jm.py): [78.5](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [bone_motion_config](/configs/BRL/ntu60_xview_LT/bm.py): [76.2](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [skip_motion_config](/configs/BRL/ntu60_xview_LT/km.py): [77.2](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | 84.0 | 84.9 | 85.4 |
| NTURGB+D 120 XSub | Official 3D Skeleton | [joint_config](/configs/BRL/ntu120_xsub_LT/j.py): [65.3](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [bone_config](/configs/BRL/ntu120_xsub_LT/b.py): [65.3](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [skip_config](/configs/BRL/ntu120_xsub_LT/k.py): [64.2](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [joint_motion_config](/configs/BRL/ntu120_xsub_LT/jm.py): [59.7](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [bone_motion_config](/configs/BRL/ntu120_xsub_LT/bm.py): [59.8](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [skip_motion_config](/configs/BRL/ntu120_xsub_LT/km.py): [59.6](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | 68.7 | 69.4 | 69.7 |
| NTURGB+D 120 XSet | Official 3D Skeleton | [joint_config](/configs/BRL/ntu120_xset_LT/j.py): [66.8](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [bone_config](/configs/BRL/ntu120_xset_LT/b.py): [66.6](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [skip_config](/configs/BRL/ntu120_xset_LT/k.py): [65.9](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [joint_motion_config](/configs/BRL/ntu120_xset_LT/jm.py): [63.5](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [bone_motion_config](/configs/BRL/ntu120_xset_LT/bm.py): [62.2](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [skip_motion_config](/configs/BRL/ntu120_xset_LT/km.py): [61.6](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | 69.7 | 71.0 | 71.3 |
| Kinetics-400 | HRNet 2D Pose | [joint_config](/configs/BRL/k400/j.py): [45.6](https://drive.google.com/drive/folders/1xIFYdvACmDy2aLr4EvuuVNAsobze0-SY) | [bone_config](/configs/BRL/k400/b.py): [45.2](https://drive.google.com/drive/folders/1xIFYdvACmDy2aLr4EvuuVNAsobze0-SY) |  | [joint_motion_config](/configs/BRL/k400/jm.py): [42.0](https://drive.google.com/drive/folders/1xIFYdvACmDy2aLr4EvuuVNAsobze0-SY) | [bone_motion_config](/configs/BRL/k400/bm.py): [41.2](https://drive.google.com/drive/folders/1xIFYdvACmDy2aLr4EvuuVNAsobze0-SY) |  | 48.1 (1:1) | 48.6 (3:3:1:1) |  |

## Supported Algorithms

- [x] [ST-GCN (AAAI 2018)](https://arxiv.org/abs/1801.07455) [[MODELZOO](/configs/stgcn/README.md)]
- [x] [AAGCN (TIP)](https://arxiv.org/abs/1912.06971) [[MODELZOO](/configs/aagcn/README.md)]
- [x] [MS-G3D (CVPR 2020 Oral)](https://arxiv.org/abs/2003.14111) [[MODELZOO](/configs/msg3d/README.md)]
- [x] [CTR-GCN (ICCV 2021)](https://arxiv.org/abs/2107.12213) [[MODELZOO](/configs/ctrgcn/README.md)]

## Other Algorithms
We test the performance of Shift-GCN, DC-GCN+ADG, MST-GCN, and InfoGCN on the NTURGB+D 60-LT and NTURGB+D 120-LT datasets.

#### Shift-GCN

| Dataset | Annotation | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone Motion Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Skeleton | [63.7](https://drive.google.com/drive/folders/14hV4DHtPhYD872r1FJksNUsJrB80Fy_6?usp=share_link) | [61.4](https://drive.google.com/drive/folders/14hV4DHtPhYD872r1FJksNUsJrB80Fy_6?usp=share_link) | [64.3](https://drive.google.com/drive/folders/14hV4DHtPhYD872r1FJksNUsJrB80Fy_6?usp=share_link) | [62.5](https://drive.google.com/drive/folders/14hV4DHtPhYD872r1FJksNUsJrB80Fy_6?usp=share_link) | 73.6 |
| NTURGB+D XView | Skeleton | [71.3](https://drive.google.com/drive/folders/1wYnlz8qP2OjEOWMbvOHjpHBjik1e-dSE?usp=share_link) | [69.7](https://drive.google.com/drive/folders/1wYnlz8qP2OjEOWMbvOHjpHBjik1e-dSE?usp=share_link) | [68.2](https://drive.google.com/drive/folders/1wYnlz8qP2OjEOWMbvOHjpHBjik1e-dSE?usp=share_link) | [68.8](https://drive.google.com/drive/folders/1wYnlz8qP2OjEOWMbvOHjpHBjik1e-dSE?usp=share_link) | 79.3 |
| NTURGB+D 120 XSub | Skeleton | [54.7](https://drive.google.com/drive/folders/1QJEbT2UKCSCACZ4NRCeGRVMC8AKxkT6Y?usp=share_link) | [54.2](https://drive.google.com/drive/folders/1QJEbT2UKCSCACZ4NRCeGRVMC8AKxkT6Y?usp=share_link) | [51.2](https://drive.google.com/drive/folders/1QJEbT2UKCSCACZ4NRCeGRVMC8AKxkT6Y?usp=share_link) | [53.2](https://drive.google.com/drive/folders/1QJEbT2UKCSCACZ4NRCeGRVMC8AKxkT6Y?usp=share_link) | 62.3 |
| NTURGB+D 120 XSet | Skeleton | [55.4](https://drive.google.com/drive/folders/1AJmbBA5KjaV7FYFrNueQi44UtenD1Rbz?usp=share_link) | [54.4](https://drive.google.com/drive/folders/1AJmbBA5KjaV7FYFrNueQi44UtenD1Rbz?usp=share_link) | [54.6](https://drive.google.com/drive/folders/1AJmbBA5KjaV7FYFrNueQi44UtenD1Rbz?usp=share_link) | [55.1](https://drive.google.com/drive/folders/1AJmbBA5KjaV7FYFrNueQi44UtenD1Rbz?usp=share_link) | 64.5 |

#### DC-GCN+ADG

| Dataset | Annotation | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone Motion Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Skeleton | [68.9](https://drive.google.com/drive/folders/1R5iT9W4WuAsyfmYn3_Yv0XDWGw6gksZe?usp=share_link) | [65.1](https://drive.google.com/drive/folders/1R5iT9W4WuAsyfmYn3_Yv0XDWGw6gksZe?usp=share_link) | [61.1](https://drive.google.com/drive/folders/1R5iT9W4WuAsyfmYn3_Yv0XDWGw6gksZe?usp=share_link) | [60.5](https://drive.google.com/drive/folders/1R5iT9W4WuAsyfmYn3_Yv0XDWGw6gksZe?usp=share_link) | 75.0 |
| NTURGB+D XView | Skeleton | [74.9](https://drive.google.com/drive/folders/1-0rYWegk-b8NzNWloaxJ1OOFOjE1y_eu?usp=share_link) | [69.9](https://drive.google.com/drive/folders/1-0rYWegk-b8NzNWloaxJ1OOFOjE1y_eu?usp=share_link) | [68.1](https://drive.google.com/drive/folders/1-0rYWegk-b8NzNWloaxJ1OOFOjE1y_eu?usp=share_link) | [67.8](https://drive.google.com/drive/folders/1-0rYWegk-b8NzNWloaxJ1OOFOjE1y_eu?usp=share_link) | 79.7 |
| NTURGB+D 120 XSub | Skeleton | [56.1](https://drive.google.com/drive/folders/1n-1ZTGhi2Z0bWZLq-L_pMY5hIOB8yZvu?usp=share_link) | [55.5](https://drive.google.com/drive/folders/1n-1ZTGhi2Z0bWZLq-L_pMY5hIOB8yZvu?usp=share_link) | [50.1](https://drive.google.com/drive/folders/1n-1ZTGhi2Z0bWZLq-L_pMY5hIOB8yZvu?usp=share_link) | [51.2](https://drive.google.com/drive/folders/1n-1ZTGhi2Z0bWZLq-L_pMY5hIOB8yZvu?usp=share_link) | 63.4 |
| NTURGB+D 120 XSet | Skeleton | [59.6](https://drive.google.com/drive/folders/1XJmwA_MMzRUJnlt7tmdKUNK-jrVKxlpc?usp=share_link) | [56.7](https://drive.google.com/drive/folders/1XJmwA_MMzRUJnlt7tmdKUNK-jrVKxlpc?usp=share_link) | [55.6](https://drive.google.com/drive/folders/1XJmwA_MMzRUJnlt7tmdKUNK-jrVKxlpc?usp=share_link) | [53.9](https://drive.google.com/drive/folders/1XJmwA_MMzRUJnlt7tmdKUNK-jrVKxlpc?usp=share_link) | 66.2 |

#### MST-GCN

| Dataset | Annotation | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone Motion Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Skeleton | [70.1](https://drive.google.com/drive/folders/1hsn2dFT9qx7_lAN6NThvjQExhAJIn9IB?usp=share_link) | [70.2](https://drive.google.com/drive/folders/1hsn2dFT9qx7_lAN6NThvjQExhAJIn9IB?usp=share_link) | [67.2](https://drive.google.com/drive/folders/1hsn2dFT9qx7_lAN6NThvjQExhAJIn9IB?usp=share_link) | [68.0](https://drive.google.com/drive/folders/1hsn2dFT9qx7_lAN6NThvjQExhAJIn9IB?usp=share_link) | 75.9 |
| NTURGB+D XView | Skeleton | [76.7](https://drive.google.com/drive/folders/1De0EplTEfB9utITPCoOqdVFCAv-DT4tC?usp=share_link) | [75.6](https://drive.google.com/drive/folders/1De0EplTEfB9utITPCoOqdVFCAv-DT4tC?usp=share_link) | [73.2](https://drive.google.com/drive/folders/1De0EplTEfB9utITPCoOqdVFCAv-DT4tC?usp=share_link) | [72.9](https://drive.google.com/drive/folders/1De0EplTEfB9utITPCoOqdVFCAv-DT4tC?usp=share_link) | 80.3 |
| NTURGB+D 120 XSub | Skeleton | [57.5](https://drive.google.com/drive/folders/1rF6CjboejxfvVDcYObIpaiM3O-_33q45?usp=share_link) | [59.9](https://drive.google.com/drive/folders/1rF6CjboejxfvVDcYObIpaiM3O-_33q45?usp=share_link) | [54.6](https://drive.google.com/drive/folders/1rF6CjboejxfvVDcYObIpaiM3O-_33q45?usp=share_link) | [56.2](https://drive.google.com/drive/folders/1rF6CjboejxfvVDcYObIpaiM3O-_33q45?usp=share_link) | 63.8 |
| NTURGB+D 120 XSet | Skeleton | [61.6](https://drive.google.com/drive/folders/1ISJFkzyrVg1JkGU3Zd4mVV1B3Ztf20Oc?usp=share_link) | [60.6](https://drive.google.com/drive/folders/1ISJFkzyrVg1JkGU3Zd4mVV1B3Ztf20Oc?usp=share_link) | [58.2](https://drive.google.com/drive/folders/1ISJFkzyrVg1JkGU3Zd4mVV1B3Ztf20Oc?usp=share_link) | [59.0](https://drive.google.com/drive/folders/1ISJFkzyrVg1JkGU3Zd4mVV1B3Ztf20Oc?usp=share_link) | 65.9 |

#### InfoGCN

| Dataset | Annotation | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone Motion Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Skeleton | [69.7](https://drive.google.com/drive/folders/1Lj3Xxcsgtu_UcMrCnfaBvXk5vqL9MXR-?usp=share_link) | [66.6](https://drive.google.com/drive/folders/1Lj3Xxcsgtu_UcMrCnfaBvXk5vqL9MXR-?usp=share_link) | [68.5](https://drive.google.com/drive/folders/1Lj3Xxcsgtu_UcMrCnfaBvXk5vqL9MXR-?usp=share_link) | [67.5](https://drive.google.com/drive/folders/1Lj3Xxcsgtu_UcMrCnfaBvXk5vqL9MXR-?usp=share_link) | 76.8 |
| NTURGB+D XView | Skeleton | [73.5](https://drive.google.com/drive/folders/1JSCTcOfym_MDbQ-Mjiq5HWpmL0GiBjyA?usp=share_link) | [68.5](https://drive.google.com/drive/folders/1JSCTcOfym_MDbQ-Mjiq5HWpmL0GiBjyA?usp=share_link) | [72.2](https://drive.google.com/drive/folders/1JSCTcOfym_MDbQ-Mjiq5HWpmL0GiBjyA?usp=share_link) | [71.5](https://drive.google.com/drive/folders/1JSCTcOfym_MDbQ-Mjiq5HWpmL0GiBjyA?usp=share_link) | 79.2 |
| NTURGB+D 120 XSub | Skeleton | [55.6](https://drive.google.com/drive/folders/14JI46wvmRRMcK4dp9DWgYxbbHmfaLuID?usp=share_link) | [60.4](https://drive.google.com/drive/folders/14JI46wvmRRMcK4dp9DWgYxbbHmfaLuID?usp=share_link) | [52.3](https://drive.google.com/drive/folders/14JI46wvmRRMcK4dp9DWgYxbbHmfaLuID?usp=share_link) | [52.1](https://drive.google.com/drive/folders/14JI46wvmRRMcK4dp9DWgYxbbHmfaLuID?usp=share_link) | 64.2 |
| NTURGB+D 120 XSet | Skeleton | [58.5](https://drive.google.com/drive/folders/1w3ZcUTWvlYTvgeHZ7aiIQ6-oqLJXyh8V?usp=share_link) | [60.8](https://drive.google.com/drive/folders/1w3ZcUTWvlYTvgeHZ7aiIQ6-oqLJXyh8V?usp=share_link) | [57.8](https://drive.google.com/drive/folders/1w3ZcUTWvlYTvgeHZ7aiIQ6-oqLJXyh8V?usp=share_link) | [55.7](https://drive.google.com/drive/folders/1w3ZcUTWvlYTvgeHZ7aiIQ6-oqLJXyh8V?usp=share_link) | 67.1 |

## Citation
```BibTeX
% ST-GCN
@inproceedings{yan2018spatial,
title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
booktitle={Thirty-second AAAI conference on artificial intelligence},
year={2018}
}
% AAGCN
@article{shi2020skeleton,
  title={Skeleton-based action recognition with multi-stream adaptive graph convolutional networks},
  author={Shi, Lei and Zhang, Yifan and Cheng, Jian and Lu, Hanqing},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={9532--9545},
  year={2020},
  publisher={IEEE}
}
% MS-G3D
@inproceedings{liu2020disentangling,
title={Disentangling and unifying graph convolutions for skeleton-based action recognition},
author={Liu, Ziyu and Zhang, Hongwen and Chen, Zhenghao and Wang, Zhiyong and Ouyang, Wanli},
booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
pages={143--152},
year={2020}
}
% CTR-GCN
@inproceedings{chen2021channel,
title={Channel-wise topology refinement graph convolution for skeleton-based action recognition},
author={Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={13359--13368},
year={2021}
}
% Shift-GCN
@inproceedings{cheng2020skeleton,
title={Skeleton-based action recognition with shift graph convolutional network},
author={Cheng, Ke and Zhang, Yifan and He, Xiangyu and Chen, Weihan and Cheng, Jian and Lu, Hanqing},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={183--192},
year={2020}
}
% DC-GCN+ADG
@inproceedings{cheng2020decoupling,
  title={Decoupling gcn with dropgraph module for skeleton-based action recognition},
  author={Cheng, Ke and Zhang, Yifan and Cao, Congqi and Shi, Lei and Cheng, Jian and Lu, Hanqing},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part XXIV 16},
  pages={536--553},
  year={2020},
  organization={Springer}
}
% MST-GCN
@inproceedings{chen2021multi,
  title={Multi-scale spatial temporal graph convolutional network for skeleton-based action recognition},
  author={Chen, Zhan and Li, Sicheng and Yang, Bing and Li, Qinghan and Liu, Hong},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={35},
  number={2},
  pages={1113--1122},
  year={2021}
}
% InfoGCN
@inproceedings{chi2022infogcn,
title={InfoGCN: Representation Learning for Human Skeleton-Based Action Recognition},
author={Chi, Hyung-gun and Ha, Myoung Hoon and Chi, Seunggeun and Lee, Sang Wan and Huang, Qixing and Ramani, Karthik},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={20186--20196},
year={2022}
}
```

