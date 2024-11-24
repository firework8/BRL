# BRL
PyTorch implementation of "Balanced Representation Learning for Long-tailed Skeleton-based Action Recognition" [[PDF](https://arxiv.org/pdf/2308.14024.pdf)]

## Dependencies

- Python >= 3.6
- PyTorch >= 1.5.0
- PyYAML, tqdm, tensorboardX
- We provide the dependency file of the experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt `
- Run `pip install -e .` 

## Data Preparation

### Download Datasets

There are 4 datasets to download:

- NTU RGB+D 60
- NTU RGB+D 120
- NW-UCLA
- Kinetics-Skeleton

Please check [Data Download](/data/pyskl_data/README.md) for the download links and descriptions of the annotation format. After finishing the download and placement, the BRL code is now ready to run.

### Instruction for Data Processing

We provide the long-tailed settings for [NTU 60-LT](/data/NTU60_LT) and [NTU 120-LT](/data/NTU120_LT), which are generated by [imblance_gentxt](/data/imblance_gentxt.py). Additionally, we use [ucla_data](data/ucla_data.py) to construct NW-UCLA-LT. Through the config, BRL will directly apply the corresponding long-tailed settings during training.

If you would like to construct the experiments with classical methods, a detailed download instructions is provided [here](/data/utils/README.md). You could follow the official processing of these classical methods for the raw data.

For a fair comparison, we provide the processing script [imblance_gendata_from_txt](/data/imblance_gendata_from_txt.py) that helps to generate the same long-tailed version of the raw data.

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# For example: train BRL on NTURGB+D XSub (Joint Modality) with one GPU, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/BRL/ntu60_xsub_LT/j.py 1 --validate --test-last --test-best
```
```shell
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval {optional_arguments}
# For example: test BRL on NTURGB+D XSub (Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/BRL/ntu60_xsub_LT/j.py checkpoints/SOME_CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```
```shell
# Ensemble the results
cd ./tools
python ensemble.py
```

## Experimental Results

We provide the results trained on NTU RGB+D 60, NTU RGB+D 120 and Kinetics-400 in the long-tailed training setting. We also provide checkpoints for six modalities: Joint, Bone, Skip, Joint Motion, Bone Motion, and Skip Motion. The accuracy of each modality links to the weight file.

### BRL Checkpoint

We release numerous checkpoints for BRL trained with various modalities, annotations on NTU RGB+D 60, NTU RGB+D 120 and Kinetics-400. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Skip Top1 | Joint Motion Top1 | Bone Motion Top1 | Skip Motion Top1 | Two-Stream Top1 | Four Stream Top1 | Six Stream Top1|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/BRL/ntu60_xsub_LT/j.py): [76.7](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [bone_config](/configs/BRL/ntu60_xsub_LT/b.py): [76.1](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [skip_config](/configs/BRL/ntu60_xsub_LT/k.py): [77.7](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [joint_motion_config](/configs/BRL/ntu60_xsub_LT/jm.py): [75.0](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [bone_motion_config](/configs/BRL/ntu60_xsub_LT/bm.py): [72.8](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [skip_motion_config](/configs/BRL/ntu60_xsub_LT/km.py): [73.4](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | 79.6 | 81.0 | 81.8 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/BRL/ntu60_xview_LT/j.py): [81.4](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [bone_config](/configs/BRL/ntu60_xview_LT/b.py): [80.3](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [skip_config](/configs/BRL/ntu60_xview_LT/k.py): [81.1](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [joint_motion_config](/configs/BRL/ntu60_xview_LT/jm.py): [78.5](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [bone_motion_config](/configs/BRL/ntu60_xview_LT/bm.py): [76.2](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [skip_motion_config](/configs/BRL/ntu60_xview_LT/km.py): [77.2](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | 84.0 | 84.9 | 85.4 |
| NTURGB+D 120 XSub | Official 3D Skeleton | [joint_config](/configs/BRL/ntu120_xsub_LT/j.py): [65.3](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [bone_config](/configs/BRL/ntu120_xsub_LT/b.py): [65.3](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [skip_config](/configs/BRL/ntu120_xsub_LT/k.py): [64.2](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [joint_motion_config](/configs/BRL/ntu120_xsub_LT/jm.py): [59.7](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [bone_motion_config](/configs/BRL/ntu120_xsub_LT/bm.py): [59.8](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [skip_motion_config](/configs/BRL/ntu120_xsub_LT/km.py): [59.6](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | 68.7 | 69.4 | 69.7 |
| NTURGB+D 120 XSet | Official 3D Skeleton | [joint_config](/configs/BRL/ntu120_xset_LT/j.py): [66.8](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [bone_config](/configs/BRL/ntu120_xset_LT/b.py): [66.6](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [skip_config](/configs/BRL/ntu120_xset_LT/k.py): [65.9](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [joint_motion_config](/configs/BRL/ntu120_xset_LT/jm.py): [63.5](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [bone_motion_config](/configs/BRL/ntu120_xset_LT/bm.py): [62.2](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [skip_motion_config](/configs/BRL/ntu120_xset_LT/km.py): [61.6](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | 69.7 | 71.0 | 71.3 |
| Kinetics-400 | HRNet 2D Pose | [joint_config](/configs/BRL/k400/j.py): [45.6](https://drive.google.com/drive/folders/1xIFYdvACmDy2aLr4EvuuVNAsobze0-SY) | [bone_config](/configs/BRL/k400/b.py): [45.2](https://drive.google.com/drive/folders/1xIFYdvACmDy2aLr4EvuuVNAsobze0-SY) |  | [joint_motion_config](/configs/BRL/k400/jm.py): [42.0](https://drive.google.com/drive/folders/1xIFYdvACmDy2aLr4EvuuVNAsobze0-SY) | [bone_motion_config](/configs/BRL/k400/bm.py): [41.2](https://drive.google.com/drive/folders/1xIFYdvACmDy2aLr4EvuuVNAsobze0-SY) |  | 48.1 (1:1) | 48.6 (3:3:1:1) |  |

### Supported Algorithms

- [x] [ST-GCN (AAAI 2018)](https://arxiv.org/abs/1801.07455) [[Model Zoo](/configs/stgcn/README.md)]
- [x] [AAGCN (TIP)](https://arxiv.org/abs/1912.06971) [[Model Zoo](/configs/aagcn/README.md)]
- [x] [MS-G3D (CVPR 2020 Oral)](https://arxiv.org/abs/2003.14111) [[Model Zoo](/configs/msg3d/README.md)]
- [x] [CTR-GCN (ICCV 2021)](https://arxiv.org/abs/2107.12213) [[Model Zoo](/configs/ctrgcn/README.md)]

For specific examples and other pre-trained models, please go to the [README](/configs/README.md).


## Acknowledgements

This repo is based on [MS-G3D](https://github.com/kenziyuliu/ms-g3d), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), and [PYSKL](https://github.com/kennymckormick/pyskl).

Thanks to the authors for their work!

## Citation

```
@article{liu2023balanced,
  title={Balanced Representation Learning for Long-tailed Skeleton-based Action Recognition},
  author={Liu, Hongda and Wang, Yunlong and Ren, Min and Hu, Junxing and Luo, Zhengquan and Hou, Guangqi and Sun, Zhenan},
  journal={arXiv preprint arXiv:2308.14024},
  year={2023}
}
```

## Contact
For any questions, feel free to contact: `hongda.liu@cripac.ia.ac.cn`
