# BRL
PyTorch implementation of "Balanced Representation Learning for Long-tailed Skeleton-based Action Recognition"

## Dependencies

- Python >= 3.6
- PyTorch >= 1.5.0
- PyYAML, tqdm, tensorboardX
- We provide the dependency file of the experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt `
- Run `pip install -e .` 

## Data Preparation

### Download datasets

There are 4 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA
- Kinetics 400 Skeleton

Please download the raw data from [NTU RGB+D 60](https://rose1.ntu.edu.sg/dataset/actionRecognition), [NTU RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition), [NW-UCLA](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0), and [Kinetics Skeleton 400](https://drive.google.com/drive/folders/1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ).

For detailed download instructions, please go to the [README](/data/README.md).

### Data Processing

#### Generating Data
Generate NTU RGB+D 60, NTU RGB+D 120, NW-UCLA, and Kinetics Skeleton 400 datasets:
```
  cd ./data
  python ntu_gendata.py
  python ntu120_gendata.py
  python ucla_data.py
  python kinetics_gendata.py
```
Meanwhile, [PYSKL](https://github.com/kennymckormick/pyskl#data-preparation) provides the processed skeleton data for all datasets as pickle files (which can be directly used for training and testing).
Please check [Data Doc](/data/pyskl_data/README.md) for the download links and descriptions of the annotation format.
You can also use the [provided script](/data/pyskl_data/ntu_preproc.py) to generate the processed pickle files. 

#### Constructing Long-tailed Datasets
Construct NTU 60-LT, NTU 120-LT, and NW-UCLA-LT：
```
  cd ./data
  python imblance_gentxt.py
  python imblance_gendata_from_txt.py
```

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

## Experimental results

We provide the results trained on NTURGB+D 60 and NTURGB+D 120 in the **long-tailed** training setting. We also provide checkpoints for six modalities: Joint, Bone, Skip, Joint Motion, Bone Motion, and Skip Motion. The accuracy of each modality links to the weight file.

### BRL Model Zoo

We release numerous checkpoints for BRL trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | Joint Top1 | Bone Top1 | Skip Top1 | Joint Motion Top1 | Bone Motion Top1 | Skip Motion Top1 | Two-Stream Top1 | Four Stream Top1 | Six Stream Top1|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | [joint_config](/configs/BRL/ntu60_xsub_LT/j.py): [76.7](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [bone_config](/configs/BRL/ntu60_xsub_LT/b.py): [76.1](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [skip_config](/configs/BRL/ntu60_xsub_LT/k.py): [77.7](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [joint_motion_config](/configs/BRL/ntu60_xsub_LT/jm.py): [75.0](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [bone_motion_config](/configs/BRL/ntu60_xsub_LT/bm.py): [72.8](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | [skip_motion_config](/configs/BRL/ntu60_xsub_LT/km.py): [73.4](https://drive.google.com/drive/folders/1ksC002PtEMxCt8A5l5ftqSN9guBzJxy0?usp=share_link) | 79.6 | 81.0 | 81.8 |
| NTURGB+D XView | Official 3D Skeleton | [joint_config](/configs/BRL/ntu60_xview_LT/j.py): [81.4](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [bone_config](/configs/BRL/ntu60_xview_LT/b.py): [80.3](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [skip_config](/configs/BRL/ntu60_xview_LT/k.py): [81.1](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [joint_motion_config](/configs/BRL/ntu60_xview_LT/jm.py): [78.5](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [bone_motion_config](/configs/BRL/ntu60_xview_LT/bm.py): [76.2](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | [skip_motion_config](/configs/BRL/ntu60_xview_LT/km.py): [77.2](https://drive.google.com/drive/folders/1KrtXE1tdJGVJz2ixWpR6Vd7l5qzAf8TV?usp=share_link) | 84.0 | 84.9 | 85.4 |
| NTURGB+D 120 XSub | Official 3D Skeleton | [joint_config](/configs/BRL/ntu120_xsub_LT/j.py): [65.3](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [bone_config](/configs/BRL/ntu120_xsub_LT/b.py): [65.3](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [skip_config](/configs/BRL/ntu120_xsub_LT/k.py): [64.2](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [joint_motion_config](/configs/BRL/ntu120_xsub_LT/jm.py): [59.7](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [bone_motion_config](/configs/BRL/ntu120_xsub_LT/bm.py): [59.8](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | [skip_motion_config](/configs/BRL/ntu120_xsub_LT/km.py): [59.6](https://drive.google.com/drive/folders/1Lgnm_phTSM1fniYHONfzahJBdJZm36IV?usp=share_link) | 68.7 | 69.4 | 69.7 |
| NTURGB+D 120 XSet | Official 3D Skeleton | [joint_config](/configs/BRL/ntu120_xset_LT/j.py): [66.8](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [bone_config](/configs/BRL/ntu120_xset_LT/b.py): [66.6](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [skip_config](/configs/BRL/ntu120_xset_LT/k.py): [65.9](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [joint_motion_config](/configs/BRL/ntu120_xset_LT/jm.py): [63.5](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [bone_motion_config](/configs/BRL/ntu120_xset_LT/bm.py): [62.2](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | [skip_motion_config](/configs/BRL/ntu120_xset_LT/km.py): [61.6](https://drive.google.com/drive/folders/1L1mmgp-RtifmXTiWTNBU21e6Q3r7QV4i?usp=share_link) | 69.7 | 71.0 | 71.3 |

### Supported Algorithms

- [x] [ST-GCN (AAAI 2018)](https://arxiv.org/abs/1801.07455) [[Model Zoo](/configs/stgcn/README.md)]
- [x] [AAGCN (TIP)](https://arxiv.org/abs/1912.06971) [[Model Zoo](/configs/aagcn/README.md)]
- [x] [MS-G3D (CVPR 2020 Oral)](https://arxiv.org/abs/2003.14111) [[Model Zoo](/configs/msg3d/README.md)]
- [x] [CTR-GCN (ICCV 2021)](https://arxiv.org/abs/2107.12213) [[Model Zoo](/configs/ctrgcn/README.md)]

For specific examples and other pre-trained models, please go to the [README](/configs/README.md).


## Acknowledgements

This repo is based on [MS-G3D](https://github.com/kenziyuliu/ms-g3d), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), and [PYSKL](https://github.com/kennymckormick/pyskl).

Thanks to the original authors for their work!

## Contact
For any questions, feel free to contact: `hongda.liu@cripac.ia.ac.cn`
