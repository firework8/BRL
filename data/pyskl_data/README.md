# Things you need to know about PYSKL data format

[PYSKL](https://github.com/kennymckormick/pyskl) provides pre-processed pickle annotations files for training and testing. Below we provide the download links and demonstrate the format of the annotation files.

## Download the pre-processed skeletons

We provide links to the pre-processed skeleton annotations.

- NTURGB+D: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl
- NTURGB+D 120: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl
- NW-UCLA: https://drive.google.com/file/d/1GEK5ORAHtVBiIKpBCDfDhU4L7tPDHrMe/view?usp=share_link
- Kinetics-Skeleton: https://download.openmmlab.com/mmaction/pyskl/data/k400/k400_hrnet.pkl (Table of contents only, no skeleton annotations)

For Kinetics-Skeleton, please use the following link to download the `kpfiles` and extract it under `$BRL/data/k400` for Kinetics-Skeleton training & testing: https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EeyDCVskqLtClMVVwqD53acBF2FEwkctp3vtRbkLfnKSTw?e=B3SZlM

Note that the `kpfiles` needs to be extracted under `Linux`. The Kinetics-Skeleton requires `Memcached` to run, which can be referred to [here](https://www.runoob.com/memcached/memcached-install.html).

## The format of the pickle files

Each pickle file corresponds to an action recognition dataset. The content of a pickle file is a dictionary with two fields: `split` and `annotations`

1. Split: The value of the `split` field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
2. Annotations: The value of the `annotations` field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
   1. `frame_dir` (str): The identifier of the corresponding video.
   2. `total_frames` (int): The number of frames in this video.
   3. `img_shape` (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
   4. `original_shape` (tuple[int]): Same as `img_shape`.
   5. `label` (int): The action label.
   6. `keypoint` (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as `total_frames`); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
   7. `keypoint_score` (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.

Note:
1. For Kinetics-Skeleton, things are a little different (for storage saving and training acceleration):
   1. The fields `keypoint`, `keypoint_score` are not in the annotation file, but stored in many different **kpfiles**.
   2. A new field named `raw_file`, which specifies the file path of the **kpfile** that contains the skeleton annotation of this video.
   3. Each **kpfile** is a dictionary: key is the `frame_dir`, value is a dictionary with a single key `keypoint`. The value of `keypoint` is an ndarray with shape [N x V x C]. N: number of skeletons in the video; V: number of keypoints; C (C=3): number of dimensions for keypoint (x, y, score).
   4. A new field named `frame_inds`, indicates the corresponding frame index of each skeleton.
   5. A new field named `box_score`, indicates the corresponding bbox score of each skeleton.
   6. A new field named `valid`, indicates how many frames (with valid skeletons) left when we only keep skeletons with bbox scores larger than a threshold.
   7. We cache the kpfiles in memory with memcache and query with `frame_dir` to obtain the skeleton annotation. Kinetics-400 skeletons are converted to normal skeleton format with operator `DecompressPose`.

You can download an annotation file and browse it to get familiar with our annotation formats.

## Process NTURGB+D raw skeleton files

If you would like to construct the above data yourself, you could refer to the steps below:

1. Assume that you are using the current directory as the working directory.
2. Download the raw skeleton files from the [official repo of NTURGB+D](https://github.com/shahroudy/NTURGB-D/), unzip and place all `.skeleton` files in a single folder  (named `nturgb+d_skeletons` in my example).
3. Run `python ntu_preproc.py` to generate processed skeleton annotations, it will generate `ntu60_3danno.pkl` and `ntu120_3danno.pkl` (If you also downloaded the NTURGB+D 120 skeletons) under your current working directory.

PS: For the best pre-processing speed, change `num_process` in `ntu_preproc.py` to the number of cores that your CPU has.

### BibTex items for each provided dataset

```BibTex
% NTURGB+D
@inproceedings{shahroudy2016ntu,
  title={Ntu rgb+ d: A large scale dataset for 3d human activity analysis},
  author={Shahroudy, Amir and Liu, Jun and Ng, Tian-Tsong and Wang, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1010--1019},
  year={2016}
}
% NTURGB+D 120
@article{liu2019ntu,
  title={Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding},
  author={Liu, Jun and Shahroudy, Amir and Perez, Mauricio and Wang, Gang and Duan, Ling-Yu and Kot, Alex C},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={42},
  number={10},
  pages={2684--2701},
  year={2019},
  publisher={IEEE}
}
% NW-UCLA
@inproceedings{wang2014cross,
title={Cross-view action modeling, learning and recognition},
author={Wang, Jiang and Nie, Xiaohan and Xia, Yin and Wu, Ying and Zhu, Song-Chun},
booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
pages={2649--2656},
year={2014}
}
% Kinetics-400
@inproceedings{carreira2017quo,
  title={Quo vadis, action recognition? a new model and the kinetics dataset},
  author={Carreira, Joao and Zisserman, Andrew},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6299--6308},
  year={2017}
}
% PYSKL
@inproceedings{duan2022pyskl,
  title={Pyskl: Towards good practices for skeleton action recognition},
  author={Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={7351--7354},
  year={2022}
}
```
