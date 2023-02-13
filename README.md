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

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `/data/nturgbd_raw`
3. Download missing skeletons lookup files：
   1. NTU RGB+D 60 Missing Skeletons: `wget https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt`
   2. NTU RGB+D 120 Missing Skeletons: `wget https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD120_samples_with_missing_skeletons.txt`

#### NW-UCLA

1. Download dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Move `all_sqe` to `/data/NW-UCLA`

#### Kinetics Skeleton 400

1. Download the data from [GoogleDrive](https://drive.google.com/drive/folders/1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ) provided by [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md#kinetics-skeleton)
2. [This](https://silicondales.com/tutorials/g-suite/how-to-wget-files-from-google-drive/) might be useful if you want to `wget` the dataset from Google Drive

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

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion. For Six-Stream results, we adopt the **2 (Joint):2 (Bone):2 (Skip):1 (Joint Motion):1 (Bone Motion):1 (Skip Motion)** fusion.

For specific examples and pre-trained models, please go to the [README](/configs/README.md).

## Acknowledgements

This repo is based on [MS-G3D](https://github.com/kenziyuliu/ms-g3d), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), and [PYSKL](https://github.com/kennymckormick/pyskl).

Thanks to the original authors for their work!

## Contact
For any questions, feel free to contact: `hongda.liu@cripac.ia.ac.cn`
