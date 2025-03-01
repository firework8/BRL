
# Download Raw Data

### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `/data/nturgbd_raw`
3. Download missing skeletons lookup files：
   1. NTU RGB+D 60 Missing Skeletons: `wget https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt`
   2. NTU RGB+D 120 Missing Skeletons: `wget https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD120_samples_with_missing_skeletons.txt`

### NW-UCLA

1. Download dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Move `all_sqe` to `/data/NW-UCLA`

### Kinetics Skeleton 400

1. Download the data from [GoogleDrive](https://drive.google.com/drive/folders/1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ) provided by [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md#kinetics-skeleton)
2. [This](https://silicondales.com/tutorials/g-suite/how-to-wget-files-from-google-drive/) might be useful if you want to `wget` the dataset from Google Drive

