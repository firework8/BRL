import os
import os.path as osp
import numpy as np


root_path = './'
stat_path = osp.join(root_path, 'statistics')
setup_file = osp.join(stat_path, 'setup.txt')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')


def get_indices(performer, camera, evaluation='CS'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'CS':  # Cross Subject (Subject IDs)
        train_ids = [1,  2,  4,  5,  8,  9,  13, 14, 15, 16,
                     17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        test_ids = [3,  6,  7,  10, 11, 12, 20, 21, 22, 23,
                    24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(np.int32)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int32)
    else:  # Cross View (Camera IDs)
        train_ids = [2, 3]
        test_ids = 1
        # Get indices of test data
        temp = np.where(camera == test_ids)[0]  # 0-based index
        test_indices = np.hstack((test_indices, temp)).astype(np.int32)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(camera == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int32)

    return train_indices, test_indices


def split_dataset(performer, camera, evaluation, skes_name, save_path):
    train_indices, test_indices = get_indices(performer, camera, evaluation)

    filename = []
    for i in train_indices:
        filename.append(skes_name[i])
    
    f = open(save_path,'w', encoding='utf-8')
    for name in filename:
        f.writelines(name)
        if name != filename[-1]:
            f.writelines('\n')
    f.close()


if __name__ == '__main__':
    camera = np.loadtxt(camera_file, dtype=np.int32)  # camera id: 1, 2, 3
    performer = np.loadtxt(performer_file, dtype=np.int32)  # subject id: 1~106

    with open(skes_name_file, 'r') as f:
            skes_name = [
                line.strip() for line in f.readlines()
            ]

    save_path = 'NTU60_xsub_train.txt'
    evaluation = 'CS'
    # save_path = 'NTU60_xview_train.txt'
    # evaluation = 'CV'
    
    split_dataset(performer, camera, evaluation, skes_name, save_path)
