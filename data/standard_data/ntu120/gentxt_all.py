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


def get_indices(performer, setup, evaluation='CSub'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'CSub':  # Cross Subject (Subject IDs)
        train_ids = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
                     31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
                     58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
                     93, 94, 95, 97, 98, 100, 103]
        test_ids = [i for i in range(1, 107) if i not in train_ids]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(np.int32)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int32)
    else:  # Cross Setup (Setup IDs)
        train_ids = [i for i in range(1, 33) if i % 2 == 0]  # Even setup
        test_ids = [i for i in range(1, 33) if i % 2 == 1]  # Odd setup

        # Get indices of test data
        for test_id in test_ids:
            temp = np.where(setup == test_id)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(np.int32)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(setup == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int32)

    return train_indices, test_indices


def split_dataset(performer, setup, evaluation, skes_name, save_path):
    train_indices, test_indices = get_indices(performer, setup, evaluation)

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
    setup = np.loadtxt(setup_file, dtype=np.int32)  # camera id: 1~32
    performer = np.loadtxt(performer_file, dtype=np.int32)  # subject id: 1~106

    with open(skes_name_file, 'r') as f:
            skes_name = [
                line.strip() for line in f.readlines()
            ]

    save_path = 'NTU120_xsub_train.txt'
    evaluation = 'CSub'
    # save_path = 'NTU120_xset_train.txt'
    # evaluation = 'CSet'
    
    split_dataset(performer, setup, evaluation, skes_name, save_path)
