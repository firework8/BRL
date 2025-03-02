import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.ntu_read_skeleton import read_xyz

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30
cls_num = 60

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")

def show_im_num_per_cls(skeleton_num_per_cls, cls_num):
    thecount = []
    for theclass in range(cls_num):
        thecount.append(skeleton_num_per_cls[theclass])
    for k in range(cls_num):
        print("A{}:{}".format(k+1, thecount[k]))
    print(sum(thecount))

def save_im_data(sample_name, 
                sample_label, 
                class_name,
                data_path,
                txt_path,
                out_path,
                benchmark,
                part,
                skeleton_type='exp',
                imb_factor=0.01, 
                rand_number=0):

    if part == 'train':

        new_name = []
        new_label = []

        with open(txt_path, 'r') as f:
            ske_data = [
                line.strip() for line in f.readlines()
            ]
    
        for filename in ske_data:
            action_class = int(
                filename[filename.find('A') + 1:filename.find('A') + 4])
            data_name = filename + ".skeleton"
            new_name.append(data_name)
            new_label.append(action_class - 1)
        

        with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
            pickle.dump((new_name, list(new_label)), f)
        # np.save('{}/{}_label.npy'.format(out_path, part), new_label)
        fp = open_memmap(
            '{}/{}_data.npy'.format(out_path, part),
            dtype='float32',
            mode='w+',
            shape=(len(new_label), 3, max_frame, num_joint, max_body))
        for i, s in enumerate(new_name):
            print_toolbar(i * 1.0 / len(new_label),
                        '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                            i + 1, len(new_name), benchmark, part))
            data = read_xyz(
                os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            fp[i, :, 0:data.shape[1], :, :] = data
        end_toolbar()

    elif part == 'val':

        with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
            pickle.dump((sample_name, list(sample_label)), f)
        # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)
        fp = open_memmap(
            '{}/{}_data.npy'.format(out_path, part),
            dtype='float32',
            mode='w+',
            shape=(len(sample_label), 3, max_frame, num_joint, max_body))
        for i, s in enumerate(sample_name):
            print_toolbar(i * 1.0 / len(sample_label),
                        '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                            i + 1, len(sample_name), benchmark, part))
            data = read_xyz(
                os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            fp[i, :, 0:data.shape[1], :, :] = data
        end_toolbar()
        print("val")
    else:
        raise ValueError()

def gendata(data_path,
            txt_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'
            ):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    
    sample_name = []
    sample_label = []

    class_name = []
    
    for class_i in range(cls_num):
        class_name.append([])
   
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)
            
            class_name[action_class - 1].append(filename)

    save_im_data(sample_name, sample_label, class_name, data_path, txt_path, out_path, benchmark, part)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='data/NTU-RGB-D/nturgb+d_skeletons')
    parser.add_argument(
        '--ignored_sample_path',
        default='resource/NTU-RGB-D/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/data/lhd/test_new_STGCN_baseline/22_07_exp_100/')

    benchmark = ['xsub']
    # benchmark = ['xview']
    
    part = ['train', 'val']
    arg = parser.parse_args()

    data_path = "/data/lhd/nturgbd_raw/nturgb+d_skeletons"
    txt_path = "./NTU60_LT/NTU60_xsub_exp_100.txt"

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                data_path,
                txt_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p
                )

