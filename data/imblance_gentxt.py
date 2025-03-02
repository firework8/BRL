import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.ntu_read_skeleton import read_xyz

max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30
cls_num = 120

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
                benchmark,
                part,
                skeleton_type='exp',
                imb_factor=0.1, 
                rand_number=0):

    if part == 'train':
        
        """
        # NTU 60
          X-sub 600 / X-view 600
        # NTU 120
          X-sub 600 / X-set 400
        """
        skeleton_max = 600
        # skeleton_max = 400
        
        skeleton_num_per_cls = []
        if skeleton_type == 'exp':
            for cls_idx in range(cls_num):
                num = skeleton_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                skeleton_num_per_cls.append(int(num))
        elif skeleton_type == 'step':
            for cls_idx in range(cls_num // 2):
                skeleton_num_per_cls.append(int(skeleton_max))
            for cls_idx in range(cls_num // 2):
                skeleton_num_per_cls.append(int(skeleton_max * imb_factor))
        else:
            skeleton_num_per_cls.extend([int(skeleton_max)] * cls_num)
        
        new_name = []
        new_label = []
        np.random.seed(rand_number)
        classes = []
        for the_i in range(cls_num) :
            classes.append(the_i)
        # np.random.shuffle(classes)
        for the_class in classes:
            the_skeleton_num = skeleton_num_per_cls[the_class]
            idx = class_name[the_class]
            np.random.shuffle(idx)
            if len(idx) > the_skeleton_num :
                for im_k in range(the_skeleton_num):
                    new_name.append(idx[im_k])
                    new_label.append(the_class)
            else :
                for im_k in range(len(idx)):
                    new_name.append(idx[im_k])
                    new_label.append(the_class)
        
        # class_num_test = []
        # for k in range(120):
        #     print("A{}:{}".format(k+1, len(class_name[k])))
        #     class_num_test.append(len(class_name[k]))

        # print(len(sample_name) / cls_num)
        # print(max(class_num_test))
        # print(min(class_num_test))
        # print(sum(class_num_test))

        # show_im_num_per_cls(skeleton_num_per_cls, cls_num)
        # print(len(sample_name))
        # print(len(new_name))

        writename = []
        for i in new_name:
            writename.append(i)
        
        f = open('./data/NTU120_xset_exp_10.txt','w', encoding='utf-8')
        for name in writename:
            f.writelines(name)
            if name != writename[-1]:
                f.writelines('\n')
        f.close()
        

def gendata(data_path,
            benchmark,
            part
            ):
    
    with open(data_path, 'r') as f:
            ske_data = [
                line.strip() for line in f.readlines()
            ]
    
    sample_name = []
    sample_label = []

    class_name = []

    for class_i in range(cls_num):
        class_name.append([])
   
    for filename in ske_data:
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])

        sample_name.append(filename)
        sample_label.append(action_class - 1)
        
        class_name[action_class - 1].append(filename)

    save_im_data(sample_name, sample_label, class_name, data_path, benchmark, part)
    
if __name__ == '__main__':

    benchmark = ['xset']
    # benchmark = ['xsub']
    part = ['train']

    data_path = "./standard_data/ntu120/NTU120_xset_train.txt"

    for b in benchmark:
        for p in part:
            gendata(
                data_path,
                benchmark=b,
                part=p
                )

