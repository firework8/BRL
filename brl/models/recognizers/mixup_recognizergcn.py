import numpy as np
import torch
import sys
import random
from ..builder import RECOGNIZERS
from .base import BaseRecognizer

# body part
Ntu_Upper_Body = [1,2,3,4,5,6,7,8,9,10,11,20,21,22,23,24]
Ntu_Lower_Body = [0,12,13,14,15,16,17,18,19]

Nw_ucla_Upper_Body = [1,2,3,4,5,6,7,8,9,10,11]
Nw_ucla_Lower_Body = [0,12,13,14,15,16,17,18,19]

def get_num_class_list(split):

    the_split = split['split']
    num_classes = split['num_classes']
    cls_num = num_classes

    if the_split == 'xset_train':
        skeleton_max = 400
        imb_factor = 0.01
    elif the_split == 'train':
        skeleton_max = 100
        imb_factor = 0.1
    else:
        skeleton_max = 600
        imb_factor = 0.01
    skeleton_num_per_cls = []
    for cls_idx in range(cls_num):  
        num = skeleton_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        skeleton_num_per_cls.append(int(num))
    return skeleton_num_per_cls

def rebalance_mix_up(keypoint, label, split):

    num_per_cls = get_num_class_list(split)
    the_split = split['split']
    if the_split == 'train':
        upper_body = Nw_ucla_Upper_Body
        lower_body = Nw_ucla_Lower_Body
    else:
        upper_body = Ntu_Upper_Body
        lower_body = Ntu_Lower_Body
    
    ratio = split['ratio']
    alpha = split['alpha']
    im_k = split['im_k']

    data_list = list(range(keypoint.shape[0]))
    choose_data = random.sample(data_list, int(ratio * len(data_list)))
    new_keypoint = keypoint

    # 16 2 100 25 3
    label_a = label
    label_b = label.clone()
    mix_alpha = 0
    
    for idx in data_list:
        if idx in choose_data:
            new_list = data_list      
            while idx in new_list:
                new_list.remove(idx)
            choose_idx = random.choice(new_list)
            for frame_idx in range(keypoint.shape[2]):
                change_x = new_keypoint[idx,:,frame_idx,upper_body,:]
                change_y = new_keypoint[choose_idx,:,frame_idx,lower_body,:]
                new_x = torch.cat((change_x, change_y),dim= 1)
                new_keypoint[idx,:,frame_idx,:,:] = new_x
            label_b[idx] = label[choose_idx]
            num_x = num_per_cls[label[idx]]
            num_y = num_per_cls[label[choose_idx]]
            if num_x > im_k * num_y :
                mix_alpha = 0
            elif num_y > im_k * num_x :
                mix_alpha = 1
            else :
                mix_alpha = alpha
    
    return new_keypoint, label_a, label_b, mix_alpha

def base_mix_up(keypoint, label, split):

    the_split = split['split']
    if the_split == 'train':
        upper_body = Nw_ucla_Upper_Body
        lower_body = Nw_ucla_Lower_Body
    else:
        upper_body = Ntu_Upper_Body
        lower_body = Ntu_Lower_Body
    
    ratio = split['ratio']
    alpha = split['alpha']
    im_k = split['im_k']

    data_list = list(range(keypoint.shape[0]))
    choose_data = random.sample(data_list, int(ratio * len(data_list)))
    new_keypoint = keypoint

    # 16 2 100 25 3
    label_a = label
    label_b = label.clone()
    mix_alpha = alpha
    
    for idx in data_list:
        if idx in choose_data:
            new_list = data_list      
            while idx in new_list:
                new_list.remove(idx)
            choose_idx = random.choice(new_list)
            for frame_idx in range(keypoint.shape[2]):
                change_x = new_keypoint[idx,:,frame_idx,upper_body,:]
                change_y = new_keypoint[choose_idx,:,frame_idx,lower_body,:]
                new_x = torch.cat((change_x, change_y),dim= 1)
                new_keypoint[idx,:,frame_idx,:,:] = new_x
            label_b[idx] = label[choose_idx]
             
    return new_keypoint, label_a, label_b, mix_alpha


@RECOGNIZERS.register_module()
class MixRecognizerGCN(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition. """

    def forward_train(self, keypoint, label, epoch, split, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]       # 16 2 100 25 3
        keypoint, label_a, label_b, mix_alpha = rebalance_mix_up(keypoint, label, split)
        
        losses = dict()
        x = self.extract_feat(keypoint)
        cls_score = self.cls_head(x)

        gt_label_a = label_a.squeeze(-1)
        gt_label_b = label_b.squeeze(-1)

        loss = self.cls_head.loss(cls_score, gt_label_a, gt_label_b, mix_alpha, epoch, split)
        losses.update(loss)

        return losses

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert self.with_cls_head or self.feat_ext
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])

        x = self.extract_feat(keypoint)
        feat_ext = self.test_cfg.get('feat_ext', False)
        pool_opt = self.test_cfg.get('pool_opt', 'all')
        score_ext = self.test_cfg.get('score_ext', False)
        if feat_ext or score_ext:
            assert bs == 1
            assert isinstance(pool_opt, str)
            dim_idx = dict(n=0, m=1, t=3, v=4)

            if pool_opt == 'all':
                pool_opt == 'nmtv'
            if pool_opt != 'none':
                for digit in pool_opt:
                    assert digit in dim_idx

            if isinstance(x, tuple) or isinstance(x, list):
                x = torch.cat(x, dim=2)
            assert len(x.shape) == 5, 'The shape is N, M, C, T, V'
            if pool_opt != 'none':
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)

            if score_ext:
                w = self.cls_head.fc_cls.weight
                b = self.cls_head.fc_cls.bias
                x = torch.einsum('nmctv,oc->nmotv', x, w)
                if b is not None:
                    x = x + b[..., None, None]
                x = x[None]
            return x.data.cpu().numpy().astype(np.float16)

        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        # harmless patch
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'

        cls_score = self.average_clip(cls_score)
        if isinstance(cls_score, tuple) or isinstance(cls_score, list):
            cls_score = [x.data.cpu().numpy() for x in cls_score]
            return [[x[i] for x in cls_score] for i in range(bs)]

        return cls_score.data.cpu().numpy()

    def forward(self, keypoint, label=None, epoch=None, split=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, epoch, split, **kwargs)

        return self.forward_test(keypoint, **kwargs)

    def extract_feat(self, keypoint):
        """Extract features through a backbone.

        Args:
            keypoint (torch.Tensor): The input keypoints.

        Returns:
            torch.tensor: The extracted features.
        """
        return self.backbone(keypoint)



@RECOGNIZERS.register_module()
class Base_MixRecognizerGCN(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition. """

    def forward_train(self, keypoint, label, epoch, split, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]       # 16 2 100 25 3
        keypoint, label_a, label_b, mix_alpha = base_mix_up(keypoint, label, split)
        
        losses = dict()
        x = self.extract_feat(keypoint)
        cls_score = self.cls_head(x)

        gt_label_a = label_a.squeeze(-1)
        gt_label_b = label_b.squeeze(-1)

        loss = self.cls_head.loss(cls_score, gt_label_a, gt_label_b, mix_alpha, epoch, split)
        losses.update(loss)

        return losses

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert self.with_cls_head or self.feat_ext
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])

        x = self.extract_feat(keypoint)
        feat_ext = self.test_cfg.get('feat_ext', False)
        pool_opt = self.test_cfg.get('pool_opt', 'all')
        score_ext = self.test_cfg.get('score_ext', False)
        if feat_ext or score_ext:
            assert bs == 1
            assert isinstance(pool_opt, str)
            dim_idx = dict(n=0, m=1, t=3, v=4)

            if pool_opt == 'all':
                pool_opt == 'nmtv'
            if pool_opt != 'none':
                for digit in pool_opt:
                    assert digit in dim_idx

            if isinstance(x, tuple) or isinstance(x, list):
                x = torch.cat(x, dim=2)
            assert len(x.shape) == 5, 'The shape is N, M, C, T, V'
            if pool_opt != 'none':
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)

            if score_ext:
                w = self.cls_head.fc_cls.weight
                b = self.cls_head.fc_cls.bias
                x = torch.einsum('nmctv,oc->nmotv', x, w)
                if b is not None:
                    x = x + b[..., None, None]
                x = x[None]
            return x.data.cpu().numpy().astype(np.float16)

        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        # harmless patch
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'

        cls_score = self.average_clip(cls_score)
        if isinstance(cls_score, tuple) or isinstance(cls_score, list):
            cls_score = [x.data.cpu().numpy() for x in cls_score]
            return [[x[i] for x in cls_score] for i in range(bs)]

        return cls_score.data.cpu().numpy()

    def forward(self, keypoint, label=None, epoch=None, split=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, epoch, split, **kwargs)

        return self.forward_test(keypoint, **kwargs)

    def extract_feat(self, keypoint):
        """Extract features through a backbone.

        Args:
            keypoint (torch.Tensor): The input keypoints.

        Returns:
            torch.tensor: The extracted features.
        """
        return self.backbone(keypoint)