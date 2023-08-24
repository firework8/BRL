# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from torch.distributions import normal

import sys

from ..builder import LOSSES
from .base import BaseWeightedLoss


def focal_loss(input_values, gamma=0.):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

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

def get_k400_num_class_list(split):
     
    num_list_path = './data/k400_num.txt'
    with open(num_list_path, 'r') as f:
        num_list = [
            line.strip() for line in f.readlines()
        ]
    for idx in range(len(num_list)):
        num_list[idx] = int(num_list[idx])
    
    return num_list

def get_weight_list(num_class_list):
    
    num_max = max(num_class_list)
    num_min = min(num_class_list)
    upsilon = 0.99
    lamda = 0.0099
    weight = []
    
    for idx in range(len(num_class_list)):  
        the_weight = (num_class_list[idx] - num_min)/(num_max - num_min) * lamda + upsilon
        weight.append(the_weight)
    
    return weight

def get_weight_class_list(split):
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

    skeleton_num_per_cls =[]
    for cls_idx in range(cls_num):  
        num = skeleton_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        skeleton_num_per_cls.append(int(num))
    
    the_sum = sum(skeleton_num_per_cls)

    class_prob = []
    for cls_idx in range(cls_num): 
        # n / ny
        prob = the_sum / skeleton_num_per_cls[cls_idx]
        class_prob.append(prob)
    
    return class_prob
    

@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target, epoch, split):

        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


@LOSSES.register_module()
class TwostageWeightedLoss(nn.Module):
    def __init__(self, weight=None):
        super(TwostageWeightedLoss, self).__init__()

        scheduler = "drw"

        self.change = 0
        self.epoch = -1
        self.step_epoch = 20

        if scheduler == "drw":
            self.betas = [0, 1]
        elif scheduler == "default":
            self.betas = [1, 1]
        self.weight = weight

    def update_weight(self, beta):
        if beta == 0:
            for i in range(len(self.prob)):
                self.prob[i] = 1
            self.weight = torch.FloatTensor(self.prob).to(self.input.device)
        if beta == 1:
            self.weight= torch.FloatTensor(self.prob).to(self.input.device)

    def reset_epoch(self, epoch):
        idx = (epoch+1) // self.step_epoch
        beta = self.betas[idx]
        self.update_weight(beta)

    def forward(self, input, target, epoch, split):

        the_epoch = epoch['epoch']
        self.input = input

        if self.change == 0:
            self.prob = get_weight_class_list(split)
            self.change = 1

        if self.epoch != the_epoch:
            self.epoch = the_epoch
            self.reset_epoch(the_epoch)
            
        return F.cross_entropy(input, target, weight= self.weight)


@LOSSES.register_module()
class TwostageFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(TwostageFocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        scheduler = "drw"

        self.epoch = -1
        self.step_epoch = 20
        if scheduler == "drw":
            self.betas = [0, 1]
        elif scheduler == "default":
            self.betas = [1, 1]

    def forward(self, input, target, epoch, split):

        the_epoch = epoch['epoch']

        idx = (the_epoch+1) // self.step_epoch
        beta = self.betas[idx]

        if beta == 0:
            return F.cross_entropy(input, target)
        if beta == 1:
            return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


@LOSSES.register_module()
class TwostageCBLoss(nn.Module):
    def __init__(self, weight=None):
        super(TwostageCBLoss, self).__init__()

        scheduler = "drw"

        self.change = 0
        self.epoch = -1
        self.step_epoch = 20

        if scheduler == "drw":
            self.betas = [0, 0.999]
        elif scheduler == "default":
            self.betas = [0.999, 0.999]
        self.weight = weight

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.input.device)

    def reset_epoch(self, epoch):
        idx = (epoch+1) // self.step_epoch
        beta = self.betas[idx]
        self.update_weight(beta)

    def forward(self, input, target, epoch, split):

        the_epoch = epoch['epoch']
        self.input = input

        if self.change == 0:
            self.num_class_list = get_num_class_list(split)
            self.change = 1

        if self.epoch != the_epoch:
            self.reset_epoch(the_epoch)
            self.epoch = the_epoch

        return F.cross_entropy(input, target, weight= self.weight)


@LOSSES.register_module()
class TwostageAALoss(nn.Module):
    def __init__(self, weight=None):
        super(TwostageAALoss, self).__init__()

        scheduler = "drw"

        self.change = 0
        self.epoch = -1
        self.step_epoch = 20

        if scheduler == "drw":
            self.betas = [0, 1]
        elif scheduler == "default":
            self.betas = [1, 1]
        self.weight = weight

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        for idx in range(len(beta)):
            beta[idx] = 1.0 - beta[idx]
        per_cls_weights = beta / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.input.device)

    def reset_epoch(self, epoch):
        idx = (epoch+1) // self.step_epoch 
        beta = self.betas[idx]
        if beta > 0:
            self.update_weight(self.weight_class_list)
        else:
            beta = [0]
            self.update_weight(beta)

    def forward(self, input, target, epoch, split):

        the_epoch = epoch['epoch']
        self.input = input

        if self.change == 0:
            self.num_class_list = get_num_class_list(split)
            self.weight_class_list = get_weight_list(self.num_class_list)
            self.change = 1

        if self.epoch != the_epoch:
            self.reset_epoch(the_epoch)
            self.epoch = the_epoch
        
        return F.cross_entropy(input, target, weight= self.weight)


@LOSSES.register_module()
class KAALoss(nn.Module):
    def __init__(self, weight=None):
        super(KAALoss, self).__init__()

        scheduler = "drw"

        self.change = 0
        self.epoch = -1
        self.step_epoch = 20

        if scheduler == "drw":
            self.betas = [0, 1]
        elif scheduler == "default":
            self.betas = [1, 1]
        self.weight = weight

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        for idx in range(len(beta)):
            beta[idx] = 1.0 - beta[idx]
        per_cls_weights = beta / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.input.device)

    def reset_epoch(self, epoch):
        idx = (epoch+1) // self.step_epoch 
        beta = self.betas[idx]
        if beta > 0:
            self.update_weight(self.weight_class_list)
        else:
            beta = [0]
            self.update_weight(beta)

    def forward(self, input, target, epoch, split):

        the_epoch = epoch['epoch']
        self.input = input

        if self.change == 0:
            self.num_class_list = get_k400_num_class_list(split)
            self.weight_class_list = get_weight_list(self.num_class_list)
            self.change = 1

        if self.epoch != the_epoch:
            self.reset_epoch(the_epoch)
            self.epoch = the_epoch
        return F.cross_entropy(input, target, weight= self.weight)


@LOSSES.register_module()
class TwostageLDAMLoss(nn.Module):

    def __init__(self, weight=None):
        super(TwostageLDAMLoss, self).__init__()

        scheduler = "drw"

        self.change = 0
        self.epoch = -1
        self.step_epoch = 20

        if scheduler == "drw":
            self.betas = [0, 0.999999]  # 0.9999
        elif scheduler == "default":
            self.betas = [0.999999, 0.999999]

        self.weight = weight
    
    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.input.device)

    def reset_epoch(self, epoch):
        idx = (epoch+1) // self.step_epoch
        if idx >= 1 :
            idx = 1
        beta = self.betas[idx]
        self.update_weight(beta)
    
    def forward(self, input, target, epoch, split):
        
        the_epoch = epoch['epoch']
        self.input = input

        if self.change == 0:
            self.num_class_list = get_num_class_list(split)
            self.change = 1
            
            self.s = 30
            max_m = 0.5
            m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.cuda.FloatTensor(m_list).to(self.input.device)
            self.m_list = m_list

        if self.epoch != the_epoch:
            self.reset_epoch(the_epoch)
            self.epoch = the_epoch
        
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = input - batch_m
        output = torch.where(index, x_m, input)

        return F.cross_entropy(self.s*output, target, weight=self.weight)


@LOSSES.register_module()
class TwostageBalancedSoftmax(_Loss):
    def __init__(self, weight=None):
        super(TwostageBalancedSoftmax, self).__init__()

        scheduler = "default"

        self.change = 0
        self.epoch = -1
        self.step_epoch = 20
        if scheduler == "drw":
            self.betas = [0, 1]
        elif scheduler == "default":
            self.betas = [1, 1]

        self.weight = weight

    def forward(self, input, target, epoch, split):

        if self.change == 0:
            freq = get_num_class_list(split)
            self.num_class_list = torch.tensor(freq)
            self.change = 1
        
        the_epoch = epoch['epoch']
        idx = (the_epoch+1) // self.step_epoch
        beta = self.betas[idx]

        if beta == 0:
            return F.cross_entropy(input, target)
        if beta == 1:
            sample_per_class = self.num_class_list
            spc = sample_per_class.type_as(input)
            spc = spc.unsqueeze(0).expand(input.shape[0], -1)
            input = input + spc.log()
            return F.cross_entropy(input=input, target=target, reduction='mean')
      

@LOSSES.register_module()
class TwostageEQL(nn.Module):
    def __init__(self):

        super(TwostageEQL, self).__init__()

        scheduler = "drw"

        self.change = 0
        self.epoch = -1
        self.step_epoch = 20
        if scheduler == "drw":
            self.betas = [0, 1]
        elif scheduler == "default":
            self.betas = [1, 1]
    
    def threshold_func(self):
        weight = self.inputs.new_zeros(self.n_c)
        weight[self.tail_flag] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight
    
    def beta_func(self):
        rand = torch.rand((self.n_i, self.n_c)).cuda()
        rand[rand<1-self.gamma] = 0
        rand[rand>=1-self.gamma] = 1
        return rand    

    def replace_masked_values(self, inputs, mask, replace_with=-1e7):
        one_minus_mask = 1 - mask
        values_to_add = replace_with * one_minus_mask
        return inputs * mask + values_to_add

    def forward(self, input, target, epoch, split):

        the_epoch = epoch['epoch']
        idx = (the_epoch+1) // self.step_epoch
        beta = self.betas[idx]

        if beta == 0:
            return F.cross_entropy(input, target)
        if beta == 1:

            if self.change == 0:
                num_class_list = get_num_class_list(split)
                cls_tail = int(len(num_class_list) / 3) * 2
                max_tail_num = num_class_list[cls_tail]
                self.gamma = 1.76 * 1e-3
                self.tail_flag = [False] * len(num_class_list)
                for i in range(len(self.tail_flag)):
                    if num_class_list[i] < max_tail_num:
                        self.tail_flag[i] = True
                self.change = 1

            self.inputs = input
            self.n_i, self.n_c = self.inputs.size()
            not_ignored = self.threshold_func()
            over_prob = self.beta_func()
            is_gt = target.new_zeros((self.n_i, self.n_c)).float()
            is_gt[torch.arange(self.n_i), target] = 1

            weights = ((not_ignored + over_prob + is_gt) > 0).float()
            inputs = self.replace_masked_values(input, weights)

            return F.cross_entropy(inputs, target)


@LOSSES.register_module()
class KPSLoss(nn.Module):

    def __init__(self, weight=None):
        super(KPSLoss, self).__init__()

        self.change = 0
        self.step_epoch = 16
        self.weight = weight
    
    def forward(self, input, target, epoch, split):
        
        the_epoch = epoch['epoch']
        self.input = input

        if self.change == 0:
            self.num_class_list = get_num_class_list(split)
            self.change = 1
            
            s_list = torch.cuda.FloatTensor(self.num_class_list).to(self.input.device)
            s_list = s_list*(50/s_list.min())
            s_list = torch.log(s_list)
            s_list = s_list*(1/s_list.min())
            self.s_list = s_list
            self.s = 1
            
            max_m = 0.5
            m_list =  torch.flip(self.s_list, dims=[0])
            m_list = m_list * (max_m / m_list.max())
            self.m_list = m_list
        
        cosine = input * self.s_list
        phi = cosine - self.m_list
        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi, cosine)

        if the_epoch < self.step_epoch:
            output *= self.s
        else:
            index_float = index.type(torch.cuda.FloatTensor)
            batch_s = torch.flip(self.s_list, dims=[0])*self.s
            batch_s = torch.clamp(batch_s, self.s, 50)
            batch_s = torch.matmul(batch_s[None, :], index_float.transpose(0,1)) 
            batch_s = batch_s.view((-1, 1))           
            output *= batch_s
        
        return F.cross_entropy(output, target, weight= self.weight)


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

@LOSSES.register_module()
class GumbelCE(nn.Module):
    def __init__(self, weight=None):
        super(GumbelCE, self).__init__()
        self.weight = weight
        self.step_epoch = 20

    def forward(self, input, target, epoch, split):
        
        the_epoch = epoch['epoch']
        
        if the_epoch < self.step_epoch:
            return F.cross_entropy(input, target, weight= self.weight)
        else:  
            pred=torch.clamp(input,min=-4,max=10)
            pestim= 1/(torch.exp(torch.exp(-(pred))))
            loss = F.cross_entropy(pestim, target, reduction='none')        
            loss = reduce_loss(loss, reduction='mean')
            loss=torch.clamp(loss,min=0,max=20)
            return loss

