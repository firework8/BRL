from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS

from ...core import top_k_accuracy
from ..builder import build_loss

import sys


class MixBaseHead(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, label_a, label_b, mix_alpha, epoch, split, **kwargs):

        losses = dict()
        if label_a.shape == torch.Size([]):
            label_a = label_a.unsqueeze(0)
            label_b = label_b.unsqueeze(0)
        elif label_a.dim() == 1 and label_a.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label_a = label_a.unsqueeze(0)
            label_b = label_b.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label_a.size():
            top_k_acc_1 = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label_a.detach().cpu().numpy(), (1, 5))
            top_k_acc_2 = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label_b.detach().cpu().numpy(), (1, 5))
            
            losses['top1_acc'] = torch.tensor(
                mix_alpha * top_k_acc_1[0] + (1 - mix_alpha) * top_k_acc_2[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                mix_alpha * top_k_acc_1[1] + (1 - mix_alpha) * top_k_acc_2[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        loss_cls1 = self.loss_cls(cls_score, label_a, epoch=epoch, split=split, **kwargs)
        loss_cls2 = self.loss_cls(cls_score, label_b, epoch=epoch, split=split, **kwargs)
        loss_cls = mix_alpha * loss_cls1 + (1 - mix_alpha) * loss_cls2

        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses



@HEADS.register_module()
class MixHead(MixBaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 mode='GCN',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode

        self.in_c = in_channels
        self.num_classes = num_classes

        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):      

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)
                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)
            
        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)
        cls_score = self.fc_cls(x)

        return cls_score

@HEADS.register_module()
class MixWeightedHead(MixHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='TwostageWeightedLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)

@HEADS.register_module()
class MixFocalHead(MixHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='TwostageFocalLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)
                
@HEADS.register_module()
class MixCBHead(MixHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='TwostageCBLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)

@HEADS.register_module()
class MixLDAMHead(MixHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='TwostageLDAMLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)

@HEADS.register_module()
class MixBSHead(MixHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='TwostageBalancedSoftmax'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)

@HEADS.register_module()
class MixEQLHead(MixHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='TwostageEQL'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)

@HEADS.register_module()
class MixAAHead(MixHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='TwostageAALoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)

@HEADS.register_module()
class MixKAAHead(MixHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='KAALoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)
