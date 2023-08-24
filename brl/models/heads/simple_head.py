import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import numpy as np

from ..builder import HEADS
from .base import BaseHead
from sklearn.manifold import TSNE
import sys

@HEADS.register_module()
class SimpleHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
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
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """             

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
class I3DHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='3D',
                         **kwargs)

@HEADS.register_module()
class I3DAAHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='KAALoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='3D',
                         **kwargs)
    
@HEADS.register_module()
class SlowFastHead(I3DHead):
    pass


@HEADS.register_module()
class GCNHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
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
class GCNFocalHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='FocalLoss'),
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
class GCNtwostageHead(SimpleHead):

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
class GCNTwostageFocalHead(SimpleHead):

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
class GCNTwostageCBHead(SimpleHead):

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
class GCNTwostageLDAMHead(SimpleHead):

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
class GCNBSHead(SimpleHead):

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
class GCNEQLHead(SimpleHead):

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
class GCNGumbelHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='GumbelCE'),
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
class GCNKPSHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='KPSLoss'),
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
class TSNHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='2D',
                         **kwargs)

@HEADS.register_module()
class MSGCN1Head(BaseHead):
    
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 mode='MSGCN1',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        self.mode = mode

        self.in_c = in_channels
        self.num_classes = num_classes
        # two Linear
        self.fc_cls1 = nn.Linear(2 * self.in_c, self.in_c)
        self.fc_cls2 = nn.Linear(self.in_c, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls1, std=self.init_std)
        normal_init(self.fc_cls2, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        feature_a = x['a']
        feature_b = x['b']

        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = feature_a.shape
        
        feature_a = feature_a.reshape(N * M, C, T, V)   # 32*256*25*25
        feature_a = pool(feature_a)                     # 32*256*1*1
        feature_a = feature_a.reshape(N, M, C)          # 16*2*256
        feature_a = feature_a.mean(dim=1)               # 16*256

        feature_b = feature_b.reshape(N * M, C, T, V)   # 32*256*25*25
        feature_b = pool(feature_b)                     # 32*256*1*1
        feature_b = feature_b.reshape(N, M, C)          # 16*2*256
        feature_b = feature_b.mean(dim=1)               # 16*256

        l = 0.5
        mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)    # 16*512

        if self.dropout is not None:
            mixed_feature = self.dropout(mixed_feature)
        
        combined_feature = self.fc_cls1(mixed_feature)          # 16*256
        cls_score = self.fc_cls2(combined_feature)
        
        return cls_score

@HEADS.register_module()
class MSGCN2Head(BaseHead):
    
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 mode='MSGCN1',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        
        self.mode = mode

        self.in_c = in_channels
        self.num_classes = num_classes
        # two Linear
        self.fc_cls1 = nn.Linear(2 * self.in_c, self.in_c)
        self.fc_cls2 = nn.Linear(self.in_c, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls1, std=self.init_std)
        normal_init(self.fc_cls2, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if isinstance(x, dict):

            feature_a = x['a']
            feature_b = x['b']

            pool = nn.AdaptiveAvgPool2d(1)
            N, M, C, T, V = feature_a.shape
            
            feature_a = feature_a.reshape(N * M, C, T, V)   # 32*256*25*25
            feature_a = pool(feature_a)                     # 32*256*1*1
            feature_a = feature_a.reshape(N, M, C)          # 16*2*256
            feature_a = feature_a.mean(dim=1)               # 16*256

            feature_b = feature_b.reshape(N * M, C, T, V)   # 32*256*25*25
            feature_b = pool(feature_b)                     # 32*256*1*1
            feature_b = feature_b.reshape(N, M, C)          # 16*2*256
            feature_b = feature_b.mean(dim=1)               # 16*256

            l = 0.5
            mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)    # 16*512

            if self.dropout is not None:
                mixed_feature = self.dropout(mixed_feature)
            
            combined_feature = self.fc_cls1(mixed_feature)          # 16*256
            cls_score = self.fc_cls2(combined_feature)
            
            return cls_score
        else:
            pool = nn.AdaptiveAvgPool2d(1)
            N, M, C, T, V = x.shape
            
            x = x.reshape(N * M, C, T, V)   # 32*512*25*25
            x = pool(x)                     # 32*512*1*1
            x = x.reshape(N, M, C)          # 16*2*512
            x = x.mean(dim=1)               # 16*512

            if self.dropout is not None:
                x = self.dropout(x)

            combined_feature = self.fc_cls1(x)          # 16*256
            cls_score = self.fc_cls2(combined_feature)
            return cls_score

@HEADS.register_module()
class VisualHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
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
        self.mode = mode

        self.in_c = in_channels
        self.num_classes = num_classes

        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """             

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
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
        
        save_x = x.data.cpu().numpy().astype(np.float32)
        
        # X_embedded = TSNE(n_components=2, init='random', perplexity=3).fit_transform(save_x)
        
        cls_score = self.fc_cls(x)

        return cls_score, save_x



