import os
import os.path as osp
import time

import numpy as np
import torch
import torch.distributed as dist
from mmcv.engine import multi_gpu_test
from mmcv.parallel import MMDistributedDataParallel

from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, build_optimizer, get_dist_info
from mmcv.utils import TORCH_VERSION, digit_version

from mmcv import print_log

from ..core import DistEvalHook
from ..datasets import build_dataloader, build_dataset
from ..utils import cache_checkpoint, get_root_logger

import sys

import copy
import logging
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torch.nn.parallel.distributed import (DistributedDataParallel,
                                           _find_tensors)


class OwnRunner(EpochBasedRunner):
    def get_split(self, data):
        the_split = data['train']['dataset']['split']
        the_class = data['train']['num_classes']

        #  16 / 25 = 0.64    11 / 20 = 0.55
        alpha = 0.64
        ratio = 1 / 16
        im_k = 3
        
        self.split = {'split':the_split, 'num_classes':the_class, 'ratio':ratio, 'alpha':alpha, 'im_k':im_k}
        

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:

        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            epoch = {'epoch': self._epoch, 'max_epochs': self._max_epochs}
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            epoch, self.split, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

class OwnDataParallel(MMDistributedDataParallel):

    
    def train_step(self, *inputs, **kwargs):
        """train_step() API for module wrapped by DistributedDataParallel.
        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.train_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """

        # In PyTorch >= 1.7, ``reducer._rebuild_buckets()`` is moved from the
        # end of backward to the beginning of forward.
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.7')
                and self.reducer._rebuild_buckets()):
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')

        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0a0')):
            if self._check_sync_bufs_pre_fwd():
                self._sync_buffers()
        else:
            if (getattr(self, 'require_forward_param_sync', False)
                    and self.require_forward_param_sync):
                self._sync_params()

        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.train_step(*inputs[0], **kwargs[0])
                
                
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.train_step(*inputs, **kwargs)

        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0a0')):
            if self._check_sync_bufs_post_fwd():
                self._sync_buffers()

        if (torch.is_grad_enabled()
                and getattr(self, 'require_backward_grad_sync', False)
                and self.require_backward_grad_sync):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
        return output

    def val_step(self, *inputs, **kwargs):


        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.val_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.val_step(*inputs[0], **kwargs[0])