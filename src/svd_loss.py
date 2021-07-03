# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn

if __name__ == '__main__':
    import sys
    sys.path.append('../')

import torch
from torch import nn
from torch.nn import functional as F

class CrossEntropyLossSVD(nn.CrossEntropyLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(CrossEntropyLossSVD, self).__init__(weight=weight,
                                                  size_average=size_average,
                                                  ignore_index=ignore_index,
                                                  reduce=reduce,
                                                  reduction=reduction)

    def forward(self,
                input,
                target,
                orthogonal_params,
                sparse_params,
                orthogonality_regularizer_weight=1.,
                sparsity_inducing_regularizer_weight=1.,
                device='cuda',
                mode='lh'):
        cross_entropy_loss = F.cross_entropy(input,
                                             target,
                                             weight=self.weight,
                                             ignore_index=self.ignore_index,
                                             reduction=self.reduction)

        def _orthogonality_regularizer(x):
            r = x.shape[1]
            return torch.norm(torch.mm(x.t(), x) - torch.eye(r).to(device), p='fro') / r / r

        def _sparsity_inducing_regularizer(x, mode='lh'):
            if mode == 'lh':
                return torch.norm(x, 1) / torch.norm(x, 2)
            elif model == 'l1':
                return torch.norm(x, 1)
            elif model == 'l2':
                return torch.norm(x, 2)
            raise Exception(f'Unknown mode: {mode}')

        # Regularizer calculation
        regularizer = torch.zeros(1, ).to(device)
        for orthogonal_param in orthogonal_params:
            regularizer += _orthogonality_regularizer(orthogonal_param) * orthogonality_regularizer_weight
        for sparse_param in sparse_params:
            regularizer += _sparsity_inducing_regularizer(sparse_param) * sparsity_inducing_regularizer_weight
        return cross_entropy_loss + regularizer
