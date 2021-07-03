# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

if __name__ == '__main__':
    import sys
    sys.path.append('../')

import math
import torch
import warnings

from torch import nn
from torch.nn import functional as F

from src.utils import get_sorted_index

class Conv2dSVD(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 decomposition_mode=None):
        super(Conv2dSVD, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding_mode=padding_mode)
        del self.weight

        # Decomposition strategy
        kernel_height, kernel_width = self.kernel_size

        if decomposition_mode is None:                                                                                  # automatically choose decomposition strategy
            channel_wise_gap = abs(out_channels - in_channels * kernel_height * kernel_width)
            spatio_wise_gap = abs(out_channels * kernel_width - in_channels * kernel_height)
            self.decomposition_mode = 'channel' if channel_wise_gap < spatio_wise_gap else 'spatial'
        else:
            self.decomposition_mode = decomposition_mode

        if self.decomposition_mode == 'channel':                                                                        # channel-wise decomposition
            rank = min(out_channels, in_channels * kernel_height * kernel_width)                                            # r = min(n, cwh)
            self.left_singular_matrix = nn.Parameter(torch.Tensor(out_channels, rank))                                # left singular matrix with shape n × r
            self.right_singular_matrix = nn.Parameter(torch.Tensor(in_channels * kernel_width * kernel_height, rank)) # right singular matrix with shape cwh × r
            self.singular_value_vector = nn.Parameter(torch.Tensor(rank, ))                                           # singlar value vector with shape r × 1
        elif self.decomposition_mode == 'spatial':                                                                      # spatio-wise decomposition
            rank = min(out_channels * kernel_width, in_channels * kernel_height)                                            # r = min(nw, ch)
            self.left_singular_matrix = nn.Parameter(torch.Tensor(out_channels * kernel_width, rank))                 # left singular matrix with shape nw × r
            self.right_singular_matrix = nn.Parameter(torch.Tensor(in_channels * kernel_height, rank))                # right singular matrix with shape ch × r
            self.singular_value_vector = nn.Parameter(torch.Tensor(rank, ))                                           # singlar value vector with shape r × 1
        else:
            raise Exception(f'Unknown decomposition mode: {decomposition_mode}')

        # Parameter initialization
        nn.init.kaiming_uniform_(self.left_singular_matrix, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.right_singular_matrix, a=math.sqrt(5))
        nn.init.uniform_(self.singular_value_vector, -1, 1)
        # nn.init.kaiming_normal_(m.left_singular_matrix, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(m.right_singular_matrix, mode='fan_out', nonlinearity='relu')
        # nn.init.normal_(self.singualr_value_vector, 0, 1)

    def forward(self, input):
        kernel_height, kernel_width = self.kernel_size

        if self.decomposition_mode == 'channel':                                                                        # channel-wise decomposition
            weight = torch.mm(torch.mm(self.left_singular_matrix, torch.diag(self.singular_value_vector)), self.right_singular_matrix.t())  # (out_channels, in_channels * kernel_width * kernel_height)
            weight = weight.reshape(self.out_channels, self.in_channels, kernel_height, kernel_width)                                       # (out_channels, in_channels , kernel_height, kernel_width)
        elif self.decomposition_mode == 'spatial':                                                                      # spatio-wise decomposition
            weight = torch.mm(torch.mm(self.left_singular_matrix, torch.diag(self.singular_value_vector)), self.right_singular_matrix.t())  # (out_channels * kernel_width, in_channels * kernel_height)
            weight = weight.reshape(self.out_channels, kernel_width, self.in_channels, kernel_height)                                       # (out_channels, kernel_width, in_channels , kernel_height)
            weight = weight.permute((0, 2, 3, 1))                                                                                           # (out_channels, in_channels , kernel_height, kernel_width)

        # Nonzero padding mode
        if not self.padding_mode == 'zeros':
            from torch._six import container_abcs
            from itertools import repeat

            def _reverse_repeat_tuple(t, n):
                return tuple(x for x in reversed(t) for _ in range(n))

            def _ntuple(n):
                def parse(x):
                    if isinstance(x, container_abcs.Iterable):
                        return x
                    return tuple(repeat(x, n))
                return parse

            _pair = _ntuple(2)
            return F.conv2d(F.pad(input, _reverse_repeat_tuple(self.padding, 2), mode=self.padding_mode), weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LinearSVD(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearSVD, self).__init__(in_features, out_features, bias)
        del self.weight
        rank = min(in_features, out_features)
        self.left_singular_matrix = nn.Parameter(torch.Tensor(out_features, rank))
        self.right_singular_matrix = nn.Parameter(torch.Tensor(in_features, rank))
        self.singular_value_vector = nn.Parameter(torch.Tensor(rank, ))

        # Parameter initialization
        nn.init.kaiming_uniform_(self.left_singular_matrix, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.right_singular_matrix, a=math.sqrt(5))
        nn.init.uniform_(self.singular_value_vector, -1, 1)
        # nn.init.kaiming_normal_(m.left_singular_matrix, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(m.right_singular_matrix, mode='fan_out', nonlinearity='relu')
        # nn.init.normal_(self.singualr_value_vector, 0, 1)

    def forward(self, input):
        weight = torch.mm(torch.mm(self.left_singular_matrix, torch.diag(self.singular_value_vector)), self.right_singular_matrix.t())
        return F.linear(input, weight, self.bias)


