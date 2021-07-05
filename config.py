# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import argparse

from torch import optim

class ModelConfig:
    parser = argparse.ArgumentParser("--")
    parser.add_argument('--summary', default=True, type=bool)
    parser.add_argument('--ckpt_cycle', default=4, type=int)
    parser.add_argument('--max_epoch', default=128, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--orthogonal_regularizer_weight', default=1., type=float)
    parser.add_argument('--sparsity_regularizer_weight', default=1., type=float)
    parser.add_argument('--orthogonal_learning_rate', default=1e-3, type=float)
    parser.add_argument('--sparsity_learning_rate', default=1e-3, type=float)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--orthogonal_momentum', default=.9, type=float)
    parser.add_argument('--sparsity_momentum', default=.9, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)
    parser.add_argument('--orthogonal_weight_decay', default=0., type=float)
    parser.add_argument('--sparsity_weight_decay', default=0., type=float)
    parser.add_argument('--optimizer', default=optim.SGD, type=type)
    parser.add_argument('--svd_prune', default=True, type=bool)
    parser.add_argument('--svd_prune_cycle', default=4, type=int)
    parser.add_argument('--svd_prune_rank_each_time', default=2, type=float)
    parser.add_argument('--svd_prune_threshold', default=None, type=float)
    parser.add_argument('--svd_prune_decay', default=None, type=float)
    parser.add_argument('--svd_prune_min_rank', default=4, type=int)
    parser.add_argument('--svd_prune_min_decay', default=.1, type=int)

if __name__ == "__main__":
    hyperparameters = HyperParameters()
    parser = hyperparameters.parser
    hp = parser.parse_args()
    print(hp)

