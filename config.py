# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import argparse

class ModelConfig:
    parser = argparse.ArgumentParser("--")
    parser.add_argument('--summary', default=True, type=bool)
    parser.add_argument('--max_epoch', default=128, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=125, type=int)
    parser.add_argument('--momentum', default=.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--svd_prune', default=True, type=bool)
    parser.add_argument('--svd_prune_cycle', default=5, type=int)
    parser.add_argument('--svd_prune_rank_each_time', default=None, type=float)
    parser.add_argument('--svd_prune_threshold', default=None, type=float)
    parser.add_argument('--svd_prune_decay', default=.95, type=float)

if __name__ == "__main__":
    hyperparameters = HyperParameters()
    parser = hyperparameters.parser
    hp = parser.parse_args()
    print(hp)

