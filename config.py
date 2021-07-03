# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import argparse


class ModelConfig:
    parser = argparse.ArgumentParser("--")
    parser.add_argument('--max_epoch', default=128, type=int)
    parser.add_argument('--learning_rate', default=.01, type=float)
    parser.add_argument('--batch_size', default=125, type=int)

if __name__ == "__main__":
    hyperparameters = HyperParameters()
    parser = hyperparameters.parser
    hp = parser.parse_args()
    print(hp)

