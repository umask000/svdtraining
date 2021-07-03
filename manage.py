# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

from src.models import svd_resnet

if __name__ == '__main__':

    model, _ = svd_resnet.resnet18()


    # for name, p in model.named_parameters():
    #     print(name, p.shape)
    #
    # print(model.layer1[0])
    # print(model.layer1[1])
    # print(model.layer1[2])


    for name, m in model.named_modules():
        print(name, type(m))