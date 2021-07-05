# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

from src.models import svd_resnet

import torch

if __name__ == '__main__':

    # state_dict_1 = torch.load('ckpt/svd_resnet18_004.pth')
    # state_dict_2 = torch.load('ckpt/svd_resnet18_008.pth')
    # for (name_1, parameter_1), (name_2, parameter_2) in zip(state_dict_1.items(), state_dict_2.items()):
    #     print(name_1, parameter_1.shape)
    #     print(name_2, parameter_2.shape)
    #     assert name_1 == name_2
    #     if name_1.split('.')[-1] == 'singular_value_vector':
    #         print(parameter_1)
    #         print(parameter_2)
    #     print('-' * 64)

    # model, name = svd_resnet.resnet18()
    # model.load_state_dict(state_dict)
    # for name, parameter in model.named_parameters():
    #     print(name, parameter.shape)

    # model, name = svd_resnet.resnet18()
    # torch.save(model, 'resnet18.h5')
    # torch.save(model.state_dict(), 'resnet18.pth')

    model = torch.load('resnet18.h5')

    for name, parameters in model.named_parameters():
        print(name, parameters.shape)