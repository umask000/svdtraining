# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

if __name__ == '__main__':
    import sys
    sys.path.append('../')

import torch

from torchvision import transforms, datasets

from src.utils import *

def load_cifar(root, download=False, batch_size=4, num_workers=0):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                       # pad zeros around and then crop images to 32 × 32
        transforms.RandomHorizontalFlip(),                                          # randomly flip images
        transforms.ToTensor(),                                                      # transform image to tensor
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),                                                      # transform image to tensor
    ])
    trainset = datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
    testset = datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader, testloader

if __name__ == '__main__':

    load_cifar(True)
