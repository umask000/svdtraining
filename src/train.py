# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

if __name__ == '__main__':
    import sys
    sys.path.append('../')

import time
import torch
import logging

from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision.models import resnet


from config import ModelConfig
from src.data import load_cifar
from src.models import svd_resnet
from src.svd_loss import CrossEntropyLossSVD
from src.utils import summary_detail, initialize_logging, load_args

def train(args):
    if __name__ == '__main__':
        ckpt_root = '../ckpt/'                                                                                          # model checkpoint path
        logging_root = '../logging/'                                                                                    # logging saving path
        data_root = '../data/'                                                                                          # dataset saving path
    else:
        ckpt_root = 'ckpt/'                                                                                             # model checkpoint path
        logging_root = 'logging/'                                                                                       # logging saving path
        data_root = 'data/'                                                                                             # dataset saving path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_name = svd_resnet.resnet18()
    model = model.to(device)

    orthogonal_params = []
    sparse_params = []
    for name, parameter in model.named_parameters():
        lastname = name.split('.')[-1]
        if lastname == 'left_singular_matrix' or lastname == 'right_singular_matrix':
            orthogonal_params.append(parameter)
        elif lastname == 'singular_value_vector':
            sparse_params.append(parameter)

    loss = CrossEntropyLossSVD()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    trainloader, testloader = load_cifar(root=data_root, download=False, batch_size=args.batch_size)
    num_batches = len(trainloader)

    initialize_logging(filename=f'{logging_root}{model_name}.log', filemode='w')

    for epoch in range(args.max_epoch):
        epoch_start_time = time.time()
        model.train()
        total_losses = 0.
        correct_count = 0
        total_count = 0
        for i, data in enumerate(trainloader, 0):
            X_train, y_train = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            y_prob = model(X_train)
            loss_value = loss(y_prob, y_train, orthogonal_params, sparse_params)
            loss_value.backward()
            optimizer.step()
            total_losses += loss_value.item()
            _, y_pred = torch.max(y_prob.data, 1)
            total_count += y_train.size(0)
            correct_count += (y_pred == y_train).sum()
            train_accuracy = 100. * correct_count / total_count
            logging.debug('[Epoch:%d, Iteration:%d] Loss: %.03f | Train accuarcy: %.3f%%' % (epoch + 1,
                                                                                             i + 1 + epoch * num_batches,
                                                                                             total_losses / (i + 1),
                                                                                             train_accuracy))
        epoch_end_time = time.time()
        logging.info('Waiting Test ...')
        model.eval()
        with torch.no_grad():
            correct_count = 0
            total_count = 0
            for data in testloader:
                X_test, y_test = data[0].to(device), data[1].to(device)
                y_prob = model(X_test)
                _, y_pred = torch.max(y_prob.data, 1)
                total_count = y_test.size(0)
                correct_count += (y_pred == y_train).sum()
            test_accuracy = 100. * correct_count / total_count
            logging.info('Saving model ...')
            torch.save(model.state_dict(), ckpt_root + model_name + '_%03d.pth' % (epoch + 1))
            logging.info('EPOCH=%03d | Accuracy= %.3f%%, Time=%.3f' % (epoch + 1, test_accuracy, epoch_end_time - epoch_start_time))


if __name__ == '__main__':
    args = load_args(ModelConfig)
    train(args)