# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

if __name__ == '__main__':
    import sys
    sys.path.append('../')

import time
import math
import thop
import json
import numpy
import torch
import pandas
import logging
import argparse

from torch import nn
from torchsummary import summary
from matplotlib import pyplot as plt

def load_args(Config):
    config = Config()
    parser = config.parser
    return parser.parse_args()

def save_args(args, save_path=None):

    class _MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, type):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    if save_path is None:
        save_path = f'../logging/config_{time.strftime("%Y%m%d%H%M%S")}.json'
    with open(save_path, 'w') as f:
        f.write(json.dumps(vars(args), cls=_MyEncoder))

def save_checkpoint(model,
                    save_path,
                    optimizer=None,
                    scheduler=None,
                    epoch=None,
                    iteration=None):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, save_path, optimizer=None, scheduler=None):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, checkpoint['epoch'], checkpoint['iteration']

def initialize_logging(filename, filemode='w'):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(filename)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=filename,
        filemode=filemode,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(filename)s | %(levelname)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def get_sorted_index(array):
    sorted_index_array = sorted(enumerate(array), key=lambda k: k[1])
    sorted_index = [x[0] for x in sorted_index_array]
    sorted_array = [x[1] for x in sorted_index_array]
    return sorted_index, sorted_array

def summary_detail(model, input_size, device='cuda'):
    sample_inputs = torch.Tensor(*input_size).to(device)
    flops, params = thop.profile(model, inputs=(sample_inputs, ))
    print(f'Total FLOPs: {flops}')
    print(f'Total params: {params}')
    summary(model, input_size=input_size[1:])

def conv2d_to_linear(conv2d, input_height, input_width):
    # The output dimension of convolution layer：O=\frac{W-K+2P}{S}+1
    def _clac_output_size(_input_size, _kernel_size, _stride_size, _padding_size=0):
        _output_size = (_input_size - _kernel_size + _padding_size * 2) / _stride_size + 1
        return int(_output_size)

    # Transform 3D index to 1D index
    def _index3d_to_index1d(_channel, _height, _width, _height_dim, _width_dim):
        return _channel * _height_dim * _width_dim + _height * _width_dim + _width

    # Transform 1D index to 3D index
    def _index1d_to_index3d(_index, _height_dim, _width_dim):
        _size = _height_dim * _width_dim
        _channel = _index // _size
        _height = (_index - _channel * _size) // _width_dim
        _width = _index - _channel * _size - _height * _width_dim
        return _channel, _height, _width

    conv_weight = conv2d.weight
    out_channels, in_channels, kernel_height, kernel_width = conv_weight.shape
    stride_height, stride_width = conv2d.stride
    padding_height, padding_width = conv2d.padding
    output_height = _clac_output_size(input_height, kernel_height, stride_height, padding_height)
    output_width = _clac_output_size(input_width, kernel_width, stride_width, padding_width)
    input_dim = in_channels * input_height * input_width
    output_dim = out_channels * output_height * output_width
    linear_weight = torch.zeros((output_dim, input_dim))
    for output_index in range(output_dim):
        _output_channel, _output_height, _output_width = _index1d_to_index3d(output_index, output_height, output_width)
        start_height = _output_height * stride_height
        start_width = _output_width * stride_width
        for _input_height in range(start_height, start_height + kernel_height):
            for _input_width in range(start_width, start_width + kernel_width):
                for in_channel in range(in_channels):
                    input_index = _index3d_to_index1d(in_channel, _input_height, _input_width, input_height, input_width)
                    linear_weight[output_index, input_index] = conv_weight[_output_channel, in_channel, _input_height - start_height, _input_width - start_width]
    return linear_weight

def svd_layer_prune(layer,
                    prune_by=2,
                    threshold=1e-6,
                    reduce_by=.98,
                    min_rank=1,
                    min_decay=.0):
    current_rank = layer.singular_value_vector.shape[0]
    min_rank = max(min_rank, math.ceil(current_rank * min_decay))                                                       # minimum rank reduced is determined by the maximum between ratio and absolute value

    # Determine the index remained after pruning
    remaining_index = []
    prune_to = current_rank
    if prune_by is not None:
        prune_to = max(current_rank - prune_by, min_rank)
    elif threshold is not None:
        for i, singular_value in enumerate(layer.singular_value_vector):
            if abs(singular_value) <= threshold:
                remaining_index.append(i)
        if len(remaining_index) < min_rank:
            warnings.warn(f'Threshold strategy leads to rank less than {min_rank}!')
            prune_to = min_rank
            remaining_index = []
    elif reduce_by is not None:
        prune_to = max(math.floor(current_rank * reduce_by), min_rank)
    else:
        raise Exception('Pruning strategy is not specified!')

    if not remaining_index:
        sorted_index, _ = get_sorted_index(torch.abs(layer.singular_value_vector))
        for i in range(current_rank):
            if sorted_index[i] < prune_to:
                remaining_index.append(i)

    # Pruning
    layer.singular_value_vector = nn.Parameter(layer.singular_value_vector[remaining_index])
    layer.left_singular_matrix = nn.Parameter(layer.left_singular_matrix[:, remaining_index])
    layer.right_singular_matrix = nn.Parameter(layer.right_singular_matrix[:, remaining_index])


def plot_logging(logging_path, export_path=None):

    if export_path is None:
        export_path = '../'

    def _parse_debug_message(x):
        x = x.replace(' ', '')
        message_1 = x[x.find('[') + 1: x.find(']')]
        epoch_message, iteration_message = message_1.split(',')
        epoch = int(epoch_message[epoch_message.find(':') + 1: ])
        iteration = int(iteration_message[iteration_message.find(':') + 1: ])
        loss = float(x[x.find('Loss:') + 5: ])
        return (epoch, iteration, loss)

    def _parse_debug_accuracy(x):
        x = x.replace(' ', '')
        accuracy = float(x[x.find(':') + 1: -1])
        return accuracy / 100.

    def _parse_info_message(x):
        x = x.replace(' ', '')
        epoch = int(x[6:])
        return epoch

    def _parse_info_accuracy(x):
        x = x.replace(' ', '')
        accuracy_message, time_message = x.split(',')
        accuracy = float(accuracy_message[accuracy_message.find('=') + 1: -1]) / 100
        time_consumed = float(time_message[time_message.find('=') + 1: ])
        return (accuracy, time_consumed)

    df = pandas.read_csv(logging_path, header=None, sep='|', dtype=str)
    df.columns = ['asctime', 'filename', 'levelname', 'message', 'accuracy']

    for column in df.columns:
        df.loc[:, column] = df[column].map(lambda x: x.strip() if x == x else x)

    df_debug = df[df['levelname'] == 'DEBUG']
    df_info = df[(df['levelname'] == 'INFO') & (df['accuracy'].notnull())]

    debug_message = df_debug['message'].apply(_parse_debug_message)
    debug_accuracy = df_debug['accuracy'].apply(_parse_debug_accuracy)
    info_message = df_info['message'].apply(_parse_info_message)
    info_accuracy = df_info['accuracy'].apply(_parse_info_accuracy)

    df_debug.loc[:, 'epoch'] = debug_message.map(lambda x: x[0])
    df_debug.loc[:, 'iteration'] = debug_message.map(lambda x: x[1])
    df_debug.loc[:, 'loss'] = debug_message.map(lambda x: x[2])
    df_debug.loc[:, 'accuracy'] = debug_accuracy
    df_info.loc[:, 'epoch'] = info_message
    df_info.loc[:, 'accuracy'] = info_accuracy.map(lambda x: x[0])
    df_info.loc[:, 'time'] = info_accuracy.map(lambda x: x[1])

    # df_debug[['asctime', 'epoch', 'iteration', 'loss', 'accuracy']].to_csv('debug.csv', sep='\t', header=True, index=False)
    # df_info[['asctime', 'epoch', 'accuracy', 'time']].to_csv('info.csv', sep='\t', header=True, index=False)

    plt.plot(df_info['epoch'], df_info['accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    plt.title('Test accuracy by epoch')
    plt.show()
    plt.close()

    plt.plot(df_info['epoch'], df_info['time'])
    plt.xlabel('epoch')
    plt.ylabel('time')
    plt.title('Epoch time')
    plt.show()
    plt.close()

    plt.plot(df_debug['iteration'], df_debug['loss'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('loss by iteration')
    plt.show()
    plt.close()

    plt.plot(df_debug['iteration'], df_debug['accuracy'])
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.title('Train accuracy by iteration')
    plt.show()
    plt.close()


if __name__ == '__main__':
    # from config import ModelConfig
    # args = load_args(ModelConfig)
    # save_args(args, )

    # plot_logging('../temp/temp3/svd_resnet18.log')
    plot_logging('../logging/svd_resnet18.log')