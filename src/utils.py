# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

if __name__ == '__main__':
    import sys
    sys.path.append('../')

import thop
import numpy
import logging
import argparse

from torchsummary import summary

def summary_detail(model, input_size, device='cuda'):
    sample_inputs = torch.Tensor(*input_size).to(device)
    flops, params = thop.profile(model, inputs=(sample_inputs, ))
    print(f'Total FLOPs: {flops}')
    print(f'Total params: {params}')
    summary(model, input_size=input_size)

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

def load_args(Config):
    config = Config()
    parser = config.parser
    return parser.parse_args()
