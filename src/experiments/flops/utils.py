from __future__ import division
import numpy as np


def count_flops_dense(in_features, out_features, bias=True, activation=True):
  flops = (2 * in_features - 1) * out_features
  if bias:
    flops += out_features
  if activation:
    flops += out_features
  return flops


def count_flops_conv(height,
                     width,
                     in_channels,
                     out_channels,
                     kernel_size,
                     stride=1,
                     padding=0,
                     bias=True,
                     activation=True):
  if isinstance(kernel_size, int):
    kernel_size = [kernel_size] * 2
  n = kernel_size[0] * kernel_size[1] * in_channels
  flops_per_instance = 2 * n - 1
  out_height = (height - kernel_size[0] + 2 * padding) / stride + 1
  out_width = (width - kernel_size[1] + 2 * padding) / stride + 1
  num_instances_per_channel = out_height * out_width
  flops_per_channel = num_instances_per_channel * flops_per_instance
  total_flops = out_channels * flops_per_channel
  if bias:
    total_flops += out_channels * num_instances_per_channel
  if activation:
    total_flops += out_channels * num_instances_per_channel
  return total_flops


def count_flops_max_pool(height,
                         width,
                         channels,
                         kernel_size,
                         stride=None,
                         padding=0):
  if isinstance(kernel_size, int):
    kernel_size = [kernel_size] * 2
  stride = kernel_size if stride is None else stride
  if isinstance(stride, int):
    stride = [stride] * 2
  flops_per_instance = kernel_size[0] * kernel_size[1]
  out_height = (height - kernel_size[0] + 2 * padding) / stride[0] + 1
  out_width = (width - kernel_size[1] + 2 * padding) / stride[1] + 1
  num_instances_per_channel = out_height * out_width
  flops_per_channel = num_instances_per_channel * flops_per_instance
  total_flops = channels * flops_per_channel
  return total_flops


def count_flops_global_avg_pool(height, width, channels):
  return channels * height * width


def lenet_conv_flops(num_units):
  return count_flops_conv(28, 28, 1, num_units[0], 5) \
              + count_flops_max_pool(24, 24, num_units[0], 2) \
              + count_flops_conv(12, 12, num_units[0], num_units[1], 5) \
              + count_flops_max_pool(8, 8, num_units[1], 2) \
              + count_flops_dense(num_units[2], num_units[3]) \
              + count_flops_dense(num_units[3], 10, activation=False)


def conv_block_flops(shape, filters, dropout=True):
  # conv + relu
  flops = conv_flops(shape, filters, 3)
  # bn
  flops += filters
  # dropout
  if dropout:
    flops += filters
  return flops


def vgg_flops(num_units, n_classes):
  flops = count_flops_conv(32, 32, 3, num_units[0], 3, padding=1) \
          + count_flops_conv(32, 32, num_units[0], num_units[1], 3, padding=1) \
          + count_flops_max_pool(32, 32, num_units[1], 2)
  flops += count_flops_conv(16, 16, num_units[1], num_units[2], 3, padding=1) \
          + count_flops_conv(16, 16, num_units[2], num_units[3], 3, padding=1) \
          + count_flops_max_pool(16, 16, num_units[3], 2)
  flops += count_flops_conv(8, 8, num_units[3], num_units[4], 3, padding=1) \
          + count_flops_conv(8, 8, num_units[4], num_units[5], 3, padding=1) \
          + count_flops_conv(8, 8, num_units[5], num_units[6], 3, padding=1) \
          + count_flops_max_pool(8, 8, num_units[6], 2)
  flops += count_flops_conv(4, 4, num_units[6], num_units[7], 3, padding=1) \
          + count_flops_conv(4, 4, num_units[7], num_units[8], 3, padding=1) \
          + count_flops_conv(4, 4, num_units[8], num_units[9], 3, padding=1) \
          + count_flops_max_pool(4, 4, num_units[9], 2)
  flops += count_flops_conv(2, 2, num_units[9], num_units[10], 3, padding=1) \
          + count_flops_conv(2, 2, num_units[10], num_units[11], 3, padding=1) \
          + count_flops_conv(2, 2, num_units[11], num_units[12], 3, padding=1) \
          + count_flops_max_pool(2, 2, num_units[12], 2)
  flops += count_flops_dense(num_units[13], num_units[14])
  flops += count_flops_dense(num_units[14], n_classes)

  return flops
