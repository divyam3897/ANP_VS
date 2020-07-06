def count_memory_dense(in_features, out_features, bias=True, batch_norm=False):
  mem = in_features * out_features
  if bias:
    mem += out_features
  if batch_norm:
    mem += 2 * in_features
  return mem


def count_memory_conv(height,
                      width,
                      in_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      padding=0,
                      bias=True,
                      batch_norm=False):
  if isinstance(kernel_size, int):
    kernel_size = [kernel_size] * 2
  n = kernel_size[0] * kernel_size[1] * in_channels
  out_height = (height - kernel_size[0] + 2 * padding) / stride + 1
  out_width = (width - kernel_size[1] + 2 * padding) / stride + 1
  mem_fmap = n * out_height * out_width
  mem_kernel = n * out_channels
  mem = mem_fmap + mem_kernel
  if batch_norm:
    mem += 2 * out_channels
  return mem


def lenet_memory(num_units):
  return count_memory_conv(28, 28, 1, num_units[0], 5) \
          + count_memory_conv(12, 12, num_units[0], num_units[1], 5) \
          + count_memory_dense(num_units[2], num_units[3]) \
          + count_memory_dense(num_units[3], 10)


def vgg_memory(num_units, n_classes):
  mem = 0
  c = [3] + num_units
  h = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
  for i in range(13):
    mem += count_memory_conv(h[i],
                             h[i],
                             c[i],
                             c[i + 1],
                             3,
                             padding=1,
                             batch_norm=True)
    mem += count_memory_dense(num_units[13], num_units[14], batch_norm=True)
    mem += count_memory_dense(num_units[14], n_classes)
  return mem
