from .net import Net
from .layers import *
from .bbdropout import BBDropout


class LeNetConv(Net):

  def __init__(self, n_units=None, mask=None, name='lenet_conv', reuse=None):
    n_units = [20, 50, 800, 500] if n_units is None else n_units
    self.mask = mask
    super(LeNetConv, self).__init__()
    with tf.variable_scope(name, reuse=reuse):
      self.base.append(Conv(1, n_units[0], 5, name='conv1'))
      self.base.append(Conv(n_units[0], n_units[1], 5, name='conv2'))
      self.base.append(Dense(n_units[2], n_units[3], name='dense3'))
      self.base.append(Dense(n_units[3], 10, name='dense4'))
      for i in range(4):
        self.bbd.append(BBDropout(n_units[i], name='bbd' + str(i + 1)))

  def __call__(self, x, train, mode='base'):
    temp_x = []
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = relu(self.apply(self.base[0](x), train, mode, 0))
    temp_x.append(x)
    x = pool(x)
    x = relu(self.apply(self.base[1](x), train, mode, 1))
    temp_x.append(x)
    x = pool(x)
    x = flatten(x)
    x = x if self.mask is None else tf.gather(x, self.mask, axis=1)
    x = relu(self.base[2](self.apply(x, train, mode, 2)))
    temp_x.append(x)
    x = self.base[3](self.apply(x, train, mode, 3))
    return x, temp_x


class LeNetConvBayes(Net):

  def __init__(self,
               sigma_prior=1.0,
               init_rho=-3.0,
               n_units=None,
               mask=None,
               name='lenet_conv_bayes',
               reuse=None):
    n_units = [20, 50, 800, 500] if n_units is None else n_units
    self.mask = mask
    super(LeNetConvBayes, self).__init__()
    with tf.variable_scope(name, reuse=reuse):
      self.base.append(
          ConvBayes(sigma_prior,
                    1,
                    n_units[0],
                    5,
                    init_rho=init_rho,
                    name='conv1'))
      self.base.append(
          ConvBayes(sigma_prior,
                    n_units[0],
                    n_units[1],
                    5,
                    init_rho=init_rho,
                    name='conv2'))
      self.base.append(
          DenseBayes(sigma_prior,
                     n_units[2],
                     n_units[3],
                     init_rho=init_rho,
                     name='dense3'))
      self.base.append(
          DenseBayes(sigma_prior,
                     n_units[3],
                     10,
                     init_rho=init_rho,
                     name='dense4'))
      for i in range(4):
        self.bbd.append(BBDropout(n_units[i], name='bbd' + str(i + 1)))

  def __call__(self, x, train, mode='base'):
    temp_x = []
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = relu(
        self.apply(self.base[0](x, mode == 'base' and train), train, mode, 0))
    temp_x.append(x)
    x = pool(x)
    x = relu(
        self.apply(self.base[1](x, mode == 'base' and train), train, mode, 1))
    temp_x.append(x)
    x = pool(x)
    x = flatten(x)
    x = x if self.mask is None else tf.gather(x, self.mask, axis=1)
    x = relu(self.base[2](self.apply(x, train, mode, 2), mode == 'base' and
                          train))
    temp_x.append(x)
    x = relu(self.base[3](self.apply(x, train, mode, 3), mode == 'base' and
                          train))
    return x, temp_x
