from __future__ import print_function
from .net import Net
from .layers import *
from .bbdropout import BBDropout
from .misc import *


class VGG(Net):

  def __init__(self, n_classes, mask=None, name='vgg', reuse=None):
    super(VGG, self).__init__()
    n_units = [
        64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512
    ]
    self.mask = mask
    self.n_classes = n_classes

    def create_block(l, n_in, n_out):
      self.base.append(
          Conv(n_in, n_out, 3, name='conv' + str(l), padding='SAME'))
      self.base.append(BatchNorm(n_out, name='bn' + str(l)))
      self.bbd.append(BBDropout(n_out, name='bbd' + str(l), a_uc_init=2.0))

    with tf.variable_scope(name, reuse=reuse):
      create_block(1, 3, n_units[0])
      for i in range(1, 13):
        create_block(i + 1, n_units[i - 1], n_units[i])

      self.bbd.append(BBDropout(n_units[13], name='bbd14'))

      self.base.append(Dense(n_units[13], n_units[14], name='dense14'))
      self.base.append(BatchNorm(n_units[14], name='bn14'))

      self.bbd.append(BBDropout(n_units[14], name='bbd15'))

      self.base.append(Dense(n_units[14], n_classes, name='dense15'))

  def __call__(self, x, train, mode='base', mask_list=[]):
    temp_x = []

    def apply_block(x, train, l, mode, p=None):
      conv = self.base[2 * l - 2]
      bn = self.base[2 * l - 1]
      x = self.apply(conv(x), train, mode, l - 1, mask_list=mask_list)
      if mode == 'sbp':
        x = relu(bn(x, False))
      else:
        x = relu(bn(x, train))
      temp_x.append(x)
      x = pool(x) if p is None else tf.layers.dropout(x, p, training=train)
      return x

    p_list = [
        0.3, None, 0.4, None, 0.4, 0.4, None, 0.4, 0.4, None, 0.4, 0.4, None
    ]
    for l, p in enumerate(p_list):
      x = apply_block(x, train, l + 1, mode, p=p)

    x = flatten(x)
    x = x if self.mask is None else tf.gather(x, self.mask, axis=1)
    x = tf.layers.dropout(x, 0.5, training=train) if mode == 'base' else x
    x = self.base[2 * 13](self.apply(x, train, mode, 13, mask_list=mask_list))
    temp_x.append(x)
    x = relu(self.base[2 * 13 + 1](x, False) if mode ==
             'sbp' else self.base[2 * 13 + 1](x, train))
    temp_x.append(x)

    x = tf.layers.dropout(x, 0.5, training=train) if mode == 'base' else x
    x = self.base[-1](self.apply(x, train, mode, 14, mask_list=mask_list))
    return x, temp_x


class VGGBayes(Net):

  def __init__(self,
               n_classes,
               sigma_prior=0.1,
               init_rho=-5.0,
               mask=None,
               name='vgg',
               reuse=None):
    super(VGGBayes, self).__init__()
    n_units = [
        64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512
    ]
    self.mask = mask
    self.n_classes = n_classes

    def create_block(l, n_in, n_out):
      self.base.append(
          ConvBayes(sigma_prior,
                    n_in,
                    n_out,
                    3,
                    init_rho=init_rho,
                    name='conv' + str(l),
                    padding='SAME'))
      self.base.append(BatchNormBayes(sigma_prior, n_out, name='bn' + str(l)))
      self.bbd.append(BBDropout(n_out, name='bbd' + str(l), a_uc_init=2.0))

    with tf.variable_scope(name, reuse=reuse):
      create_block(1, 3, n_units[0])
      for i in range(1, len(n_units) - 2):
        create_block(i + 1, n_units[i - 1], n_units[i])

      self.bbd.append(BBDropout(n_units[-2], name='bbd14'))

      self.base.append(
          DenseBayes(sigma_prior,
                     n_units[-2],
                     n_units[-1],
                     init_rho=init_rho,
                     name='dense14'))
      self.base.append(
          BatchNormBayes(sigma_prior,
                         n_units[-1],
                         init_rho=init_rho,
                         name='bn14'))

      self.bbd.append(BBDropout(n_units[-1], name='bbd15'))

      self.base.append(
          DenseBayes(sigma_prior,
                     n_units[-1],
                     n_classes,
                     init_rho=init_rho,
                     name='dense15'))

  def __call__(self, x, train, mode='base', mask_list=[]):
    temp_x = []

    def apply_block(x, train, l, mode, p=None):
      conv = self.base[2 * l - 2]
      bn = self.base[2 * l - 1]
      x = self.apply(conv(x, mode == 'base' and train),
                     train,
                     mode,
                     l - 1,
                     mask_list=mask_list)
      if mode == 'sbp':
        x = relu(bn(x, False))
      else:
        x = relu(bn(x, train))
      temp_x.append(x)
      x = pool(x) if p is None else tf.layers.dropout(x, p, training=train)
      return x

    p_list = [
        0.3, None, 0.4, None, 0.4, 0.4, None, 0.4, 0.4, None, 0.4, 0.4, None
    ]
    for l, p in enumerate(p_list):
      x = apply_block(x, train, l + 1, mode, p=p)

    x = flatten(x)
    x = x if self.mask is None else tf.gather(x, self.mask, axis=1)
    x = tf.layers.dropout(x, 0.5, training=train) if mode == 'base' else x
    x = self.base[2 * 13](self.apply(x, train, mode, 13, mask_list=mask_list),
                          mode == 'base' and train)
    temp_x.append(x)
    x = relu(self.base[2 * 13 + 1](x, False) if mode ==
             'sbp' else self.base[2 * 13 + 1](x, train))
    temp_x.append(x)

    x = tf.layers.dropout(x, 0.5, training=train) if mode == 'base' else x
    x = self.base[-1](self.apply(x, train, mode, 14, mask_list=mask_list),
                      mode == 'base' and train)
    return x, temp_x
