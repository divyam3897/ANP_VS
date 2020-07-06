import tensorflow as tf
from .misc import softmax_cross_entropy, accuracy, softmax_cross_entropy_no_reduction

FLAGS = tf.app.flags.FLAGS


class Net(object):

  def __init__(self):
    self.base = []
    self.bbd = []

  def params(self, mode=None, trainable=None):
    params = []
    if mode is None:
      for layer in self.base:
        params += layer.params(trainable=trainable)
      for layer in self.bbd:
        params += layer.params(trainable=trainable)
    else:
      for layer in getattr(self, mode):
        params += layer.params(trainable=trainable)
    return params

  def __call__(self, x, train=True, mode='base'):
    raise NotImplementedError()

  def apply(self, x, train, mode, l, mask_list=None):
    if mode == 'base':
      return x
    elif mode == 'bbd':
      return self.bbd[l](x, train)
    else:
      raise ValueError('Invalid mode {}'.format(mode))

  def classify(self, x, y, train=True, mode='base'):
    x, temp_x = self.__call__(x, train=train, mode=mode)
    cent = softmax_cross_entropy(x, y)
    acc = accuracy(x, y)
    return cent, acc

  def vulnerability(self, x_adv, x_clean, train=True, mode='base'):
    logits_clean, temp_x_clean = self.__call__(x_clean, train=train, mode=mode)
    logits_adv, temp_x_adv = self.__call__(x_adv, train=train, mode=mode)
    vulnerability = tf.Variable(0.)
    for i in range(len(temp_x_clean)):
      if i == 0:
        vulnerability = tf.assign(
            vulnerability,
            (tf.reduce_mean(tf.abs(tf.subtract(temp_x_clean[i], temp_x_adv[i])),
                            axis=[3, 2, 0, 1])))
      else:
        if len(temp_x_clean[i].shape) > 2:
          vulnerability = tf.add(
              tf.reduce_mean(tf.abs(tf.subtract(temp_x_clean[i],
                                                temp_x_adv[i])),
                             axis=[3, 2, 0, 1]), vulnerability)
        else:
          vulnerability = tf.add(
              tf.reduce_mean(tf.abs(tf.subtract(temp_x_clean[i],
                                                temp_x_adv[i])),
                             axis=[0, 1]), vulnerability)
    return vulnerability / len(temp_x_clean)

  def classify_no_reduction(self, x, y, train=True, mode='base'):
    x, temp_x = self.__call__(x, train=train, mode=mode)
    cent = softmax_cross_entropy_no_reduction(x, y)
    acc = accuracy(x, y)
    return cent, acc

  def kl(self, mode=None):
    if mode is None:
      raise ValueError('Invalide mode {}'.format(mode))
    kl = [
        layer.kl() if hasattr(layer, 'kl') else 0.
        for layer in getattr(self, mode)
    ]
    return tf.add_n(kl)

  def reg(self, y, train=True):
    key = 'train_probit' if train else 'test_probit'
    cent = [softmax_cross_entropy(getattr(layer, key), y) \
            for layer in self.dbbd]
    cent = tf.add_n(cent) / float(len(cent))
    return cent

  def n_active(self, mode=None):
    if mode is None:
      raise ValueError('Invalid mode {}'.format(mode))
    return [layer.n_active() for layer in getattr(self, mode)]

  def n_active_map(self, mode=None):
    if mode is None:
      raise ValueError('Invalid mode {}'.format(mode))
    return [layer.n_active_map() for layer in getattr(self, mode)]
