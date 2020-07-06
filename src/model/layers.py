import tensorflow as tf
import numpy as np
import math

exp = tf.exp
log = lambda x: tf.log(x + 1e-20)
logit = lambda x: log(x) - log(1 - x)
softplus = tf.nn.softplus
softmax = tf.nn.softmax
tanh = tf.nn.tanh
relu = tf.nn.relu
sigmoid = tf.nn.sigmoid
flatten = tf.layers.flatten


class Dense(object):

  def __init__(self, n_in, n_out, name='dense', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
      self.W = tf.get_variable('W', shape=[n_in, n_out])
      self.b = tf.get_variable('b', shape=[n_out])

  def __call__(self, x, activation=None, in_mask=None, out_mask=None):
    W = self.W if in_mask is None else \
            tf.gather(self.W, in_mask, axis=0)
    W = W if out_mask is None else \
            tf.gather(W, out_mask, axis=1)
    b = self.b if out_mask is None else tf.gather(self.b, out_mask)
    x = tf.matmul(x, W) + b
    x = x if activation is None else activation(x)
    return x

  def params(self, trainable=None):
    return [self.W, self.b]


class DenseBayes(object):

  def __init__(self,
               sigma_prior,
               n_in,
               n_out,
               init_rho=None,
               name='dense_bayes',
               reuse=None):
    self.sigma_prior = sigma_prior

    limit = 1.0 / math.sqrt(n_in)
    with tf.variable_scope(name, reuse=reuse):
      self.mu_w = tf.get_variable('mu_w',
                                  shape=[n_in, n_out],
                                  initializer=tf.initializers.random_uniform(
                                      -limit, limit),
                                  dtype=tf.float32)
      self.mu_b = tf.get_variable('mu_b',
                                  shape=[n_out],
                                  initializer=tf.initializers.random_uniform(
                                      -limit, limit),
                                  dtype=tf.float32)
      self.epsilon_w = tf.random.normal(shape=[n_in, n_out], mean=0., stddev=1.)
      self.epsilon_b = tf.random.normal(shape=[n_out], mean=0., stddev=1.)
      if init_rho is None:
        self.rho_w = tf.get_variable('rho_w',
                                     shape=[n_in, n_out],
                                     dtype=tf.float32)
        self.rho_b = tf.get_variable('rho_b', shape=[n_out], dtype=tf.float32)
      else:
        self.rho_w = tf.get_variable(
            'rho_w',
            shape=[n_in, n_out],
            initializer=tf.constant_initializer(value=init_rho),
            dtype=tf.float32)
        self.rho_b = tf.get_variable(
            'rho_b',
            shape=[n_out],
            initializer=tf.constant_initializer(value=init_rho),
            dtype=tf.float32)
    self.sigma_w = softplus(self.rho_w)
    self.sigma_b = softplus(self.rho_b)

  def __call__(self, x, train, activation=None, in_mask=None, out_mask=None):
    if train:
      self.W = self.mu_w + tf.multiply(self.sigma_w, self.epsilon_w)
      self.b = self.mu_b + tf.multiply(self.sigma_b, self.epsilon_b)
    else:
      self.W = self.mu_w
      self.b = self.mu_b

    W = self.W if in_mask is None else \
            tf.gather(self.W, in_mask, axis=2)
    W = W if out_mask is None else \
            tf.gather(self.b, out_mask, axis=3)
    b = self.b if out_mask is None else \
            tf.gather(self.b, out_mask)
    x = tf.matmul(x, W) + b
    x = x if activation is None else activation(x)
    return x

  def kl(self):
    kl_W = log(tf.divide(self.sigma_prior, self.sigma_w)) + (tf.square(self.sigma_w) + \
            tf.square(self.mu_w)) / (2 * tf.square(self.sigma_prior)) - 0.5
    kl_b = log(tf.divide(
        self.sigma_prior, self.sigma_b)) + (tf.square(self.sigma_b) + tf.square(
            self.mu_b)) / (2 * tf.square(self.sigma_prior)) - 0.5
    kl = tf.reduce_sum(kl_W) + tf.reduce_sum(kl_b)
    return kl

  def params(self, trainable=None):
    return [self.mu_w, self.rho_w, self.mu_b, self.rho_b]


class Conv(object):

  def __init__(self,
               n_in,
               n_out,
               kernel_size,
               strides=1,
               padding='VALID',
               name='conv',
               reuse=None):
    with tf.variable_scope(name, reuse=reuse):
      self.W = tf.get_variable('W',
                               shape=[kernel_size, kernel_size, n_in, n_out])
      self.b = tf.get_variable('b', shape=[n_out])
    self.strides = 1
    self.padding = padding

  def __call__(self, x, activation=None, in_mask=None, out_mask=None):
    W = self.W if in_mask is None else \
            tf.gather(self.W, in_mask, axis=2)
    W = W if out_mask is None else \
            tf.gather(self.b, out_mask, axis=3)
    b = self.b if out_mask is None else \
            tf.gather(self.b, out_mask)
    x = tf.nn.conv2d(x,
                     W,
                     strides=[1, 1, self.strides, self.strides],
                     padding=self.padding,
                     data_format='NCHW')
    x = tf.nn.bias_add(x, b, data_format='NCHW')
    x = x if activation is None else activation(x)
    return x

  def params(self, trainable=None):
    return [self.W, self.b]


class ConvBayes(object):

  def __init__(self,
               sigma_prior,
               n_in,
               n_out,
               kernel_size,
               init_rho=None,
               strides=1,
               padding='VALID',
               name='conv_bayes',
               reuse=None):
    self.strides = 1
    self.padding = padding
    self.sigma_prior = sigma_prior
    self.kernel_size = kernel_size

    limit = 1.0 / math.sqrt(n_in * kernel_size**2)
    with tf.variable_scope(name, reuse=reuse):
      self.mu_w = tf.get_variable('mu_w',
                                  shape=[kernel_size, kernel_size, n_in, n_out],
                                  initializer=tf.initializers.random_uniform(
                                      -limit, limit),
                                  dtype=tf.float32)
      self.mu_b = tf.get_variable('mu_b',
                                  shape=[n_out],
                                  initializer=tf.initializers.random_uniform(
                                      -limit, limit),
                                  dtype=tf.float32)
      self.epsilon_w = tf.random.normal(
          shape=[kernel_size, kernel_size, n_in, n_out], mean=0., stddev=1.)
      self.epsilon_b = tf.random.normal(shape=[n_out], mean=0., stddev=1.)
      if init_rho is None:
        self.rho_w = tf.get_variable(
            'rho_w',
            shape=[kernel_size, kernel_size, n_in, n_out],
            dtype=tf.float32)
        self.rho_b = tf.get_variable('rho_b', shape=[n_out], dtype=tf.float32)
      else:
        self.rho_w = tf.get_variable(
            'rho_w',
            shape=[kernel_size, kernel_size, n_in, n_out],
            initializer=tf.constant_initializer(value=init_rho),
            dtype=tf.float32)
        self.rho_b = tf.get_variable(
            'rho_b',
            shape=[n_out],
            initializer=tf.constant_initializer(value=init_rho),
            dtype=tf.float32)
      self.sigma_w = softplus(self.rho_w)
      self.sigma_b = softplus(self.rho_b)

  def __call__(self, x, train, activation=None, in_mask=None, out_mask=None):
    if train:
      self.W = self.mu_w + tf.multiply(self.sigma_w, self.epsilon_w)
      self.b = self.mu_b + tf.multiply(self.sigma_b, self.epsilon_b)
    else:
      self.W = self.mu_w
      self.b = self.mu_b

    W = self.W if in_mask is None else \
            tf.gather(self.W, in_mask, axis=2)
    W = W if out_mask is None else \
            tf.gather(self.b, out_mask, axis=3)
    b = self.b if out_mask is None else \
            tf.gather(self.b, out_mask)
    x = tf.nn.conv2d(x,
                     W,
                     strides=[1, 1, self.strides, self.strides],
                     padding=self.padding,
                     data_format='NCHW')
    x = tf.nn.bias_add(x, b, data_format='NCHW')
    x = x if activation is None else activation(x)
    return x

  def kl(self):
    kl_W = log(tf.divide(self.sigma_prior, self.sigma_w)) + (tf.square(self.sigma_w) + \
            tf.square(self.mu_w)) / (2 * tf.square(self.sigma_prior)) - 0.5
    kl_b = log(tf.divide(self.sigma_prior, self.sigma_b)) + (tf.square(self.sigma_b) + \
            tf.square(self.mu_b)) / (2 * tf.square(self.sigma_prior)) - 0.5
    kl = tf.reduce_sum(kl_W) + tf.reduce_sum(kl_b)
    return kl

  def params(self, trainable=None):
    return [self.mu_w, self.rho_w, self.mu_b, self.rho_b]


class BatchNorm(object):

  def __init__(self,
               n_in,
               momentum=0.99,
               beta_initializer=tf.zeros_initializer(),
               gamma_initializer=tf.ones_initializer(),
               name='batch_norm',
               reuse=None):
    self.momentum = momentum
    with tf.variable_scope(name, reuse=reuse):
      self.moving_mean = tf.get_variable('moving_mean', [n_in],
                                         initializer=tf.zeros_initializer(),
                                         trainable=False)
      self.moving_var = tf.get_variable('moving_var', [n_in],
                                        initializer=tf.ones_initializer(),
                                        trainable=False)
      self.beta = tf.get_variable('beta', [n_in], initializer=beta_initializer)
      self.gamma = tf.get_variable('gamma', [n_in],
                                   initializer=gamma_initializer)

  def __call__(self, x, train, mask=None):
    beta = self.beta if mask is None else tf.gather(self.beta, mask)
    gamma = self.gamma if mask is None else tf.gather(self.gamma, mask)
    moving_mean = self.moving_mean if mask is None \
            else tf.gather(self.moving_mean, mask)
    moving_var = self.moving_var if mask is None \
            else tf.gather(self.moving_var, mask)
    if train:
      if len(x.shape) == 4:
        x, batch_mean, batch_var = tf.nn.fused_batch_norm(x,
                                                          gamma,
                                                          beta,
                                                          data_format='NCHW')
      else:
        batch_mean, batch_var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma,
                                      1e-3)
      tf.add_to_collection(
          tf.GraphKeys.UPDATE_OPS,
          moving_mean.assign_sub(
              (1 - self.momentum) * (moving_mean - batch_mean)))
      tf.add_to_collection(
          tf.GraphKeys.UPDATE_OPS,
          moving_var.assign_sub((1 - self.momentum) * (moving_var - batch_var)))
    else:
      if len(x.shape) == 4:
        x, batch_mean, batch_var = tf.nn.fused_batch_norm(x,
                                                          gamma,
                                                          beta,
                                                          mean=moving_mean,
                                                          variance=moving_var,
                                                          is_training=False,
                                                          data_format='NCHW')
      else:
        x = tf.nn.batch_normalization(x, moving_mean, moving_var, beta, gamma,
                                      1e-3)
    return x

  def params(self, trainable=None):
    params = [self.beta, self.gamma]
    params = params + [self.moving_mean, self.moving_var] \
            if trainable is None else params
    return params


class BatchNormBayes(object):

  def __init__(self,
               sigma_prior,
               n_in,
               init_rho=None,
               momentum=0.99,
               beta_initializer=tf.zeros_initializer(),
               gamma_initializer=tf.ones_initializer(),
               name='batch_norm_bayes',
               reuse=None):
    self.momentum = momentum
    self.sigma_prior = sigma_prior
    with tf.variable_scope(name, reuse=reuse):
      self.moving_mean = tf.get_variable('moving_mean', [n_in],
                                         initializer=tf.zeros_initializer(),
                                         trainable=False)
      self.moving_var = tf.get_variable('moving_var', [n_in],
                                        initializer=tf.ones_initializer(),
                                        trainable=False)
      self.mu_gamma = tf.get_variable(
          'mu_gamma',
          shape=[n_in],
          initializer=tf.initializers.random_uniform(0, 1))
      self.mu_beta = tf.get_variable('mu_beta',
                                     shape=[n_in],
                                     initializer=tf.zeros_initializer())
      self.epsilon_gamma = tf.random.normal(shape=[n_in], mean=0., stddev=1.)
      self.epsilon_beta = tf.random.normal(shape=[n_in], mean=0., stddev=1.)
      if init_rho is None:
        self.rho_gamma = tf.get_variable('rho_gamma', shape=[n_in])
        self.rho_beta = tf.get_variable('rho_beta', shape=[n_in])

      else:
        self.rho_gamma = tf.get_variable(
            'rho_gamma',
            shape=[n_in],
            initializer=tf.constant_initializer(value=init_rho))
        self.rho_beta = tf.get_variable(
            'rho_beta',
            shape=[n_in],
            initializer=tf.constant_initializer(value=init_rho))

      self.sigma_gamma = softplus(self.rho_gamma)
      self.sigma_beta = softplus(self.rho_beta)

  def __call__(self, x, train, mask=None):
    if train:
      self.gamma = self.mu_gamma + tf.multiply(self.sigma_gamma,
                                               self.epsilon_gamma)
      self.beta = self.mu_beta + tf.multiply(self.sigma_beta, self.epsilon_beta)
    else:
      self.gamma = self.mu_gamma
      self.beta = self.mu_beta

    beta = self.beta if mask is None else tf.gather(self.beta, mask)
    gamma = self.gamma if mask is None else tf.gather(self.gamma, mask)
    moving_mean = self.moving_mean if mask is None \
            else tf.gather(self.moving_mean, mask)
    moving_var = self.moving_var if mask is None \
            else tf.gather(self.moving_var, mask)
    if train:
      if len(x.shape) == 4:
        x, batch_mean, batch_var = tf.nn.fused_batch_norm(x,
                                                          gamma,
                                                          beta,
                                                          data_format='NCHW')
      else:
        batch_mean, batch_var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma,
                                      1e-3)
      tf.add_to_collection(
          tf.GraphKeys.UPDATE_OPS,
          moving_mean.assign_sub(
              (1 - self.momentum) * (moving_mean - batch_mean)))
      tf.add_to_collection(
          tf.GraphKeys.UPDATE_OPS,
          moving_var.assign_sub((1 - self.momentum) * (moving_var - batch_var)))
    else:
      if len(x.shape) == 4:
        x, batch_mean, batch_var = tf.nn.fused_batch_norm(x,
                                                          gamma,
                                                          beta,
                                                          mean=moving_mean,
                                                          variance=moving_var,
                                                          is_training=False,
                                                          data_format='NCHW')
      else:
        x = tf.nn.batch_normalization(x, moving_mean, moving_var, beta, gamma,
                                      1e-3)
    return x

  def kl(self):
    kl_gamma = log(tf.divide(self.sigma_prior, self.sigma_gamma)) + (tf.square(self.sigma_gamma) + \
            tf.square(self.mu_gamma)) / (2 * tf.square(self.sigma_prior)) - 0.5
    kl_beta = log(tf.divide(self.sigma_prior, self.sigma_beta)) + (tf.square(self.sigma_beta) + \
            tf.square(self.mu_beta)) / (2 * tf.square(self.sigma_prior)) - 0.5
    kl = tf.reduce_sum(kl_gamma) + tf.reduce_sum(kl_beta)
    return kl

  def params(self, trainable=None):
    params = [self.mu_gamma, self.rho_gamma, self.mu_beta, self.rho_beta]
    params = params + [self.moving_mean, self.moving_var] \
            if trainable is None else params
    return params


def pool(x, **kwargs):
  return tf.layers.max_pooling2d(x,
                                 2,
                                 2,
                                 data_format='channels_first',
                                 **kwargs)


def global_avg_pool(x):
  return tf.reduce_mean(x, axis=[2, 3])
