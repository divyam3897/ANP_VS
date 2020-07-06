from .layers import *


def ber_concrete(temp, p_logits, n_samples=None):
  shape = tf.shape(p_logits) if n_samples is None else \
          tf.concat([[n_samples], tf.shape(p_logits)], 0)
  u = tf.random_uniform(shape)
  return sigmoid((logit(u) + p_logits) / temp)


class BBDropout(object):

  def __init__(self,
               n_in,
               alpha=1e-4,
               a_uc_init=-1.0,
               thres=1e-3,
               name='bbd',
               reuse=None):
    self.n_in = n_in
    self.alpha = alpha
    self.thres = thres
    with tf.variable_scope(name, reuse=reuse):
      self.a_uc = tf.get_variable(
          'a_uc', shape=[n_in], initializer=tf.constant_initializer(a_uc_init))
      self.a = softplus(tf.maximum(-10.0, self.a_uc))
      self.b_uc = tf.get_variable('b_uc',
                                  shape=[n_in],
                                  initializer=tf.constant_initializer(0.5413))
      self.b = softplus(tf.clip_by_value(self.b_uc, -10.0, 50.0))

  def kl(self):
    Euler = 0.577215664901532
    kl = (1 - self.alpha/self.a)*(-Euler \
            - tf.digamma(self.b) - 1./self.b) \
            + log(self.a*self.b/self.alpha) - (self.b-1)/self.b
    return tf.reduce_sum(kl)

  def sample_pi(self):
    u = tf.random_uniform([self.n_in], minval=1e-10, maxval=1 - 1e-10)
    return tf.pow(1 - tf.pow(u, 1. / self.b), 1. / self.a)

  def Epi(self):
    return self.b*exp(tf.lgamma(1 + 1./self.a) \
            + tf.lgamma(self.b) \
            - tf.lgamma(1 + 1./self.a + self.b))

  def n_active(self):
    Epi = self.Epi()
    return tf.reduce_sum(tf.to_int32(tf.greater(Epi, self.thres)))

  def n_active_map(self):
    Epi = self.Epi()
    return tf.to_int32(tf.greater(Epi, self.thres))

  def mask(self, x, train):
    if train:
      pi = self.sample_pi()
      z = ber_concrete(1e-1, logit(pi), n_samples=tf.shape(x)[0])
    else:
      z = self.Epi()
      z = tf.tile(tf.expand_dims(z, 0), [tf.shape(x)[0], 1])
    return z

  def mask_ind(self):
    return tf.reshape(tf.where(tf.greater(self.Epi(), self.thres)), [-1])

  def __call__(self, x, train):
    z = self.mask(x, train)
    z = z if train else tf.where(tf.greater(z, self.thres), z, tf.zeros_like(z))
    z = tf.reshape(z, [-1, self.n_in, 1, 1]) if len(x.shape) == 4 else z
    return x * z

  def params(self, trainable=None):
    return [self.a_uc, self.b_uc]

  def mask_ops(self, layer, mask):
    return [
        tf.assign(layer.a_uc, tf.gather(self.a_uc, mask)),
        tf.assign(layer.b_uc, tf.gather(self.b_uc, mask))
    ]
