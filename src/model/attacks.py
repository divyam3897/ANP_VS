import logging
import numpy as np
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


class FGSMAttack:

  def __init__(self, x_nat, y_nat, net, mode, epsilon):
    """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
    self.net = net
    self.mode = mode
    self.epsilon = epsilon
    self.x = x_nat
    self.y = y_nat

    loss, acc = self.net.classify(self.x, self.y, mode=self.mode, train=False)
    self.grad = tf.gradients(loss, self.x)[0]
    self.count = 0

  def perturb(self, x_nat, y_nat, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm."""

    x_adv = np.copy(x_nat)
    g = sess.run(self.grad, feed_dict={self.x: x_nat, self.y: y_nat})
    x_adv = x_adv + self.epsilon * np.sign(g)

    return x_adv


class PGDAttack:

  def __init__(self,
               x_nat,
               y_nat,
               net,
               mode,
               epsilon,
               num_steps,
               step_size,
               random_start=True):
    """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
    self.net = net
    self.mode = mode
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start
    self.mode = mode
    self.x = x_nat
    self.y = y_nat
    self.loss, self.acc = self.net.classify(self.x,
                                            self.y,
                                            mode=self.mode,
                                            train=False)
    self.loss_no_reduction, self.acc_no_reduction = self.net.classify_no_reduction(
        self.x, self.y, mode=self.mode, train=False)

    self.grad = tf.gradients(self.loss, self.x)[0]
    self.count = 0
    self.sparsity = 0

  def perturb(self, x_nat, y_nat, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm."""

    if self.rand:
      x_adv = x_nat + np.random.uniform(-self.epsilon, self.epsilon,
                                        x_nat.shape)
      x_adv = np.clip(x_adv, 0.0, 1.0)  # ensure valid pixel range
    else:
      x_adv = np.copy(x_nat)
    x_adv = x_adv.astype(np.float32)

    for i in range(self.num_steps):
      g = sess.run(self.grad, feed_dict={self.x: x_adv, self.y: y_nat})

      x_adv = np.add(x_adv,
                     self.step_size * np.sign(g),
                     out=x_adv,
                     casting='unsafe')
      x_adv = np.clip(x_adv, x_nat - self.epsilon, x_nat + self.epsilon)
      x_adv = np.clip(x_adv, 0.0, 1.0)

    return x_adv
