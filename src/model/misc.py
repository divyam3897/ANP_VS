import tensorflow as tf


def softmax_cross_entropy(logits, labels):
  return tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)


def softmax_cross_entropy_no_reduction(logits, labels):
  return tf.losses.softmax_cross_entropy(logits=logits,
                                         onehot_labels=labels,
                                         reduction="none")


def accuracy(logits, labels):
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  return tf.reduce_mean(tf.to_float(correct))


def l2_loss(var_list=None):
  var_list = tf.trainable_variables() if var_list is None else var_list
  return tf.add_n([tf.nn.l2_loss(v) for v in var_list])


def get_staircase_lr(global_step, bdrs, vals):
  lr = tf.train.piecewise_constant(tf.to_int32(global_step), bdrs, vals)
  return lr
