from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from ..model.lenet import LeNetConv, LeNetConvBayes
from ..model.attacks import FGSMAttack, PGDAttack
from ..model.vgg import VGG, VGGBayes
from ..model.misc import *
from ..utils.logger import Logger
from ..utils import mnist, cifar10, cifar100
from ..utils.paths import RESULTS_PATH
from .memory.utils import vgg_memory, lenet_memory
from .flops.utils import lenet_conv_flops, vgg_flops
import os
import re
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy import linalg as LA

np.set_printoptions(precision=4, suppress=True)

# General flags
tf.flags.DEFINE_string("eval_mode", "train", "Which evaluation mode")
tf.flags.DEFINE_integer("gpuid", 0, "Which gpu id to use")

# Training flags
tf.flags.DEFINE_string("net", "lenet_fc", "Which net to use.")
tf.flags.DEFINE_string("mode", "base", "Which mode to use.")
tf.flags.DEFINE_string("data", "cifar10", "Which dataset to use.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.flags.DEFINE_integer("save_freq", 20, "Saving frequency for model")
tf.flags.DEFINE_integer("n_epochs", 200, "No of epochs")
tf.flags.DEFINE_integer("directory", None, "Train directory")
tf.flags.DEFINE_integer("seed", None, "Random seed.")
tf.flags.DEFINE_integer("net_weights", 0, "Total weights for net")
tf.flags.DEFINE_float("kl_weight", 100.0,
                      "Weight for kl term in bayesian network")
tf.flags.DEFINE_float("beta_weight", 1.0, "Weight for kl term for pruning")
tf.flags.DEFINE_float("thres", 1e-3, "Threshold for dropout")
tf.flags.DEFINE_float("init_lr", 1e-2, "Learning rate")
tf.flags.DEFINE_boolean("adv_train", False, "Adversarial training or not")
tf.flags.DEFINE_boolean("train_source", False,
                        "Train source model for black box")

# Attack flags
tf.flags.DEFINE_string("attack", "pgd", "Which attack to use.")
tf.flags.DEFINE_string("attack_source", "base", "Source model for attack.")
tf.flags.DEFINE_string("attack_ord", "inf", "L_inf/ L_2.")
tf.flags.DEFINE_integer("pgd_steps", 40, "No of pgd steps")
tf.flags.DEFINE_float("adv_weight", 2.0, "Weight for adversarial cost")
tf.flags.DEFINE_float("eps", 0.03, "Epsilon for attack")
tf.flags.DEFINE_float("lambda_weight", 1e-4, "Vulnerability weight")
tf.flags.DEFINE_float("step_size", 0.007, "Step size for attack")
tf.flags.DEFINE_boolean("white_box", True, "White box/black box attack")
tf.flags.DEFINE_boolean("draw_loss", False, "Draw loss surface")
tf.flags.DEFINE_boolean("vulnerability", True,
                        "Use vulnerability as regularizer")
tf.flags.DEFINE_string("vulnerability_type", "expected",
                       "Expected/max vulnerability")

FLAGS = tf.app.flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpuid)


def init_random_seeds():
  tf.set_random_seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)


if FLAGS.net.startswith('lenet_conv'):
  FLAGS.data = 'mnist'
  input_fn = mnist.input_fn
  NUM_TRAIN = mnist.NUM_TRAIN
  NUM_TEST = mnist.NUM_TEST
  n_classes = 10
  FLAGS.net_weights = 1370
  orig_net = [20, 50, 800, 500]
  if FLAGS.train_source:
    net = LeNetConvBayes(name=FLAGS.net+'_source')  if FLAGS.net.endswith('bayes') \
        else LeNetConv(name=FLAGS.net+'_source')
  else:
    net = LeNetConvBayes() if FLAGS.net.endswith('bayes') \
        else LeNetConv()
  if (not FLAGS.white_box):
    source_net = LeNetConvBayes(name=FLAGS.net+'_source') if FLAGS.net.endswith('bayes') \
        else LeNetConv(name=FLAGS.net+'_source')

elif FLAGS.net.startswith('vgg'):
  vgg_mode = int(FLAGS.net.split('_')[0][-2:])
  FLAGS.net_weights = 5248
  orig_net = [
      64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512
  ]
  if FLAGS.data == 'cifar10':
    input_fn = cifar10.input_fn
    NUM_TRAIN = cifar10.NUM_TRAIN
    NUM_TEST = cifar10.NUM_TEST
    n_classes = 10
  elif FLAGS.data == 'cifar100':
    input_fn = cifar100.input_fn
    NUM_TRAIN = cifar100.NUM_TRAIN
    NUM_TEST = cifar100.NUM_TEST
    n_classes = 100
  if FLAGS.train_source:
    net = VGGBayes(n_classes, name=FLAGS.net+'_source') if FLAGS.net.endswith('bayes') \
        else VGG(n_classes, name=FLAGS.net+'_source')
  else:
    net = VGGBayes(n_classes) if FLAGS.net.endswith('bayes') \
        else VGG(n_classes)
  if (not FLAGS.white_box):
    source_net = VGGBayes(n_classes, name=FLAGS.net+'_source') if FLAGS.net.endswith('bayes') \
        else VGG(n_classes,name=FLAGS.net+'_source')
else:
  raise ValueError('Invalid net {}'.format(FLAGS.net))

if FLAGS.directory is None:
  if FLAGS.adv_train:
    if FLAGS.train_source:
      directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'adv_train',
                               'attack_source', FLAGS.mode)
      base_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data,
                                    'attack_source', 'base')
    else:
      directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'adv_train',
              'default', FLAGS.mode)
      base_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data,
                                    'default', 'base')
  elif (not FLAGS.adv_train and FLAGS.train_source):
    directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data,
                             'attack_source', FLAGS.mode)
    base_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data,
                                  'attack_source', 'base')
  elif (not FLAGS.train_source):
    directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data, 'default',
                             FLAGS.mode)
    base_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data,
                                  'default', 'base')
else:
  directory = FLAGS.directory

if not FLAGS.white_box:
  if FLAGS.adv_train:
    attack_source_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data,
                                           'adv_train', 'attack_source',
                                           FLAGS.attack_source)
  else:
    attack_source_directory = os.path.join(RESULTS_PATH, FLAGS.net, FLAGS.data,
                                           'attack_source', FLAGS.attack_source)

x, y = input_fn(True, FLAGS.batch_size)
tx, ty = input_fn(False, FLAGS.batch_size)
n_train_batches = NUM_TRAIN // FLAGS.batch_size
n_test_batches = NUM_TEST // FLAGS.batch_size

if FLAGS.eval_mode == 'attack' or FLAGS.adv_train:
  x_clean = tf.placeholder(tf.float32, shape=x.shape)
  y_clean = tf.placeholder(tf.float32, shape=y.shape)
  x_adv = tf.placeholder(tf.float32, shape=x.shape)

  if FLAGS.attack == 'fgsm':
    if (not FLAGS.white_box):
      attack = FGSMAttack(x_clean,
                          y_clean,
                          source_net,
                          FLAGS.attack_source,
                          epsilon=FLAGS.eps)
    else:
      attack = FGSMAttack(x_clean, y_clean, net, FLAGS.mode, epsilon=FLAGS.eps)
  elif FLAGS.attack == 'pgd':
    if (not FLAGS.white_box):
      attack = PGDAttack(x_clean,
                         y_clean,
                         source_net,
                         FLAGS.attack_source,
                         epsilon=FLAGS.eps,
                         num_steps=FLAGS.pgd_steps,
                         step_size=FLAGS.step_size,
                         random_start=True)
    else:
      attack = PGDAttack(x_clean,
                         y_clean,
                         net,
                         FLAGS.mode,
                         epsilon=FLAGS.eps,
                         num_steps=FLAGS.pgd_steps,
                         step_size=FLAGS.step_size,
                         random_start=True)
  else:
    raise ValueError('Invalid attack {}'.format(FLAGS.attack))


def adv_train_model():
  if not os.path.isdir(directory):
    os.makedirs(directory)
  print('results saved in {}'.format(directory))

  global_step = tf.train.get_or_create_global_step()
  sess = tf.Session()

  cent_clean, acc_clean = net.classify(x_clean, y_clean, mode=FLAGS.mode)
  cent_adv, acc_adv = net.classify(x_adv, y_clean, mode=FLAGS.mode)

  tcent_clean, tacc_clean = net.classify(x_clean,
                                         y_clean,
                                         mode=FLAGS.mode,
                                         train=False)
  tcent_adv, tacc_adv = net.classify(x_adv,
                                     y_clean,
                                     mode=FLAGS.mode,
                                     train=False)
  if FLAGS.vulnerability:
    train_vulnerability = net.vulnerability(x_adv, x_clean, mode=FLAGS.mode)
    test_vulnerability = net.vulnerability(x_adv,
                                           x_clean,
                                           mode=FLAGS.mode,
                                           train=False)

  base_vars = net.params('base')
  base_trn_vars = net.params('base', trainable=True)

  if FLAGS.mode != 'base':
    mode_vars = net.params(FLAGS.mode, trainable=True)
    kl = net.kl(mode=FLAGS.mode)
    n_active = net.n_active(mode=FLAGS.mode)

  if FLAGS.net.startswith('lenet_conv'):
    if FLAGS.mode == 'base':
      bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .7]]
      vals = [1e-3, 1e-4, 1e-5]
    else:
      bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .7]]
      vals1 = [FLAGS.init_lr * r for r in [1., 0.1, 0.01]]
      vals2 = [0.1 * v for v in vals1]
    gamma = 1e-5
  elif FLAGS.net.startswith('vgg'):
    if FLAGS.mode == 'base':
      bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .8]]
      vals = [1e-3, 1e-4, 1e-5]
    else:
      bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .8]]
      vals1 = [FLAGS.init_lr * r for r in [1., 0.1, 0.01]]
      vals2 = [0.1 * v for v in vals1]
    gamma = 1e-5

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  if FLAGS.mode == 'base':
    lr = get_staircase_lr(global_step, bdrs, vals)
    if FLAGS.vulnerability:
      loss_mix = (cent_clean + FLAGS.adv_weight * cent_adv) / (
          1 + FLAGS.adv_weight
      ) + 1e-4 * l2_loss(base_trn_vars) + FLAGS.lambda_weight * train_vulnerability
    else:
      loss_mix = (cent_clean + FLAGS.adv_weight * cent_adv) / (
          1 + FLAGS.adv_weight) + 1e-4 * l2_loss(base_trn_vars)
    if FLAGS.net.endswith('bayes'):
      loss_mix = (cent_clean + FLAGS.adv_weight * cent_adv) / (1 +
                                                               FLAGS.adv_weight)
      loss_mix += net.kl(mode=FLAGS.mode) / (NUM_TRAIN * FLAGS.kl_weight)
    with tf.control_dependencies(update_ops):
      train_op = tf.train.AdamOptimizer(lr).minimize(loss_mix,
                                                     global_step=global_step)
    saver = tf.train.Saver(base_vars)
  else:
    lr1 = get_staircase_lr(global_step, bdrs, vals1)
    lr2 = get_staircase_lr(global_step, bdrs, vals2)
    if FLAGS.vulnerability:
      adv_loss = cent_adv + FLAGS.beta_weight * kl / NUM_TRAIN + 1e-4 * l2_loss(
          base_trn_vars)
      loss_mix = (cent_clean + FLAGS.adv_weight * cent_adv) / (
          1 + FLAGS.adv_weight
      ) + FLAGS.beta_weight * kl / NUM_TRAIN + 1e-4 * l2_loss(
          base_trn_vars) + FLAGS.lambda_weight * train_vulnerability
    else:
      adv_loss = cent_adv + FLAGS.beta_weight * kl / NUM_TRAIN + 1e-4 * l2_loss(
          base_trn_vars)
      loss_mix = (cent_clean + FLAGS.adv_weight * cent_adv) / (
          1 + FLAGS.adv_weight
      ) + FLAGS.beta_weight * kl / NUM_TRAIN + 1e-4 * l2_loss(base_trn_vars)
    with tf.control_dependencies(update_ops):
      train_op1 = tf.train.AdamOptimizer(lr1).minimize(adv_loss,
                                                       var_list=mode_vars,
                                                       global_step=global_step)
      train_op2 = tf.train.AdamOptimizer(lr2).minimize(loss_mix,
                                                       var_list=base_trn_vars)
      train_op = tf.group(train_op1, train_op2)
    saver = tf.train.Saver(base_vars + mode_vars)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  train_loss_summary = tf.summary.scalar("Training clean Cross entropy",
                                         cent_clean)
  train_acc_summary = tf.summary.scalar("Training clean accuracy", acc_clean)
  test_loss_summary = tf.summary.scalar("Test clean Cross entropy", tcent_clean)
  test_acc_summary = tf.summary.scalar("Test clean accuracy", tacc_clean)

  train_adv_loss_summary = tf.summary.scalar("Training adv Cross entropy",
                                             cent_adv)
  train_adv_acc_summary = tf.summary.scalar("Training adv accuracy", acc_adv)
  test_adv_loss_summary = tf.summary.scalar("Test adv Cross entropy", tcent_adv)
  test_adv_acc_summary = tf.summary.scalar("Test adv accuracy", tacc_adv)

  if FLAGS.vulnerability:
    train_vul_summary = tf.summary.scalar("Train vulnerability",
                                          train_vulnerability)
    test_vul_summary = tf.summary.scalar("Test vulnerability",
                                         test_vulnerability)
  summary_op = tf.summary.merge_all()

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(directory, sess.graph)
  ckpt = tf.train.get_checkpoint_state(directory)
  base_ckpt = tf.train.get_checkpoint_state(base_directory)
  start_epoch = 1
  if base_ckpt and base_ckpt.model_checkpoint_path and FLAGS.mode != 'base':
    print("Base model restored from: ", base_ckpt.model_checkpoint_path)
    tf.train.Saver(base_trn_vars).restore(sess, base_ckpt.model_checkpoint_path)
  if ckpt and ckpt.model_checkpoint_path:
    print("Model restored from: ", ckpt.model_checkpoint_path)
    tf.train.Saver(base_trn_vars).restore(sess, ckpt.model_checkpoint_path)
    global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    start_epoch = global_step // n_train_batches + 1
    print(global_step, FLAGS.n_epochs)

  logfile = open(os.path.join(directory, 'train.log'), 'w')
  train_clean_logger = Logger('cent', 'acc')
  if FLAGS.vulnerability:
    train_adv_logger = Logger('cent', 'acc', 'vul')
  else:
    train_adv_logger = Logger('cent', 'acc')

  test_clean_logger = Logger('cent', 'acc')
  if FLAGS.vulnerability:
    test_adv_logger = Logger('cent', 'acc', 'vul')
    train_to_run = [
        train_op, cent_clean, acc_clean, cent_adv, acc_adv, train_vulnerability,
        summary_op
    ]
    test_to_run = [
        tcent_clean, tacc_clean, tcent_adv, tacc_adv, test_vulnerability
    ]
  else:
    test_adv_logger = Logger('cent', 'acc')
    train_to_run = [
        train_op, cent_clean, acc_clean, cent_adv, acc_adv, summary_op
    ]
    test_to_run = [tcent_clean, tacc_clean, tcent_adv, tacc_adv]

  for epoch in range(start_epoch, (FLAGS.n_epochs + 1)):
    if FLAGS.mode == 'base':
      line = 'Epoch {}, lr {:.3e}'.format(epoch, sess.run(lr))
      print(line)
      logfile.write(line + '\n')
    else:
      np_lr1, np_lr2 = sess.run([lr1, lr2])
      line = 'Epoch {}, {} lr {:.3e}, base lr {:.3e}'.format(
          epoch, FLAGS.mode, np_lr1, np_lr2)
      print(line)
      logfile.write(line + '\n')
    train_clean_logger.clear()
    train_adv_logger.clear()

    for it in tqdm(range(1, n_train_batches + 1)):
      x_np, y_np = sess.run([x, y])
      x_adv_np = attack.perturb(x_np, y_np, sess)
      x_adv_np = x_adv_np.astype(np.float32)
      if FLAGS.vulnerability:
        _, n_cent, n_acc, a_cent, a_acc, t_vul, summary = sess.run(
            train_to_run,
            feed_dict={
                x_clean: x_np,
                x_adv: x_adv_np,
                y_clean: y_np
            })
        train_adv_logger.record([a_cent, a_acc, t_vul])
      else:
        _, n_cent, n_acc, a_cent, a_acc, summary = sess.run(train_to_run,
                                                            feed_dict={
                                                                x_clean: x_np,
                                                                x_adv: x_adv_np,
                                                                y_clean: y_np
                                                            })
        train_adv_logger.record([a_cent, a_acc])
      train_clean_logger.record([n_cent, n_acc])

    writer.add_summary(summary, global_step=epoch)
    train_clean_logger.show(header='train_clean', epoch=epoch, logfile=logfile)
    train_adv_logger.show(header='train_adv', epoch=epoch, logfile=logfile)

    test_clean_logger.clear()
    test_adv_logger.clear()
    np_n_active_x = 0
    for it in range(1, n_test_batches + 1):
      tx_clean_np, ty_clean_np = sess.run([tx, ty])
      tx_adv_np = attack.perturb(tx_clean_np, ty_clean_np, sess)
      tx_adv_np = tx_adv_np.astype(np.float32)

      res = sess.run(test_to_run,
                     feed_dict={
                         x_clean: tx_clean_np,
                         y_clean: ty_clean_np,
                         x_adv: tx_adv_np
                     })
      test_clean_logger.record(res[:2])
      test_adv_logger.record(res[2:])

    test_clean_logger.show(header='test_clean', epoch=epoch, logfile=logfile)
    test_adv_logger.show(header='test_adv', epoch=epoch, logfile=logfile)
    if FLAGS.mode != 'base':
      np_kl, np_n_active = sess.run([kl, n_active])
      line = 'kl: ' + str(np_kl) + '\n'
      line += 'n_active: ' + str(np_n_active) + '\n'
      weights = 100 - ((sum(np_n_active) / FLAGS.net_weights) * 100)
      line += "Percent of Zero weights: " + str(weights) + '\n'
      print(line)
      logfile.write(line + '\n')
      print()
      logfile.write('\n')
    if epoch % FLAGS.save_freq == 0:
      saver.save(sess,
                 os.path.join(directory, 'model'),
                 global_step=global_step)
  logfile.close()
  saver.save(sess, os.path.join(directory, 'model'), global_step=global_step)


def train_model():
  if not os.path.isdir(directory):
    os.makedirs(directory)
  print('results saved in {}'.format(directory))

  cent, acc = net.classify(x, y, mode=FLAGS.mode)
  tcent, tacc = net.classify(tx, ty, mode=FLAGS.mode, train=False)

  base_vars = net.params('base')
  base_trn_vars = net.params('base', trainable=True)

  if FLAGS.mode != 'base':
    mode_vars = net.params(FLAGS.mode, trainable=True)
    kl = net.kl(mode=FLAGS.mode)
    n_active = net.n_active(mode=FLAGS.mode)

  global_step = tf.train.get_or_create_global_step()
  train_loss_summary = tf.summary.scalar("Training Cross entropy", cent)
  train_acc_summary = tf.summary.scalar("Training accuracy", acc)
  test_loss_summary = tf.summary.scalar("Test Cross entropy", tcent)
  test_acc_summary = tf.summary.scalar("Test accuracy", tacc)
  summary_op = tf.summary.merge_all()

  if FLAGS.net.startswith('lenet_conv'):
    if FLAGS.mode == 'base':
      bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .7]]
      vals = [1e-3, 1e-4, 1e-5]
    else:
      bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .7]]
      vals1 = [FLAGS.init_lr * r for r in [1., 0.1, 0.01]]
      vals2 = [0.1 * v for v in vals1]
    gamma = 1e-5
  elif FLAGS.net.startswith('vgg'):
    if FLAGS.mode == 'base':
      bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .8]]
      vals = [1e-3, 1e-4, 1e-5]
    else:
      bdrs = [int(n_train_batches * FLAGS.n_epochs * r) for r in [.5, .8]]
      vals1 = [FLAGS.init_lr * r for r in [1., 0.1, 0.01]]
      vals2 = [0.1 * v for v in vals1]
    gamma = 1e-5

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  if FLAGS.mode == 'base':
    lr = get_staircase_lr(global_step, bdrs, vals)
    l2 = l2_loss(base_trn_vars)
    loss = cent + 1e-4 * l2
    if FLAGS.net.endswith('bayes'):
      loss = cent
      kl = net.kl(FLAGS.mode) / (NUM_TRAIN * FLAGS.kl_weight)
      loss += kl
    with tf.control_dependencies(update_ops):
      train_op = tf.train.AdamOptimizer(lr).minimize(loss,
                                                     global_step=global_step)
    saver = tf.train.Saver(base_vars)
  else:
    lr1 = get_staircase_lr(global_step, bdrs, vals1)
    lr2 = get_staircase_lr(global_step, bdrs, vals2)
    loss = cent + kl / NUM_TRAIN + 1e-4 * l2_loss(base_trn_vars)
    with tf.control_dependencies(update_ops):
      train_op1 = tf.train.AdamOptimizer(lr1).minimize(loss,
                                                       var_list=mode_vars,
                                                       global_step=global_step)
      train_op2 = tf.train.AdamOptimizer(lr2).minimize(loss,
                                                       var_list=base_trn_vars)
      train_op = tf.group(train_op1, train_op2)
      saver = tf.train.Saver(base_vars + mode_vars)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(directory, sess.graph)
  ckpt = tf.train.get_checkpoint_state(directory)
  base_ckpt = tf.train.get_checkpoint_state(base_directory)
  start_epoch = 1
  if base_ckpt and base_ckpt.model_checkpoint_path:
    print("Base model restored from: ", base_ckpt.model_checkpoint_path)
    tf.train.Saver(base_trn_vars).restore(sess, base_ckpt.model_checkpoint_path)
  if ckpt and ckpt.model_checkpoint_path:
    print("Model restored from: ", ckpt.model_checkpoint_path)
    tf.train.Saver(base_trn_vars).restore(sess, ckpt.model_checkpoint_path)
    global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    start_epoch = global_step // n_train_batches + 1
    print(global_step, FLAGS.n_epochs)

  logfile = open(os.path.join(directory, 'train.log'), 'w')
  train_logger = Logger('cent', 'acc')
  train_to_run = [train_op, cent, acc]
  test_logger = Logger('cent', 'acc')
  test_to_run = [tcent, tacc]

  for epoch in range(start_epoch, FLAGS.n_epochs + 1):
    if FLAGS.mode == 'base':
      line = 'Epoch {}, lr {:.3e}'.format(epoch, sess.run(lr))
      print(line)
      logfile.write(line + '\n')
    else:
      np_lr1, np_lr2 = sess.run([lr1, lr2])
      line = 'Epoch {}, {} lr {:.3e}, base lr {:.3e}'.format(
          epoch, FLAGS.mode, np_lr1, np_lr2)
      print(line)
      logfile.write(line + '\n')
    train_logger.clear()
    for it in tqdm(range(1, n_train_batches + 1)):
      train_logger.record(sess.run(train_to_run))
      summary = sess.run(summary_op)

    writer.add_summary(summary, global_step=epoch)
    train_logger.show(header='train', epoch=epoch, logfile=logfile)

    test_logger.clear()
    np_n_active_x = 0
    for it in range(1, n_test_batches + 1):
      res = sess.run(test_to_run)
      test_logger.record(res)

    test_logger.show(header='test', epoch=epoch, logfile=logfile)
    if FLAGS.mode != 'base':
      np_kl, np_n_active = sess.run([kl, n_active])
      line = 'kl: ' + str(np_kl) + '\n'
      line += 'n_active: ' + str(np_n_active) + '\n'
      print()
      print(line)
      logfile.write(line + '\n')
      print()
      logfile.write('\n')
    if epoch % FLAGS.save_freq == 0:
      saver.save(sess,
                 os.path.join(directory, 'model'),
                 global_step=global_step)
  logfile.close()
  saver.save(sess, os.path.join(directory, 'model'), global_step=global_step)


def test_model():
  print("Eval directory", directory)
  cent, acc = net.classify(tx, ty, mode=FLAGS.mode, train=False)

  sess = tf.Session()
  ckpt = tf.train.get_checkpoint_state(directory)
  if FLAGS.mode == 'base':
    if ckpt and ckpt.model_checkpoint_path:
      tf.train.Saver(net.params(FLAGS.mode)).restore(sess,
                                                     ckpt.model_checkpoint_path)
  else:
    kl = net.kl(mode=FLAGS.mode)
    if ckpt and ckpt.model_checkpoint_path:
      tf.train.Saver(net.params('base') + net.params(FLAGS.mode)).restore(
          sess, ckpt.model_checkpoint_path)
      n_active = net.n_active(mode=FLAGS.mode)
      n_active_map = net.n_active_map(FLAGS.mode)

  logger = Logger('cent', 'acc')
  np_n_active_x = 0
  for it in range(1, n_test_batches + 1):
    logger.record(sess.run([cent, acc]))
  if FLAGS.mode != 'base':
    np_kl, np_n_active, np_n_active_map = sess.run([kl, n_active, n_active_map])
    if FLAGS.net == "lenet_conv":
      memory = lenet_memory(np_n_active)
      orig_memory = lenet_memory(orig_net)
      flops = lenet_conv_flops(np_n_active)
      orig_flops = lenet_conv_flops(orig_net)
    elif FLAGS.net == "vgg16":
      memory = vgg_memory(np_n_active, n_classes)
      orig_memory = vgg_memory(orig_net, n_classes)
      flops = vgg_flops(np_n_active, n_classes)
      orig_flops = vgg_flops(orig_net, n_classes)
    print('kl: {:.4f}'.format(np_kl))
    print('n_active: ' + ' '.join(map(str, np_n_active)))
    print('Flops {:2.2f}'.format(float(orig_flops) / float(flops)))
    print('Memory {:2.2f}'.format(float(memory) / float(orig_memory) * 100))
    print('% of zero weights {:2.2f}'.format(100 - (sum(np_n_active) /
                                                    sum(orig_net) * 100)))
  logger.show(header='test')


def attack_model():
  print("Target mode", FLAGS.mode)
  print("Source mode", FLAGS.attack_source)
  tcent_clean, tacc_clean = net.classify(x_clean,
                                         y_clean,
                                         mode=FLAGS.mode,
                                         train=False)
  tcent_adv, tacc_adv = net.classify(x_adv,
                                     y_clean,
                                     mode=FLAGS.mode,
                                     train=False)
  vulnerability = net.vulnerability(x_adv,
                                    x_clean,
                                    mode=FLAGS.mode,
                                    train=False)

  sess = tf.Session()
  ckpt = tf.train.get_checkpoint_state(directory)
  print("Target for attack: ", directory)
  print("White box attack:", FLAGS.white_box)
  if FLAGS.mode == 'base':
    if ckpt and ckpt.model_checkpoint_path:
      tf.train.Saver(net.params(FLAGS.mode)).restore(sess,
                                                     ckpt.model_checkpoint_path)
  else:
    kl = net.kl(mode=FLAGS.mode)
    if ckpt and ckpt.model_checkpoint_path:
      tf.train.Saver(net.params('base') + net.params(FLAGS.mode)).restore(
          sess, ckpt.model_checkpoint_path)
      n_active = net.n_active(mode=FLAGS.mode)

  if (not FLAGS.white_box):
    print("Source for attack: ", attack_source_directory)
    attack_source_ckpt = tf.train.get_checkpoint_state(attack_source_directory)
    if FLAGS.attack_source != 'base':
      tf.train.Saver(
          source_net.params('base') +
          source_net.params(FLAGS.attack_source)).restore(
              sess, attack_source_ckpt.model_checkpoint_path)
    else:
      tf.train.Saver(source_net.params('base')).restore(
          sess, attack_source_ckpt.model_checkpoint_path)

  adv_logger = Logger('cent', 'acc')
  clean_logger = Logger('cent', 'acc')
  test_linf_norm = 0
  test_l2_norm = 0
  test_percent_defended = 0
  for it in tqdm(range(1, n_test_batches + 1)):
    tx_clean_np, ty_clean_np = sess.run([tx, ty])
    tx_adv_np = attack.perturb(tx_clean_np, ty_clean_np, sess)
    tx_adv_np = tx_adv_np.astype(np.float32)

    delta = tx_adv_np - tx_clean_np
    linf_norm = np.amax(np.absolute(delta), axis=1)
    l2_norm = LA.norm(delta, axis=1)
    avg_l2_norm = np.sum(l2_norm) / l2_norm.shape[0]
    avg_linf_norm = np.sum(linf_norm) / linf_norm.shape[0]
    min = np.amin(np.absolute(delta), axis=1)
    percent_defended = np.sum(
        (linf_norm < FLAGS.eps).astype(np.int32)) / FLAGS.batch_size
    test_l2_norm += avg_l2_norm
    test_linf_norm += avg_linf_norm
    test_percent_defended += percent_defended
    res = sess.run(
        [tcent_clean, tacc_clean, tcent_adv, tacc_adv, vulnerability],
        feed_dict={
            x_clean: tx_clean_np,
            y_clean: ty_clean_np,
            x_adv: tx_adv_np
        })
    clean_logger.record(res[:2])
    adv_logger.record(res[2:4])

  print("Linf norm and L2 norm", test_linf_norm / n_test_batches,
        test_l2_norm / n_test_batches)
  print("Percent defended", test_percent_defended / n_test_batches)
  clean_logger.show(header='clean')
  adv_logger.show(header='adv')
  print("Vulnerability of model", res[-1])


def main(_):
  init_random_seeds()
  if (FLAGS.adv_train and FLAGS.eval_mode == 'train'):
    adv_train_model()
  elif (not FLAGS.adv_train and FLAGS.eval_mode == 'train'):
    train_model()
  elif FLAGS.eval_mode == 'test':
    test_model()
  elif FLAGS.eval_mode == 'attack':
    attack_model()
  else:
    raise ValueError('Invalid mode {}'.format(FLAGS.eval_mode))


if __name__ == '__main__':
  tf.app.run()
