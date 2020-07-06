from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


class Logger():

  def __init__(self, *args):
    self.args = args
    self.argnum = {}
    for i, arg in enumerate(args):
      self.argnum[arg] = i
    self.sums = [0] * len(args)
    self.cnt = 0
    self.clock = time.time()

  def record(self, val):
    val = [val] if type(val) is not list else val
    val = [v for v in val if v is not None]
    for i, v in enumerate(val):
      self.sums[i] += val[i]
    self.cnt += 1

  def clear(self):
    self.sums = [0] * len(self.args)
    self.cnt = 0
    self.clock = time.time()

  def get(self, arg):
    i = self.argnum.get(arg)
    if i is not None:
      return self.sums[i] / self.cnt

  def show(self, header=None, epoch=None, it=None, logfile=None):
    et = time.time() - self.clock
    line = '' if header is None else header + ': '
    if epoch is not None:
      line += 'epoch {:d}, '.format(epoch)
    if it is not None:
      line += 'iter {:d}, '.format(it)
    for arg in self.args:
      line += '{} {:f}, '.format(arg, self.sums[self.argnum[arg]] / self.cnt)
    line += '({:.3f} secs)'.format(et)
    print(line)
    if logfile is not None:
      logfile.write(line + '\n')
