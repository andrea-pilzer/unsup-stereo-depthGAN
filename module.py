# Code for
# Unsupervised Adversarial Depth Estimation using Cycled Generative Networks
# Andrea Pilzer, Dan Xu, Mihai Puscas, Elisa Ricci, Nicu Sebe
#
# 3DV 2018 Conference, Verona, Italy
#
# parts of the code from https://github.com/mrharicot/monodepth
#

from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)
