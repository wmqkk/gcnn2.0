import tensorflow as tf
import numpy as np
from layers import *


class gcnnmodel(tf.Module):
    def __init__(self):
        super(gcnnmodel, self).__init__()
        filters = 64  # Number of convolution kernels
        self.layer = []
        self.layer.append(GraphConvolution(shape=(4, filters), dropout=0, name='layer1'))
        self.layer.append(GraphConvolution(shape=(filters, filters), dropout=0, name='layer2'))
        self.layer.append(GraphConvolution(shape=(filters, filters), dropout=0, name='layer3'))
        self.layer.append(Dense(shape=(filters, 3), dropout=0, name='layer4'))

    def __call__(self, features, L):
        x = features
        n = len(L)
        x = tf.cast(x, dtype=tf.float64)
        for i in range(len(self.layer)-1):
            x = self.layer[i](x, L, n)
        x = self.layer[-1](x, n)
        x = tf.convert_to_tensor(x)
        return x


class gcnmodel(tf.Module):
    def __init__(self):
        super(gcnmodel, self).__init__()
        filters = 32  # Number of convolution kernels
        self.layer = []
        self.layer.append(GraphConvolution(shape=(4, filters), dropout=0, name='layer1'))
        self.layer.append(GraphConvolution(shape=(filters, filters), dropout=0., name='layer2'))
        self.layer.append(GraphConvolution(shape=(filters, 2), dropout=0., act=lambda x:x, name='layer3'))

    def __call__(self, features, L):
        x = features
        n = 1
        x = tf.cast(x, dtype=tf.float64)
        for i in range(len(self.layer)):
            x = self.layer[i](x, L, n)
        return tf.nn.softmax(x)




