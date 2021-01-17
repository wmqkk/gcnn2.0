import tensorflow as tf
import numpy as np


class Layer(tf.Module):
    def __init__(self, shape, name="layer", dropout=0.5, act=tf.nn.relu):
        super(Layer, self).__init__(name=name)
        self.vars = None
        self.shape = shape
        self.losses = []
        self.name = name
        self.dropout = dropout
        self.act = act


class GraphConvolution(Layer):
    def __call__(self, input_layer, support, n, sparse_inputs=False):
        if self.vars is None:
            self.vars = {}
            with self.name_scope:
                for i in range(len(support[0])):
                    self.vars['weights_' + str(i)] = init_weight(shape=self.shape, name=self.name + 'weights_' + str(i))
                self.vars['bias'] = init_bias(shape=self.shape[1], name=self.name + 'bias')
        result_out = []
        for num in range(n):
            x = input_layer[num]
            supports = list()
            if self.dropout > 0:
                x = tf.nn.dropout(x, 1 - self.dropout)
            for i in range(len(support[num])):
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=sparse_inputs)
                tem_support = dot(support[num][i], pre_sup, sparse=True)
                supports.append(tem_support)
            output = tf.add_n(supports)
            output += self.vars['bias']
            result_out.append(self.act(output))
        return result_out


class Dense(Layer):
    def __call__(self, input_layer, n, act=tf.nn.softmax, sparse_inputs=False):
        if self.vars is None:
            with self.name_scope:
                self.vars = {'weights': init_weight(shape=self.shape, name=self.name + 'weight'),
                             'bias': init_bias(shape=self.shape[1], name=self.name + 'bias')}
        result_out = []
        for num in range(n):
            x = input_layer[num]
            if self.dropout > 0:
                x = tf.nn.dropout(x, 1 - self.dropout)
            output = dot(x, self.vars['weights'], sparse=sparse_inputs)
            output = tf.reduce_mean(output, axis=0)
            output += self.vars['bias']
            result_out.append(act(output))
        return result_out


def init_weight(shape, name):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float64)
    weight = tf.Variable(initial, name=name, trainable=True)
    return weight


def init_bias(shape, name):
    initial = tf.zeros(shape, dtype=tf.float64)
    return tf.Variable(initial, name=name, trainable=True)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res
