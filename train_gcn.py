import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import input_sz,preprocess_adj, chebyshev_polynomials
from model import gcnmodel
from metrics import *

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

early_stopping = 100

num_supports = 1
layers = 4
num = 80
weight_decay = 5e-4
batch_size = 50
epochs = 1000

learning_rate = 0.002

adj, data, y_train, y_test, train_mask, test_mask, labels = input_sz()
print('load_data success')

support = chebyshev_polynomials(adj,4)

data = tf.constant([data])
y_train = tf.constant(y_train)
y_test = tf.constant(y_test)
train_mask = tf.constant(train_mask)
test_mask = tf.constant(test_mask)
labels = tf.constant(labels)
supports = [[tf.sparse.SparseTensor( tf.cast(item[0], dtype=tf.int64), item[1], item[2]) for item in support]]

print('support success')
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

out_shp = []
# gpu_options = tf.GPUOptions(allow_growth=True)
all_cost = []

model = gcnmodel()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Sum(name='train_acc')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Sum(name='test_acc')

@tf.function
def train_step(features, L, labels):
  with tf.GradientTape() as tape:
    predictions = model(features, L)
    predictions = tf.squeeze(predictions)
    loss = masked_softmax_cross_entropy(predictions, labels, train_mask)
    for var in model.trainable_variables:
      loss += weight_decay*tf.nn.l2_loss(var)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  acc = masked_accuracy(predictions, labels, train_mask)
  train_accuracy(acc)

@tf.function
def test_step(features, L, labels):
  predictions = model(features,L)
  predictions = tf.squeeze(predictions)
  t_loss = masked_softmax_cross_entropy(predictions, labels, test_mask)

  test_loss(t_loss)
  acc = masked_accuracy(predictions, labels, test_mask)
  test_accuracy(acc)

cost = []
test_cost = []
acc = []
test_acc = []
for j in range(epochs):
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  train_step(data, supports, y_train)
  test_step(data, supports, y_test)
  # train_step([data], [support], y_train)
  # test_step([data], [support], y_test)
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  cost.append(train_loss.result())
  test_cost.append(test_loss.result())
  acc.append(train_accuracy.result())
  test_acc.append(test_accuracy.result())
  print(template.format(j + 1,
                        train_loss.result(),
                        100*train_accuracy.result(),
                        test_loss.result(),
                        100*test_accuracy.result()))
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(test_cost, color=(0.3, 0.6, 0.8), label='test_loss')
ax1.plot(cost, color=(1, 137/255, 89/255), label='train_loss')
plt.yticks(fontproperties='Times New Roman', size=14)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.legend()
#list1 = list(range(0,1400,200))
#ax1.set_xticks(list1)
#
ax2 = ax1.twinx()
ax2.plot(acc, color =(255/255, 144.0/255, 148.0/255), label='train_acc')
ax2.plot(test_acc, color=(120.0/255, 180.0/255, 90.0/255), label='test_acc')
plt.yticks(fontproperties='Times New Roman', size=14,)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.legend()
plt.show()
