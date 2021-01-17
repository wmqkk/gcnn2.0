import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import input_all_data
from model import gcnnmodel

from sklearn.metrics import accuracy_score, roc_auc_score,log_loss,fbeta_score,f1_score

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

early_stopping = 100

num_supports = 1
layers = 4
num = 80
weight_decay = 5e-5
batch_size = 50
epochs = 1000

learning_rate = 0.000125

train_feature, train_y, train_support, test_feature, test_y, test_support, test_index = input_all_data(num, num_supports)
def gen(feature,supports,y,batchsize):
  count=0
  while(count<len(supports)):
    sub_support = supports[count:count+batchsize]
    for i in range(len(sub_support)):
      sub_support[i] = [tf.sparse.SparseTensor( tf.cast(item[0], dtype=tf.int64), item[1], item[2]) for item in sub_support[i]]
    yield [tf.constant(feature[count:count+batchsize]), sub_support, tf.constant(y[count:count+batchsize])]
    count+=batchsize
out_shp = []
all_cost = []

model = gcnnmodel()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(features, L, labels):
  with tf.GradientTape() as tape:
    predictions = model(features, L)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels, predictions)
    for var in model.trainable_variables:
      loss += weight_decay*tf.nn.l2_loss(var)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  labels = tf.argmax(labels, axis=1)
  train_accuracy(labels, predictions)

@tf.function
def test_step(features, L, labels):
  predictions = model(features,L)
  t_loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels, name=None)

  test_loss(t_loss)
  labels = tf.argmax(labels, axis=1)
  test_accuracy(labels, predictions)

cost = []
test_cost = []
acc = []
test_acc = []
for j in range(epochs):
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  f = gen(train_feature, train_support, train_y, batch_size)
  for train_f, train_L, train_label in f:
    train_step(train_f, train_L, train_label)
  f = gen(test_feature, test_support, test_y, batch_size)
  for test_f, test_L, test_label in f:
    test_step(test_f, test_L, test_label)
  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  cost.append(train_loss.result())
  test_cost.append(test_loss.result())
  acc.append(train_accuracy.result())
  test_acc.append(test_accuracy.result())
  print(template.format(j + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))
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
ax2.plot(acc, color =(255/255, 144.0/255, 148.0/255), label = 'train_acc')
ax2.plot(test_acc, color=(120.0/255, 180.0/255, 90.0/255), label='test_acc')
plt.yticks(fontproperties='Times New Roman', size=14,)
plt.xticks(fontproperties='Times New Roman', size=14)
plt.legend()
plt.show()
