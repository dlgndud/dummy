import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

batch_size = 20

classes = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120']

num_classes = len(classes)

validation_size = 0.2
img_size = 128
num_channels = 3
train_path = './training_data'

data = dataset.read_train_sets(train_path=train_path, image_size=img_size, classes=classes, validation_size=validation_size)
#print('>>>>',data.train.images)
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

#label
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y_true") #출력값은 결국 dogs, cats
y_true_cls = tf.argmax(y_true, axis=1)

keep_prob = tf.placeholder(tf.float32)
biases = tf.Variable(tf.constant(0.05, shape=[num_classes]))

W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3_flat = tf.reshape(L3, shape=[-1, 16 * 16 * 128])

W4 = tf.get_variable("W4", shape=[128 * 16 * 16, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)

W5 = tf.get_variable("W5", shape=[625, num_classes], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([num_classes]))
logits = tf.matmul(L4, W5) + b5
y_pred = tf.nn.softmax(logits, name="y_pred")

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	
    for i in range(3):
        cost_i, _ = sess.run([cost, optimizer], feed_dict={x : data.train.images, y_true : data.train.labels})
        print(i, cost_i)
    print('Training Done!')

