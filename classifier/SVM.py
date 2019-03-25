import os

import tensorflow as tf
import numpy as np


def pre_data():
    dir = 'bias'


    f_max_train = np.load(os.path.join(dir, 'f_max_train.npy'))
    f_max_test = np.load(os.path.join(dir, 'f_max_test.npy'))
    f_avg_train = np.load(os.path.join(dir, 'f_avg_train.npy'))
    f_avg_test = np.load(os.path.join(dir, 'f_avg_test.npy'))

    t_max_train = np.load(os.path.join(dir, 't_max_train.npy'))
    t_max_test = np.load(os.path.join(dir, 't_max_test.npy'))
    t_avg_train = np.load(os.path.join(dir, 't_avg_train.npy'))
    t_avg_test = np.load(os.path.join(dir, 't_avg_test.npy'))


    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for val in f_max_train:
        x_train.append([val, val])
        y_train.append([1])
    for val in t_max_train:
        x_train.append([val, val])
        y_train.append([-1])
    for val in f_max_test:
        x_test.append([val, val])
        y_test.append([1])
    for val in t_max_test:
        x_test.append([val, val])
        y_test.append([-1])

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

def linear_SVM():

    batch_size = 256
    it_num = 5000


    x_train, x_test, y_train, y_test = pre_data()

    x = tf.placeholder(tf.float32, [None, 2], 'x')
    y = tf.placeholder(tf.float32, [None, 1], 'y')


    W = tf.Variable(tf.random_normal([2, 1]))
    b = tf.Variable(tf.random_normal([1, 1]))

    predict = tf.subtract(tf.matmul(x, W), b)

    l2_norm = tf.reduce_sum(tf.square(W))

    #alpha
    alpha = tf.constant([0.01])
    # loss = hinge loss + l2 norm * alpha
    loss = tf.add(tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(predict, y)))), tf.multiply(alpha, l2_norm))

    opt = tf.train.AdamOptimizer(0.01)
    train_step = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for it_index in range(100):
            for it in range(it_num):
                rand_index = np.random.choice(len(x_train), size=batch_size)
                x_rand = x_train[rand_index]
                y_rand = y_train[rand_index]

                tmp_loss, _ = sess.run([loss, train_step], feed_dict={x: x_rand, y: y_rand})

            sign_pre = tf.sign(predict)
            acc = tf.reduce_mean(tf.cast(tf.equal(sign_pre, y), tf.float32))

            tmp_acc = sess.run(acc, feed_dict={x: x_test, y: y_test})
            print tmp_acc


def Gaussian_Svm():
    batch_size = 256
    it_num = 5000

    x_train, x_test, y_train, y_test = pre_data()


    x = tf.placeholder(tf.float32, [None, 2], 'x')
    y = tf.placeholder(tf.float32, [None, 1], 'y')
    x_ = tf.placeholder(tf.float32, [None, 2], 'x_')

linear_SVM()