#coding:utf8

import numpy as np
import tensorflow as tf

'''
pre data
rtype: x:[None, 61] y:[None, 1]
'''
def pre_t_data():
    f_max_vec = np.load("../Data/tvec/f_max_vec.npy")
    t_max_vec = np.load("../Data/tvec/t_max_vec.npy")

    x = []
    y = []
    for vec in f_max_vec:
        x.append(vec)
        y.append([-1])
    for vec in t_max_vec:
        x.append(vec)
        y.append([1])

    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)

    x_train = x[:1600]
    x_test = x[1600:]
    y_train = y[:1600]
    y_test = y[1600:]

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

'''
pre data
rtype x:[None, 2]  y:[None, 1]
'''
def pre_data():
    f_max_train = np.load("../Data/bias/f_max_train.npy")
    t_max_train = np.load("../Data/bias/t_max_train.npy")
    f_max_test = np.load("../Data/bias/f_max_test.npy")
    t_max_test = np.load("../Data/bias/t_max_test.npy")

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for val in f_max_train:
        x_train.append([val, val])
        y_train.append([1])
    for val in t_max_train:
        x_train.append([val, val])
        y_train.append([0])
    for val in f_max_test:
        x_test.append([val, val])
        y_test.append([1])
    for val in t_max_test:
        x_test.append([val, val])
        y_test.append([0])

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

def linear_svm(dim):
    batch_size = 128
    x_train, x_test, y_train, y_test = pre_t_data()

    x_data = tf.placeholder(shape=[None, dim], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # 创建权值参数
    A = tf.Variable(tf.random_normal(shape=[dim, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # 定义线性模型: y = Ax + b
    model_output = tf.subtract(tf.matmul(x_data, A), b)

    #
    pre = tf.sign(model_output)
    acc = tf.reduce_mean(tf.cast(tf.equal(pre, y_target), tf.float32))

    # Declare vector L2 'norm' function squared
    l2_norm = tf.reduce_sum(tf.square(A))

    # Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
    alpha = tf.constant([0.01])
    classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
    loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    # 持久化
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training loop
        for i in range(5000):
            rand_index = np.random.choice(len(x_train), size=batch_size)
            rand_x = x_train[rand_index]
            rand_y = y_train[rand_index]
            tmp_loss, tmp_acc, _ = sess.run([loss, acc, train_step], feed_dict={x_data: rand_x, y_target: rand_y})
            print("train_acc: " + str(tmp_acc))

        test_acc = sess.run([acc], feed_dict={x_data: x_test, y_target: y_test})
        print(test_acc)


def Gaussian_SVM(dim):
    batch_size = 512
    x_train, x_test, y_train, y_test = pre_t_data()

    x_data = tf.placeholder(shape=[None, dim], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    prediction_grid = tf.placeholder(shape=[None, dim], dtype=tf.float32)

    # Create variables for svm
    b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

    # Apply kernel
    # Linear Kernel
    # my_kernel = tf.matmul(x_data, tf.transpose(x_data))

    # Gaussian (RBF) kernel
    # 该核函数用矩阵操作来表示
    # 在sq_dists中应用广播加法和减法操作
    # 线性核函数可以表示为：my_kernel=tf.matmul（x_data，tf.transpose（x_data）。
    gamma = tf.constant(-50.0)
    dist = tf.reduce_sum(tf.square(x_data), 1)
    dist = tf.reshape(dist, [-1, 1])
    sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
    my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    # Compute SVM Model
    # 对偶问题。为了最大化，这里采用最小化损失函数的负数tf.negative
    first_term = tf.reduce_sum(b)
    b_vec_cross = tf.matmul(tf.transpose(b), b)
    y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
    second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
    loss = tf.negative(tf.subtract(first_term, second_term))

    # Create Prediction Kernel
    # Linear prediction kernel
    # my_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

    # Gaussian (RBF) prediction kernel
    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
    rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
                          tf.transpose(rB))
    pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

    prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)
    prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    sess = tf.Session()

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training loop
    loss_vec = []
    batch_accuracy = []
    for i in range(1000):
        rand_index = np.random.choice(len(x_train), size=batch_size)
        rand_x = x_train[rand_index]
        rand_y = y_train[rand_index]
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

        acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                                 y_target: rand_y,
                                                 prediction_grid: rand_x})
        batch_accuracy.append(acc_temp)

        if (i + 1) % 25 == 0:
            print('Step #' + str(i + 1))
            print('Loss = ' + str(temp_loss))
            print('acc_temp' + str(acc_temp))

    for i in range(10):
        rand_index = np.random.choice(len(x_test), size=batch_size)
        rand_x = x_test[rand_index]
        rand_y = y_test[rand_index]
        acc = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: rand_x})
        print("test_acc" + str(acc))


Gaussian_SVM(61)