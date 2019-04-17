from __future__ import print_function

import os
import time
from datetime import timedelta
import tensorflow as tf
import numpy as np

from Classifier.hyperparams import Hyperparams as hp
from Classifier.data_load import load_train_data, load_val_data
from Classifier.Classifier_model import classifier_model

save_dir = '../CheckPionts/Classifier/'
tensorboard_dir = '../tensorboard/Classifier'
save_path = os.path.join(save_dir, 'TransE/model')
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(X, Y, sess, model):

    indices = np.random.permutation(np.arange(len(X)))
    X = X[indices]
    Y = Y[indices]

    sum_loss = .0
    sum_acc = .0
    for i in range(len(X) // hp.batch_size):
        ### Get mini-batches
        x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
        y = Y[i * hp.batch_size: (i + 1) * hp.batch_size]

        _loss, _acc = sess.run([model.loss, model.acc], {model.x: x, model.y: y})
        sum_loss += _loss
        sum_acc += _acc

    print('val acc is {:.4f}'.format(sum_acc / (len(X) // hp.batch_size)))
    return sum_loss/(len(X) // hp.batch_size), sum_acc/(len(X) // hp.batch_size)


def train():
    # Construct graph
    g = classifier_model(True)
    print("Graph loaded")

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", g.loss)
    tf.summary.scalar("accuracy", g.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    X, Y = load_train_data()

    val_X, val_Y = load_val_data()

    start_time = time.time()
    total_batch = 1  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(1, hp.num_epochs + 1):
        print('Epoch:', epoch)
        indices = np.random.permutation(np.arange(len(X)))
        X = X[indices]
        Y = Y[indices]
        for i in range(len(X) // hp.batch_size):
            ### Get mini-batches
            x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
            y = Y[i * hp.batch_size: (i + 1) * hp.batch_size]
            feed_dict = {g.x: x, g.y: y}

            if total_batch % hp.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % hp.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                loss_train, acc_train = sess.run([g.loss, g.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(val_X, val_Y, sess, g)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=sess, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))


            sess.run(g.train_op, feed_dict=feed_dict)
            total_batch += 1

        #     if total_batch - last_improved > require_improvement:
        #         # 验证集正确率长期不提升，提前结束训练
        #         print("No optimization for a long time, auto-stopping...")
        #         flag = True
        #         break  # 跳出循环
        # if flag:  # 同上
        #     break
train()