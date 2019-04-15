from __future__ import print_function

import os

import tensorflow as tf

from BiLSTM_CNN.hyperparams import Hyperparams as hp
from BiLSTM_CNN.data_load import load_train_data, get_batch_data
from BiLSTM_CNN.modules import *
from BiLSTM_CNN.BiLSTM_CNN_Model import bilstm_cnn_model

save_dir = '../CheckPionts/BILSTM/BILSTM_100/BILSTM_100'
tensorboard_dir = '../tensorboard/BILSTM'

def train():
    # Construct graph
    g = bilstm_cnn_model(True)
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

    print(X.shape, Y.shape)
    for epoch in range(1, hp.num_epochs + 1):

        indices = np.random.permutation(np.arange(len(X)))
        X = X[indices]
        Y = Y[indices]

        sum_loss = 0
        batch_cnt = 0
        sum_acc = 0.0
        for i in range(len(X) // hp.batch_size):
            ### Get mini-batches
            x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
            y = Y[i * hp.batch_size: (i + 1) * hp.batch_size]
            _acc, _loss, _ = sess.run([g.acc, g.loss, g.train_op], {g.x: x, g.y: y})
            sum_loss += _loss
            sum_acc += _acc
            batch_cnt += 1
            print('\tacc:{:.3f}, loss:{:.3f}'.format(_acc, _loss))
        print('epoch {}, avg_loss:{:.3f}, avg_acc:{:.3f}'.format(epoch, sum_loss / batch_cnt, sum_acc/batch_cnt))
        # break

    saver.save(sess=sess, save_path=save_dir)
print("Done")

train()