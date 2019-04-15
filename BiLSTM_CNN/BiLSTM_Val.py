from __future__ import print_function

import os

import tensorflow as tf

from BiLSTM_CNN.hyperparams import Hyperparams as hp
from BiLSTM_CNN.data_load import load_val_data, load_test_data
from BiLSTM_CNN.modules import *
from BiLSTM_CNN.BiLSTM_CNN_Model import bilstm_cnn_model

save_dir = '../CheckPionts/BILSTM/BILSTM_100/BILSTM_100'

def val():
    model = bilstm_cnn_model(False)
    print("Graph loaded")

    hp.use_kg_embd = True
    # Load data
    X, Y = load_test_data()

    indices = np.random.permutation(np.arange(len(X)))
    X = X[indices]
    Y = Y[indices]

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_dir)

    sum_acc = .0
    for i in range(len(X) // hp.batch_size):
        ### Get mini-batches
        x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
        y = Y[i * hp.batch_size: (i + 1) * hp.batch_size]

        _acc = session.run(model.acc, {model.x: x, model.y: y})
        print(_acc)
        sum_acc += _acc

    print('val acc is {:.4f}'.format(sum_acc/(len(X)//hp.batch_size)))




val()
print("Done")