from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
from Classifier.hyperparams import Hyperparams as hp
from Classifier.data_load import load_val_data, load_test_data
from Classifier.Classifier_model import classifier_model

save_dir = '../CheckPionts/Classifier/'
tensorboard_dir = '../tensorboard/Classifier'
save_path = os.path.join(save_dir, 'TransE/model')

def Test():
    model = classifier_model(False)
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
    saver.restore(sess=session, save_path=save_path)

    sum_acc = .0
    for i in range(len(X) // hp.batch_size):
        ### Get mini-batches
        x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
        y = Y[i * hp.batch_size: (i + 1) * hp.batch_size]

        _acc = session.run(model.acc, {model.x: x, model.y: y})
        print(_acc)
        sum_acc += _acc

    print('test acc is {:.4f}'.format(sum_acc/(len(X)//hp.batch_size)))




Test()
print("Done")