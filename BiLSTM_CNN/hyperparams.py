# -*- coding: utf-8 -*-
'''
'''
import os


class Hyperparams:
    '''Hyperparameters'''
    # data
    data_base_dir = '../Data/train_news_data/2016_election'
    train_path = os.path.join(data_base_dir, 'train.txt')
    test_path = os.path.join(data_base_dir, 'test.txt')
    val_path = os.path.join(data_base_dir, 'val.txt')
    vocab_path = os.path.join(data_base_dir, 'vocab_cnt.txt')

    kg_embd_model_dir = '../Model/kg_embedding_model/dbpedia_model/e_tr/'  #e_tr:transR   e_te:transE

    kg_embd_dim = 20 # kg embedding dim

    num_filters = 32 # num of cnn filters
    cnn_kernel_size = 3 # the kernel size

    lstm_hidden_units = 512
    fc_hidden_dim = 128  #

    # training
    batch_size = 32  # alias = N
    lr = 0.001  # learning rate. In paper, learning rate is adjusted to the global step.
    class_num = 2  # num of kinds of rel
    maxlen = 50  # Maximum number of words in a sentence. alias = T.
    num_epochs = 100
    dropout_rate = 0.1


