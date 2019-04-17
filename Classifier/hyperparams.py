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

    kg_embd_model_dir = '../Model/kg_embedding_model/dbpedia_model/e_te/'  #e_tr:transR   e_te:transE   #dbpedia_model: dbpedia KG     TML1K:KG of 1K True news

    input_dim = 61
    kg_embd_dim = 20

    hidden_dim = 512
    drop_rate = 0.1

    mode = 'max_bias' # use: max_bias avg_bias

    # training
    batch_size = 32  # alias = N
    lr = 0.001  # learning rate. In paper, learning rate is adjusted to the global step.
    class_num = 2  # num of kinds of rel
    maxlen = 5  # Maximum number of triples
    num_epochs = 500

    save_per_batch = 50
    print_per_batch = 20


