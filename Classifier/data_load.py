# -*- coding: utf-8 -*-
from __future__ import print_function
from Classifier.hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import codecs
from Model.kg_embedding_model.triple2vec import triple2vec

def load_label2id():
    relid2target_dict = {'fake': 0, 'true': 1}
    return relid2target_dict

def triple_kg_embd(content, kg_embedding_model):
    content = content.split('\t')
    h = np.array(kg_embedding_model.entity2vec(content[0].strip()))
    r = np.array(kg_embedding_model.relation2vec(content[1].strip()))
    t = np.array(kg_embedding_model.entity2vec(content[2].strip()))
    bias = sum([x**2 for x in h + r - t])
    embd = np.concatenate([h, r, t, [bias]])
    return embd

def create_data(sentences, targets, kg_embedding_model):

    x_list, y_list, Sentences, Targets = [], [], [], []

    print('create data')
    pbar = tqdm(range(len(sentences)))
    for i, (sentence, target) in enumerate(zip(sentences, targets)):
        # print('processing No.{}'.format(i))
        pbar.update()
        x = []
        for content in sentence.strip().split('<END>'):
            if content:
                x.append(triple_kg_embd(content.strip(), kg_embedding_model))
        x = np.array(x)
        y = [target]
        if hp.mode == 'max_bias':
            x = x[np.argmax(x[:,-1])]
        elif hp.mode == 'avg_bias':
            x = np.mean(x, axis=0)
        # if len(x) > hp.maxlen:
        #     x = x[:hp.maxlen]
        x_list.append(np.array(x, np.float32))
        y_list.append(np.array(y, np.float32))
        Sentences.append(sentence)
        Targets.append(target)
    print('create finished')

    X = np.array(x_list)
    Y = np.array(y_list)

    # X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    # Pad      
    # X = np.zeros([len(x_list), hp.maxlen, hp.input_dim], np.float32)
    # Y = np.zeros([len(y_list), hp.class_num], np.int32)
    # for i, (x, y) in enumerate(zip(x_list, y_list)):
    #     X[i] = np.lib.pad(x, ((0, (hp.maxlen - len(x))), (0, 0)), 'constant', constant_values=0)
    #     Y[i] = y
    
    return X, Y, Sentences, Targets

def load_train_data():
    kg_embedding_model = triple2vec(20, hp.kg_embd_model_dir)
    label2id = load_label2id()

    train_triples = []
    train_labels = []
    for label, content in [line.strip().split('<#>') for line in codecs.open(hp.train_path, 'r', 'utf-8').readlines() if
                           line]:
        train_triples.append(content.strip())
        train_labels.append(label2id[label])

    # print('len', len(train_sentences), len(train_targets))
    X, Y, Sources, Targets = create_data(train_triples, train_labels, kg_embedding_model)
    return X, Y


def load_val_data():
    kg_embedding_model = triple2vec(20, hp.kg_embd_model_dir)
    label2id = load_label2id()

    train_triples = []
    train_labels = []
    for label, content in [line.strip().split('<#>') for line in codecs.open(hp.val_path, 'r', 'utf-8').readlines() if
                           line]:
        train_triples.append(content.strip())
        train_labels.append(label2id[label])

    # print('len', len(train_sentences), len(train_targets))
    X, Y, Sources, Targets = create_data(train_triples, train_labels, kg_embedding_model)
    return X, Y


def load_test_data():
    kg_embedding_model = triple2vec(20, hp.kg_embd_model_dir)
    label2id = load_label2id()

    train_triples = []
    train_labels = []
    for label, content in [line.strip().split('<#>') for line in codecs.open(hp.test_path, 'r', 'utf-8').readlines() if
                           line]:
        train_triples.append(content.strip())
        train_labels.append(label2id[label])

    # print('len', len(train_sentences), len(train_targets))
    X, Y, Sources, Targets = create_data(train_triples, train_labels, kg_embedding_model)
    return X, Y

def get_batch_data():

    X, Y = load_train_data()
    # calc total batch count
    num_batch = len(X) // hp.batch_size

    indices = np.random.permutation(np.arange(len(X)))
    X = X[indices]
    Y = Y[indices]

    for i in range(num_batch):
        batch_x = X[i*hp.batch_size: (i + 1)*hp.batch_size]
        batch_y = Y[i*hp.batch_size: (i + 1)*hp.batch_size]
        yield  batch_x, batch_y

# X, Y = load_val_data()
# print(X.shape, Y.shape)
# print(X[0:10])
# print(Y[0:10])