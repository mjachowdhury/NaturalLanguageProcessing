# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 00:12:15 2020

@author: mohammed alom R00144214
"""


import numpy as np
import codecs
from collections import defaultdict
from numpy.random import permutation, shuffle, rand
from sklearn.model_selection import train_test_split

#File dir
corpus = 'E:/MSc/NLP/Assignment1/Mohammed_Alom_R00144214/data'
#Loading files
train_file = 'E:/MSc/NLP/Assignment1/Mohammed_Alom_R00144214/data/MADAR-Corpus-26-train.tsv'
dev_file = 'E:/MSc/NLP/Assignment1/Mohammed_Alom_R00144214/data/MADAR-Corpus-26-dev.tsv'

#train_file = 'E:/MSc/NLP/Assignment1/Mohammed_Alom_R00144214/data/MADAR-Corpus-6-train.tsv'
#dev_file = 'E:/MSc/NLP/Assignment1/Mohammed_Alom_R00144214/data/MADAR-Corpus-6-dev.tsv'

test_file = 'E:/MSc/NLP/Assignment1/Mohammed_Alom_R00144214/data/test.tsv'


def prepareTrainData(corpus, include_dev=False):
    dataset = defaultdict(list)
    X, y = [], []

    with codecs.open(train_file,encoding="utf8") as training:
        for i, line in enumerate(training):
            sentence_label = line.strip().split('\t')
            X.append(sentence_label[0])
            y.append(sentence_label[1])
            dataset[sentence_label[1]].append(sentence_label[0])

    if include_dev:
        with codecs.open(dev_file,encoding="utf8") as training:
            for i, line in enumerate(training):
                sentence_label = line.strip().split('\t')
                X.append(sentence_label[0])
                y.append(sentence_label[1])
                dataset[sentence_label[1]].append(sentence_label[0])
    return train_test_split(X, y, test_size=1000, shuffle=True)


def prepareDevData(corpus):
    dataset = defaultdict(list)
    X, y = [], []

    with codecs.open(dev_file,encoding="utf8") as training:
        for i, line in enumerate(training):
            sentence_label = line.strip().split('\t')
            X.append(sentence_label[0])
            y.append(sentence_label[1])
            dataset[sentence_label[1]].append(sentence_label[0])
    return X, y


def prepareTestData(corpus):
    X = []
    with codecs.open(test_file,encoding="utf8") as training:
        for i, line in enumerate(training):
            sentence = line.strip()
            X.append(sentence)
    return X