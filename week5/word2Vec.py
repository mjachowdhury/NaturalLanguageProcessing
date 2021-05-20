#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:21:54 2018

@author: haithem.afli
"""

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

# define training data
sentences = [['this', 'is', 'the', 'second', 'lecture', 'about', 'word2vec'],
			['this', 'is', 'the', 'first', 'slide'],
			['yet', 'another', 'slide'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]


# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)


# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])

# summarize vocabulary
words = list(model.wv.vocab)
print(words)

for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

# access vector for one word
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)



