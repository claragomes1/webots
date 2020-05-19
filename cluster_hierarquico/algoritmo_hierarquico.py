#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020

@author: clara
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn import metrics

saida_direita = pd.read_csv('saida_direita.csv', index_col='object')
saida_esquerda = pd.read_csv('saida_esquerda.csv', index_col = 'object')
saida_direita_esquerda = pd.read_csv('saida_direita_esquerda.csv', index_col='object')
encruzilhada_esquerda = pd.read_csv('encruzilhada_esquerda.csv', index_col='object')
encruzilhada_direita = pd.read_csv('encruzilhada_direita.csv', index_col='object')
encruzilhada = pd.read_csv('encruzilhada.csv', index_col='object')
corredor = pd.read_csv('corredor.csv', index_col='object')
teste = pd.read_csv('corredor_encruzilhada_corredor_esquerda.csv', index_col='object')


dataset = pd.concat([saida_direita, saida_esquerda, saida_direita_esquerda, encruzilhada_esquerda, encruzilhada_direita, encruzilhada, corredor, teste], axis=0, ignore_index=True)
dataset.head()

dataset['label'] = dataset['label'].replace('saida_direita',0)
dataset['label'] = dataset['label'].replace('saida_esquerda',1)
dataset['label'] = dataset['label'].replace('saida_direita_esquerda',2)
dataset['label'] = dataset['label'].replace('encruzilhada_esquerda',3)
dataset['label'] = dataset['label'].replace('encruzilhada_direita',4)
dataset['label'] = dataset['label'].replace('encruzilhada',5)
dataset['label'] = dataset['label'].replace('corredor',6)
dataset['label'] = dataset['label'].replace('teste',7)

datasetNoLabel = dataset.drop(columns=['label'])
print(datasetNoLabel)

datasetNoLabel = normalize(datasetNoLabel)


plt.figure(figsize=(25, 20))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dend = shc.dendrogram(shc.linkage(datasetNoLabel, method='ward'), truncate_mode='lastp', leaf_rotation=45., leaf_font_size=10., show_contracted=True )
plt.axhline(y=10, color='r', linestyle='--')

cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
cluster.fit_predict(datasetNoLabel)
print(cluster.labels_)

plt.figure(figsize=(10, 7))
plt.scatter(dataset.index, cluster.labels_, c=dataset['label'], cmap='rainbow')
plt.title('teste_corredor_encruzilhada_corredor_sEsquerda')
plt.xlabel('object')
plt.ylabel('cluster')

#plt.savefig('teste_1.pdf')
plt.show()

metrics.adjusted_rand_score(cluster.labels_, dataset['label'])


