import sys
import numpy as np
import pandas as pd
import csv
import scipy as sp
import datetime
import statsmodels.api as sm
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from django.core.cache import cache

def make_graph(data2, last_p, last_p2):
    data2.plot(kind='scatter', x='a', y='b', c='clusters')
    plt.plot(last_p,last_p2,'ro') 
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'staticfiles/clusters.png'))

def getStat1(last_p):
    getStat1 = last_p
    return(getStat1)

def getStat2():
    getStat2 = 12
    return(getStat2)    
    
def generateImage():
    import reports.modules.hashtool as hashtool
    
    data_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/move_rates2.csv')
    data_file_md5 = hashtool.md5file(data_file_path)
    
    cache_key = 'vol_regime' + ':' + data_file_md5
    cached_data2 = cache.get(cache_key)
    
    if cached_data2 is not None:
        return make_graph(cached_data2)
    
    data = pd.read_csv(data_file_path, index_col=0, header=0)
    l = len(data)
    data2 = pd.DataFrame(np.zeros((7247,3)), columns=['a', 'b', 'clusters'])  ## was 7247
    
    kmeans = KMeans(n_clusters=5, random_state=0).fit(data)
    labels = kmeans.labels_

    for i in range(1,7246): ## was 7246
        data2.iloc[i,0] = data.iloc[i,0]
        data2.iloc[i,1] = data.iloc[i,1]
        data2.iloc[i,2] = int(labels[i])

    tran_mat = pd.DataFrame(np.zeros((5,5)))
    prob_mat = pd.DataFrame(np.zeros((5,5)))

    for i in range(1,7246):  ## was 7246
        a = int(data2.iloc[i,2])
        b = int(data2.iloc[i+1,2])
        tran_mat.iloc[a,b] = tran_mat.iloc[a,b] + 1

    for i in range(0,5):
        for j in range(0,5):
            prob_mat.iloc[i,j] = tran_mat.iloc[i,j]/7246   ## was 7246
            
    cache.set(cache_key, data2)
    
    last_p = data2.iloc[7246,0]
    last_p2 = data2.iloc[7246,1]

    make_graph(data2, last_p, last_p2)
    getStat1(last_p)
    getStat2(15)


