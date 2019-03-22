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

def make_graph(data2, data2_r, last_p, last_p2, last_p_r, last_p2_r):
    data2.plot(kind='scatter', x='VIX_Level', y='Returns', c='clusters')
    plt.plot(last_p,last_p2,'ro') 
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'staticfiles/clusters.png'))

    data2_r.plot(kind='scatter', x='MOVE_Level', y='Returns', c='clusters')
    plt.plot(last_p_r,last_p2_r,'ro') 
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'staticfiles/clusters2.png'))

def getStat1(last_p):
    getStat1 = last_p
    return(getStat1)

def getStat2(last_p2):
    getStat2 = last_p2
    return(getStat2)    

def getStat3(last_p_r):
    getStat3 = last_p_r
    return(getStat3)

def getStat4(last_p2_r):
    getStat4 = last_p2_r
    return(getStat4)   
    
def generateImage():
    from django.core.cache import cache
    import reports.modules.hashtool as hashtool
    
    data_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/vix_sp_test2.csv')
    data_file_path_r = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/move_rates2.csv')
    data_file_md5 = hashtool.md5file(data_file_path)
    

    # Caches previous
    cache_key = 'vol_regime' + ':' + data_file_md5
    cached_data2 = cache.get(cache_key)
   
    if cached_data2 is not None:
        return cached_data2
    

    ## This sections Fits the VIX history
    data = pd.read_csv(data_file_path, index_col=0, header=0)
    l = len(data)
    data2 = pd.DataFrame(np.zeros((l+1,3)), columns=['VIX_Level', 'Returns', 'clusters'])  ## was 7247

    kmeans = KMeans(n_clusters=5, random_state=0).fit(data)
    labels = kmeans.labels_

    for i in range(1,l): ## was 7246
        data2.iloc[i,0] = data.iloc[i,0]
        data2.iloc[i,1] = data.iloc[i,1]
        data2.iloc[i,2] = int(labels[i])
    

    ## This sections Fits the MOVE history
    data_r = pd.read_csv(data_file_path_r, index_col=0, header=0)
    l_r = len(data_r)
    data2_r = pd.DataFrame(np.zeros((l_r+1,3)), columns=['MOVE_Level', 'Returns', 'clusters'])  ## was 7247

    kmeans_r = KMeans(n_clusters=5, random_state=0).fit(data_r)
    labels_r = kmeans_r.labels_

    for i in range(1,l_r): ## was 7246
        data2_r.iloc[i,0] = data_r.iloc[i,0]
        data2_r.iloc[i,1] = data_r.iloc[i,1]
        data2_r.iloc[i,2] = int(labels_r[i])


    last_p = data2.iloc[l-1,0]
    last_p2 = data2.iloc[l-1,1]

    last_p_r = data2_r.iloc[l_r-1,0]
    last_p2_r = data2_r.iloc[l_r-1,1]

    make_graph(data2, data2_r, last_p, last_p2, last_p_r, last_p2_r)
    
    result = [getStat1(last_p), getStat2(last_p2), getStat3(last_p_r), getStat3(last_p2_r)]
    
    cache.set(cache_key, result, 604800)
    
    return result



    ## This section calculates transition probability matrices
    # tran_mat = pd.DataFrame(np.zeros((5,5)))
    # prob_mat = pd.DataFrame(np.zeros((5,5)))

    # for i in range(1,l):  ## was 7246
    #     a = int(data2.iloc[i,2])
    #     b = int(data2.iloc[i+1,2])
    #     tran_mat.iloc[a,b] = tran_mat.iloc[a,b] + 1

    # for i in range(0,5):
    #     for j in range(0,5):
    #         prob_mat.iloc[i,j] = tran_mat.iloc[i,j]/l   ## was 7246
