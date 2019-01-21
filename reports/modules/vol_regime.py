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


data = pd.read_csv('reports/data/vix_sp_test2.csv', index_col=0, header=0)
data = pd.read_csv('reports/data/move_rates2.csv', index_col=0, header=0)
data2 = pd.DataFrame(np.zeros((7247,3)), columns=['a', 'b', 'clusters'])

#n_samples = 7046
#n_features = 2
#shape = [n_samples, n_features]

# This is just an array for testing data
#X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

def make_graph(data2):
	#data2.to_csv('cluster_output.csv')
	data2.plot(kind='scatter', x='a', y='b', c='clusters')
	plt.savefig('staticfiles/Clusters.png')
	#plt.show()


def generateImage():
	kmeans = KMeans(n_clusters=5, random_state=0).fit(data)
	labels = kmeans.labels_
	#print(type(kmeans.labels_))
	#print(kmeans.predict(data))
	#print(kmeans.cluster_centers_)

	#np.savetxt("clusters.csv", labels, delimiter=",")

	for i in range(1,7246):
		data2.iloc[i,0] = data.iloc[i,0]
		data2.iloc[i,1] = data.iloc[i,1]
		data2.iloc[i,2] = int(labels[i])


	#columns=['a', 'b', 'c', 'd', 'e']
	tran_mat = pd.DataFrame(np.zeros((5,5)))
	prob_mat = pd.DataFrame(np.zeros((5,5)))

	for i in range(1,7246):
		a = int(data2.iloc[i,2])
		b = int(data2.iloc[i+1,2])
		tran_mat.iloc[a,b] = tran_mat.iloc[a,b] + 1


	for i in range(0,5):
		for j in range(0,5):
			prob_mat.iloc[i,j] = tran_mat.iloc[i,j]/7246

	make_graph(data2)

#print(tran_mat)
#print(prob_mat)


def getStat1():
	getStat1 = 15
	return(getStat1)

def getStat2():
	getStat2 = 12
	return(getStat2)

# plt.figure('K-Means')
# plt.scatter(data[2:,11], data[2:,1], c=kmeansoutput.labels_)
# plt.xlabel('Volatility')
# plt.ylabel('S&P 500 Level')
# plt.title('K-Means')
# plt.show()





