# this is a clean simulation script for all uses

import sys
import math
import numpy as np
import pandas as pd
import csv
import scipy as sp
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
import heapq
import decimal
#from decimal import Decimal as D

from scipy.optimize import minimize
from numpy import matrix
from numpy.linalg import inv
from heapq import nlargest

from sklearn import ensemble
from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

D = decimal.Decimal

#len = 100
#data = pd.read_csv('tsalgo.csv')
mxDrawdown = np.array(np.zeros((252,1)))
maxArrayS = pd.DataFrame(np.zeros((200,8)))
maxArray = pd.DataFrame(np.zeros((200,8)))
#maxMatS = pd.DataFrame(np.zeros((20,60)))

### -------------------------- Simulate Equity Returns Given the Parameters -----------------------------------

# sp = pd.DataFrame(np.zeros((253,3)))
# spg = pd.DataFrame(np.zeros((253,3)))
# spm = pd.DataFrame(np.zeros((253,3)))
# spou = pd.DataFrame(np.zeros((253,3)))
# sp.iloc[0,0] = 100
# spg.iloc[0,0] = 100
# spm.iloc[0,0] = 100
# spou.iloc[0,0] = 100
ret = pd.DataFrame(np.zeros((260,3)))
sd = pd.DataFrame(np.zeros((260,3)))

delta = .0039
dt = .0039
mu = .10
sigma = .10
lamda = .70

n = 20
sum1 = 0
lambda1 = .05
delta1 = 1
sig1 = .05
mu1 = 100

end = 252

### Simulates a process for stock prices ---------------------

def sim(mu, sigma, lamda, steps, proc):
	mu = float(mu)
	sigma = float(sigma)
	lamda = float(lamda)
	steps = int(steps)
	### Brownian Motion
	if (proc == "brownian_motion"):
		sp = pd.DataFrame(np.zeros((steps,3)))
		sp.iloc[0,0] = 100
		for i in range(1,steps):
			sp.iloc[i,0] = sp.iloc[i-1,0] * (1+mu*delta + sigma*math.sqrt(delta)*np.random.normal(0,1))
	### Geometric Brownian Motion
	elif (proc == "geometric_motion"):
		sp = pd.DataFrame(np.zeros((steps,3)))
		sp.iloc[0,0] = 100
		for i in range(1,steps):
			sp.iloc[i,0] = sp.iloc[i-1,0] * math.exp((mu-0.5*sigma**2)*delta + sigma*math.sqrt(delta)*np.random.normal(0,1))
	### Brownian motion with momentum
	elif (proc == "brownian_momentum"):
		sp = pd.DataFrame(np.zeros((steps,3)))
		sp.iloc[0,0] = 100
		for i in range(1,steps):
			sp.iloc[i,1] = sp.iloc[i-1,1] + mu*delta + sigma*math.sqrt(delta)*np.random.normal(0,1) + delta*lamda*(sum1/n)*ret.iloc[i-1,1]
	### OU Process
	elif (proc == "OU_process"):
		sp = pd.DataFrame(np.zeros((steps,3)))
		sp.iloc[0,0] = 100
		for i in range(1,steps):
			sp.iloc[i,1] = sp.iloc[i-1,1]*math.exp(-lamda1*delta1) + mu1*(1-math.exp(-lamda1*delta1)) + (sig1*math.sqrt((1-math.exp(-2*lamda1*delta1))/(2*lamda1)))*np.random.randn()		
	#print(sp)
	return(sp)

### Calculate the returns for a simulated process ------------

def returns(spg):
	for i in range(1,252):
		ret.iloc[i,0] = (spg.iloc[i,0]/spg.iloc[i-1,0])-1
	return ret


#sp = sim(mu, sigma, lamda)
#ret = returns(sp)

#print(sp)
#print(ret)

### ------- This Section will plot the results of the simulation and the distribution of returns


# Graph of the process
# fig = plt.figure(figsize=(16,12))
# sp.iloc[1:(end-1),0].plot()
# plt.title("Simulation")
# plt.savefig('graphs/Time Series')

# #1 Distribution of returns
# fig = plt.figure();
# ret.iloc[1:(end-1),0].diff().hist(bins=50)
# plt.title("Distribution")
# plt.savefig('graphs/Histogram')









