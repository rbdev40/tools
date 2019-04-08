import sys
import math
import numpy as np
import pandas as pd
import csv
import scipy as sp
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
import os
import reports.modules.hashtool as hashtool

from scipy.optimize import minimize
from numpy import matrix
from numpy.linalg import inv
from pandas import Panel, DataFrame

corMat = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/cormat.csv'))
covMat = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/covar.csv'))
Data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/weights6.csv'), index_col=0,header=0)

#l = len(corMat)
l = 5
## change all values to l to make code more dynamic


#initialize blank matrices
stdMat = pd.DataFrame(np.zeros((l,l)))  ### Every <l> wreplaces the value <18>
covMatrix = pd.DataFrame(np.zeros((l,l)))
prices = pd.DataFrame(np.zeros((300,l)))
fin_corr = pd.DataFrame(np.zeros((300,l)))
portfolio = pd.DataFrame(np.zeros((300,1)))

mxDrawdown = np.array(np.zeros((252,1)))
maxArray = pd.DataFrame(np.zeros((500,10)))

weight = np.zeros(l)
mu = np.zeros(l)
sigma = np.zeros(l)

for i in range(0,l):
    weight[i] = Data.iloc[0,i]  
    mu[i] = Data.iloc[1,i]  
    sigma[i] = Data.iloc[2,i]

# ------ Step 1 - Calculate Portfolio Variance

# Expected Return of the Portfolio:
portExpRet = weight.dot(mu)

# Expected Variance of the Portfolio
portVar = weight.T.dot(covMat.dot(weight))
portVol = math.sqrt(portVar) #*math.sqrt(252)
portSharpe = (portExpRet/portVol)

# ------ Step 2 - Simulate Portfolio Returns

n_days = 252
n_assets = l
n_sims = 1000
dt = .005
x = np.linalg.cholesky(corMat)

def sim_cor(mu,sigma,x):
    
    rand_values = np.random.standard_normal(size = (n_days, n_assets))
    corr_values = rand_values.dot(x)*sigma*math.sqrt(dt)
    portfolio.iloc[0,0] = 100
    
    for i in range(0,252):
        for j in range(0,l):
            fin_corr.iloc[i,j] = corr_values[i][j]

    for j in range(0,l):
        prices.iloc[0,j] = 100 

    for i in range(1,252):
        ret = 0
        for j in range(0,l):
            # This is modeled as GBM (i.e. stock returns). Think more about this
            b = (mu[j]*dt) + fin_corr.iloc[i,j]
            prices.iloc[i,j] = prices.iloc[i-1,j] * (1+b)
            ret = ret + ((prices.iloc[i,j]/prices.iloc[i-1,j])-1)*weight[j]
        portfolio.iloc[i,0] = portfolio.iloc[i-1,0]*(1+ret)
    return portfolio, prices



# Graph The Results of the Correlated Assets
def make_graph(prices):
    fig = plt.figure(figsize=(32,24))
    prices.iloc[0:252,0:12].plot()
    plt.title('Correlation')
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'staticfiles/drawdown'))


## --- Step 4 - Calculates the expected drawdown of the correlated portfolio ---------

def drawDown(data):
    for i in range(1,251):
        t = data.iloc[i,0]
        t1 = data.iloc[i+1,0]
        if(t1 < t):
            peak = t
            low = t1
            for j in range(i,251):
                x1 = data.iloc[j+1,0]
                if(x1 <= low):
                    low = x1
                    #print("value of low", low)
                if(x1 < peak):
                    pass
                elif(x1 > peak):
                    break   
            mxDrawdown[i] = (low - peak)/peak
        else:
            mxDrawdown[i] = 0

    Min = min(mxDrawdown)
    return(Min)

### ------- This Code will Calculate the Average Largest Drawdown -----------

def generateImage():
    num = 10

    for k in range(0,10):
        portfolio, prices = sim_cor(mu, sigma, x)
        maxArray.iloc[k,1] = drawDown(portfolio)

    make_graph(prices)

def getDrawdown():
    p = (maxArray.iloc[:10,1].min()*100)
    z = round(p,2) 
    return(z)

def getVar():
    t = (maxArray.iloc[:10,1].mean()*100)
    q = round(t,2)
    return(q)

