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

from scipy.optimize import minimize
from numpy import matrix
from numpy.linalg import inv
from pandas import Panel, DataFrame

corMat = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/cormat.csv'))
covMat = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/covar.csv'))
Data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/weights6.csv'), index_col=0,header=0)
stdMat = pd.DataFrame(np.zeros((18,18)))
covMatrix = pd.DataFrame(np.zeros((18,18)))
prices = pd.DataFrame(np.zeros((300,18)))
fin_corr = pd.DataFrame(np.zeros((300,18)))
fin_corr = pd.DataFrame(np.zeros((300,18)))
portfolio = pd.DataFrame(np.zeros((300,1)))

mxDrawdown = np.array(np.zeros((252,1)))
maxArrayS = np.array(np.zeros((500,10)))
maxArray = pd.DataFrame(np.zeros((500,10)))
maxMatS = pd.DataFrame(np.zeros((20,60)))

weight = np.zeros(18)
mu = np.zeros(18)
sigma = np.zeros(18)

for i in range(0,18):
    weight[i] = Data.iloc[0,i]  
    mu[i] = Data.iloc[1,i]  
    sigma[i] = Data.iloc[2,i]   

#valuesArr = np.array(np.zeros((18,18)))

# ------ Step 1 - Calculate Portfolio Variance

# for i in range(0,18):
#       stdMat.iloc[i,i] = sigma[i] 

# print(stdMat)
# print(corMat.shape)

# a = np.dot(np.dot(stdMat,corMat),stdMat)

# for i in range(0,18):
#   for j in range(0,18):
#       covMat.iloc[i,j] = a[i][j]

# print(covMat)
# covMat.to_csv('covMat.csv')

# Expected Return of the Portfolio:
portExpRet = weight.dot(mu)
# print("Portfolio Returns", portExpRet)

# Expected Variance of the Portfolio
portVar = weight.T.dot(covMat.dot(weight))
portVol = math.sqrt(portVar) #*math.sqrt(252)
portSharpe = (portExpRet/portVol)
# print("Portfolio Variance", portVar)
# print("Portfolio Volatility", portVol)
# print("Portfolio Sharpe", portSharpe)

#valuesArr[0][0] = portExpRet
#valuesArr[0][1] = portVar

# ------ Step 2 - Simulate Portfolio Returns

n_days = 252
n_assets = 18
n_sims = 1000
dt = .005
#mu = rets.mean().values
#sigma = rets.std().values*sqrt(252)
x = np.linalg.cholesky(corMat)

def sim_cor(mu,sigma,x):
    
    rand_values = np.random.standard_normal(size = (n_days, n_assets))
    corr_values = rand_values.dot(x)*sigma*math.sqrt(dt)
    portfolio.iloc[0,0] = 100
    
    for i in range(0,252):
        for j in range(0,18):
            fin_corr.iloc[i,j] = corr_values[i][j]

    for j in range(0,18):
        prices.iloc[0,j] = 100 

    for i in range(1,252):
        ret = 0
        for j in range(0,18):
            # This is modeled as GBM (i.e. stock returns). Think more about this
            #a = math.exp((mu[j]-.5*sigma[j]**2)*dt + math.sqrt(dt)*fin_corr.iloc[i,j])
            #b = (1+mu[j]*dt + sigma[j]*math.sqrt(dt)*np.random.normal(0,1))
            b = (mu[j]*dt) + fin_corr.iloc[i,j]
            prices.iloc[i,j] = prices.iloc[i-1,j] * (1+b)
            ret = ret + ((prices.iloc[i,j]/prices.iloc[i-1,j])-1)*weight[j]
            #print("this is the return",ret)
            #print("this is the weight", weight[j]) 
        portfolio.iloc[i,0] = portfolio.iloc[i-1,0]*(1+ret)
    # portfolio.to_csv("output/portfolio.csv")
    # prices.to_csv("output/prices.csv")
    #fin_corr.to_csv("fin_corr.csv")
    return portfolio, prices



# Graph The Results of the Correlated Assets
def make_graph(prices):
    fig = plt.figure(figsize=(16,12))
    prices.iloc[0:252,0:18].plot()
    plt.title('Correlation')
    plt.savefig('staticfiles/drawdown')

# Output results to a csv File
# with open('output2.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows( ('Ret','Var') )
#     writer.writerows(valuesArr)


## --- Step 4 - Calculates the expected drawdown of the correlated portfolio ---------

def drawDown(data):
    for i in range(1,251):
        t = data.iloc[i,0]
        t1 = data.iloc[i+1,0]
        #print("value of t", t)
        #print("value of t1", t1)
        if(t1 < t):
            peak = t
            low = t1
            #print("value of peak", peak)
            #low = t1
            for j in range(i,251):
                #x = data.iloc[j,0]
                x1 = data.iloc[j+1,0]
                if(x1 <= low):
                    low = x1
                    #print("value of low", low)
                if(x1 < peak):
                    pass
                elif(x1 > peak):
                    break   
            #print("we experienced a drawdown")
            mxDrawdown[i] = (low - peak)/peak
            #print(mxDrawdown[i])
        else:
            #print("We did not experience a drawdown")
            mxDrawdown[i] = 0

    Min = min(mxDrawdown)
    #print(Min)
    return(Min)

### ------- This Code will Calculate the Average Largest Drawdown -----------

def generateImage():
    num = 100

    for k in range(0,100):
        print("Simulation #", k )
        portfolio, prices = sim_cor(mu, sigma, x)
        maxArray.iloc[k,1] = drawDown(portfolio)

    make_graph(prices)

# z = maxArray.iloc[:100,1].mean()
# print(z)
#
# q = maxArray.iloc[:100,1].min()
# print(q)


# np.savetxt("output/MaxDDArray_Corr.csv", maxArray, delimiter=",")
# avg = np.average(maxArray)

