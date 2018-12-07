from __future__ import print_function

import datetime
import numpy as np
import matplotlib
import pandas as pd
#import pylab as pl
import hmmlearn

matplotlib.use('TkAgg')
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
#from hmmlearn.hmm import warnings


print(__doc__)

###############################################################################
# Downloading the data
date1 = datetime.date(1990,1,10)  # start date
date2 = datetime.date(2015,12,23)  # end date
# get quotes from yahoo finance
# quotes = quotes_historical_yahoo_ochl("ibm", date1, date2)
# if len(quotes) == 0:
#     raise SystemExit

quotes = pd.read_csv('reports/data/test.csv', index_col=0)
#print(quotes)
#quotes = pd.read_csv('test2.csv')

# unpack quotes
# dates = np.array([q[0] for q in quotes], dtype=int)
#dates = quotes.Date.values
dates = quotes.index.values
# close_v = np.array([q[2] for q in quotes])
close_v = quotes['Last Price'].values
# volume = np.array([q[5] for q in quotes])[1:]
volume = quotes.Volume.values

# take diff of close value
# this makes len(diff) = len(close_t) - 1
# therefore, others quantity also need to be shifted
diff = close_v[1:]/close_v[:-1] - 1
dates = dates[1:]
close_v = close_v[1:]
volume = volume[1:]

# pack diff and volume for training before
# X = np.column_stack([diff, volume])
X = np.array([diff]).T

###############################################################################
# Run Gaussian HMM
print("fitting to HMM and decoding ...", end='')

# make an HMM instance and execute fit
model = GaussianHMM(n_components=4, covariance_type="full", n_iter=1000).fit([X]) # guassian emissions  
# putting [x] in brackets above was necessary for some reason. Github post online said it had to do with different versions

#model = GMMHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(X) # gaussian mixture emissions

# predict the optimal sequence of internal hidden state
# This is the 2nd problem (i.e. find the most probable sequence of states via Max-Product)

hidden_states = model.predict(X)
print(hidden_states)
#print(type(hidden_states))
#print(len(hidden_states))
print("done\n")

#transfer hidden states from numpy to pandas dataframe

predictions = pd.DataFrame(np.zeros((2357,2)), columns=['returns','states'])
predictions = predictions.reindex_like(quotes)
#print(predictions)
for i in range(0, 2357):
	predictions.iloc[i,1] = diff[i]
	predictions.iloc[i,0] = hidden_states[i]
predictions.to_csv('hidden_states.csv')

getStat1 = predictions.groupby(['Last Price']).mean()
getStat2 = predictions.iloc[-1:,0]

#print("This is state zero", state_zero)
#print("This is the final state", final_state)


#res = np.column_stack([diff, hidden_states])
###############################################################################
# print trained parameters and plot
a = model.score(X)

print("Transition matrix")
print(model.transmat_)
print()
print("Initial State Occupation")
print(model.startprob_)
print()
print("Log probability under the model")
print(a)
print()

### This is only in the event that the GMMHMM model works
#print("Weights")
#print(model.weights_)
#print(weights)
#print()


### This is the 3rd problem (i.e. learn the parameters of the model)
# print("means and vars of each hidden state")
# for i in range(model.n_components):
#     print("%dth hidden state" % i)
#     print("mean = ", model.means_[i])
#     print("var = ", np.diag(model.covars_[i]))
#     print()


### We need to write a forward/backward algorithm to solve the 1st problem. 
# That problem is: what is the probability of seeing the sequence of states observed. 

### We can also do prediction on the next time step. 
## https://stackoverflow.com/questions/44350447/hmmlearn-how-to-get-the-prediction-for-the-hidden-state-probability-at-time-t1

# import numpy as np
# from sklearn.utils import check_random_state
# sates = model.predict(X)
# transmat_cdf = np.cumsum(model.transmat_, axis=1)
# random_sate = check_random_state(model.random_state)
# next_state = (transmat_cdf[states[-1]] > random_state.rand()).argmax()


##### This section produces the graph
years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')
fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(model.n_components):
    # use fancy indexing to plot data in each state
    idx = (hidden_states == i)
    ax.plot_date(dates[idx], diff[idx], 'o', label="%dth hidden state" % i)
ax.legend()

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()

# format the coords message box
ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.fmt_ydata = lambda x: '$%1.2f' % x
ax.grid(True)

fig.autofmt_xdate()
plt.savefig('HMM_Model')
plt.show()
