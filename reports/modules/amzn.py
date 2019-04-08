## This is a script to systematically scrape amazon prices on a daily basis
# R. Dewey 3/20/2019

import datetime
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import smtplib
from pandas import Series
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as dates

import os

def generateImage():
	time_series = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/time3.xlsx'))
	make_graph(time_series)


#### very simple graph 
def make_graph(time_series):

	# steps = len(time_series.index)
	# c = []
	# for i in range(0,steps):
	# 	if (i % 12 == 0):
	# 		c.append(i) ## This could be changed to a different arrary with months or something
	# 	else:
	# 		c.append("")
	# labels = c

	time_series.iloc[:,0] = pd.to_datetime(time_series.iloc[:,0], format='%d/%b/%Y', utc=True)

	x1 = time_series.iloc[:,0]
	x2 = time_series.iloc[:,0]
	x3 = time_series.iloc[:,0]
	x4 = time_series.iloc[:,0]

	#x1 = dates.date2num(time_series.iloc[:,0])
	#x2 = dates.date2num(time_series.iloc[:,0])

	#x1 = labels
	#x2 = labels

	y1 = time_series.iloc[:,1]
	y2 = time_series.iloc[:,2]
	y3 = time_series.iloc[:,3]
	y4 = time_series.iloc[:,4]

	plt.subplot(221)
	plt.plot(x1, y1, 'g-')
	plt.title('Batteries')
	plt.xticks(rotation=45)
	plt.ylabel('Prices')

	plt.subplot(222)
	plt.plot(x2, y2, 'r.-')	
	plt.title('Laptops')
	plt.xlabel('time (days)')
	plt.xticks(rotation=45)
	plt.ylabel('Prices')
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

	plt.subplot(223)
	plt.plot(x3, y3, 'r.-')	
	plt.title('Flatscreen Televisions')
	plt.xlabel('time (days)')
	plt.xticks(rotation=45)
	plt.ylabel('Prices')
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

	plt.subplot(224)
	plt.plot(x4, y4, 'r.-')	
	plt.title('Coffee')
	plt.xlabel('time (days)')
	plt.xticks(rotation=45)
	plt.ylabel('Prices')
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

	plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'staticfiles/Amazon.png'))



#### very simple graph 
# def make_graph(time_series):
# 	#labels = time_series.iloc[:,0].astype(numeric)
# 	labels = dates.date2num(time_series.iloc[:,0])
# 	plt.xticks(labels, rotation=0) ##rotation='vertical'
# 	time_series.iloc[:,1:3].plot(subplots=True, sharex=True, figsize=(6, 6)); plt.legend(loc='Graph')
# 	plt.title('Test')
# 	plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'staticfiles/Amazon.png'))





