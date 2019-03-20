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

import os

def generateImage():
	time_series = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/time2.xlsx'))
	make_graph(time_series)


#### very simple graph 
def make_graph(time_series):
	time_series.plot(subplots=True, figsize=(6, 6)); plt.legend(loc='Amazon')
	plt.title('Amazon')
	plt.savefig('staticfiles/Amazon.png')





