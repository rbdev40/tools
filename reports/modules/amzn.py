## This is a script to systematically scrape amazon prices on a daily basis
# R. Dewey 10/25/2017


import datetime
import numpy as np
import pandas as pd
#import requests
#import bs4
#import BeautifulSoup

from bs4 import BeautifulSoup
import smtplib
#import urllib3
import urllib3
#from urllib import request

import requests

import decimal
from pandas import Series
from matplotlib import pyplot
import matplotlib.pyplot as plt
#from crontab import CronTab

#df2 = pd.DataFrame([0], index=[j])
#df2 = pd.DataFrame([0], index=[j], columns=['seed'])
#time_series = pd.concat([time_series, df2])
#print(inc)
#inc.iloc[j,0] = today
#inc.to_csv('/Users/development/Desktop/inc.csv')

#http = urllib3.PoolManager()

def findPrice(url, selector):
	userAgent = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.86 Safari/537.36'}
	#userAgent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36"
	#req = urllib.request.urlopen(url)
	#req = urllib.request(url, None, {'User-Agent': userAgent})
	#html = urlopen(req).read()	
	#soup = BeautifulSoup(req.data)
	#soup = BeautifulSoup(html, "lxml")
	#http = urllib3.PoolManager(10, headers=userAgent)
	#http = urllib3.PoolManager()
	# response = http.request('GET', url)
	#response = http.request('GET', url, preload_content=False)
	#soup = BeautifulSoup(response)


	r = requests.get(url)
	soup = BeautifulSoup(r.content)

	return decimal.Decimal(soup.select(selector)[0].contents[0].strip().strip("$"))

try:
	batteries = findPrice("https://www.amazon.com/Energizer-Batteries-Battery-Alkaline-E91BP-24/dp/B004U429AQ", "#unifiedPrice_feature_div .a-size-large")
	laptops = findPrice("https://www.amazon.com/dp/B07GBJW7TS/ref=dp_prsubs_1", "#price .a-color-price")
	#laptops = findPrice("https://www.amazon.com/dp/B07GBJW7TS/ref=dp_prsubs_1") # erroneous call for testing purposes

except:
	print("An exception has occured")
	recipients = ['richdewey@gmail.com', 'rich@royalbridgecap.com']
	smtpObj = smtplib.SMTP('smtp.office365.com', 587) 
	smtpObj.ehlo()
	smtpObj.starttls()
	smtpObj.login('rich@royalbridgecap.com', 'Mot*bur66')
	smtpObj.sendmail('rich@royalbridgecap.com',recipients, 'Subject: Problem with amazon script') # first is "from" second is "to"
	smtpObj.quit()


#print message/send email that there has been an error

#batteries = 12.50
#laptops = 234

### This section Calculates The Correct Time and Prints Time Series
now = datetime.datetime.now()
today = now.strftime("%Y-%m-%d")
#today = now.strftime("%m/%d/%y") # alternative format


### This section will update the csv file
#time_series = pd.read_csv('/Users/development/Desktop/time_series2.csv', index_col=0)
time_series = pd.read_excel('/Users/development/Desktop/dewey-dist/reports/data/time2.xlsx')
last = (time_series.last_valid_index())
pos = last+1

last_date = (time_series.iloc[last,0].strftime("%Y-%m-%d"))
#last_date = (time_series.iloc[last,0].strftime("%m/%d/%y")) # alternative format

#print(today)
#print("The is the last value", last)
#print("This is today", today)
#print("this is last date", last_date)

time_series.iloc[pos,0] = today
time_series.iloc[pos,1] = batteries
time_series.iloc[pos,2] = laptops
time_series.to_excel("/Users/development/Desktop/dewey-dist/reports/data/time2.xlsx")

def generateImage():
	#time_series = pd.read_csv('/Users/development/Desktop/time_series2.csv', index_col=0)
	time_series = pd.read_excel('/Users/development/Desktop/dewey-dist/reports/data/time2.xlsx')
	#last = (time_series.last_valid_index())
	#pos = last+1

	#last_date = (time_series.iloc[last,0].strftime("%Y-%m-%d"))
	#last_date = (time_series.iloc[last,0].strftime("%m/%d/%y")) # alternative format

	#print(today)
	#print("The is the last value", last)
	#print("This is today", today)
	#print("this is last date", last_date)

	#time_series.iloc[pos,0] = today
	#time_series.iloc[pos,1] = batteries
	#time_series.iloc[pos,2] = laptops
	#time_series.to_excel("/Users/development/Desktop/dewey-dist/reports/data/time2.xlsx")

	make_graph(time_series)


#### very simple graph 

def make_graph(time_series):
	#timeseries = Series.from_csv('daily-minimum-temperatures.csv', header=0)
	#time_series.plot()
	time_series.plot(subplots=True, figsize=(6, 6)); plt.legend(loc='Amazon')
	plt.title('Amazon')
	plt.savefig('staticfiles/Amazon.png')
	#plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'staticfiles/Amazon'))
	#plt.show()





