import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#data_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/rb_final_aqr_nets_v3_scaled.xls')
data_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/rb_final_aqr_nets_v5_scaled_cleaned.xls')

#l = len(data_file_path)
l = 471
print(l)

temp = pd.DataFrame(np.zeros((l,4)), columns=['Inflation', 'FX', '2yr', 'Equities']) # was 456
temp2 = pd.DataFrame(np.zeros((l,4)), columns=['Inflation', 'FX', '2yr', 'Equities']) # was 456

t_port = pd.read_excel(data_file_path)
idx = t_port.index

temp3 = pd.DataFrame(np.zeros((l,35)), index=idx) # was 456
d_port = pd.DataFrame(np.zeros((l,35)), index=idx) # was 456
d_ret = pd.DataFrame(np.zeros((l,35)), index=idx) # was 456
total_series = pd.DataFrame(np.zeros((l,35)), index=idx) # was 456
sum1 = pd.DataFrame(np.zeros((100,35)))

countries = ['US', 'UK', 'Japan', 'Canada', 'Australia', 'Switzerland', 'Denmark', 'HK', 'Sweden', 'NZ']
frames = [None]*5

def read(z):
    xl_dict = {}
    sheetname_list = ['US', 'UK', 'Japan', 'Canada', 'Australia', 'Switzerland', 'Denmark', 'HK', 'Sweden', 'NZ']
    for sheet in sheetname_list:
        xl_dict[sheet] = pd.read_excel(data_file_path, sheet_name=sheet)
    return(xl_dict)

def transfer(xl_dict, k):
    c = countries[k]
    temp = xl_dict[c]
    return temp 

def returns(data,inc):
    for j in range(0,4):
        for i in range(116,l): # was 456
            if data.iloc[i-1,j] == 0:
                data.iloc[i-1,j] = data.iloc[i-1,j] + .0001
            else:
                pass
            temp2.iloc[i,j] = (data.iloc[i,j]/data.iloc[i-12,j]-1) # twelve month return
            temp3.iloc[i,j+inc] = (data.iloc[i,j]/data.iloc[i-1,j]-1) # twelve month return
    return temp2, temp3

def cleaned(k,inc):
    xl_dict = read(k)
    temp = transfer(xl_dict, k)
    temp2,temp3 = returns(temp,inc)
    return(temp2,temp3)

def dir_portfolio(data,inc):
    labels = ['Inflation', 'FX', '2yr', 'Equities']
    for j in range(0,4):    
        for i in range(116,l): # was 456
            if data.iloc[i,j] > 0.01:
                d_port.iloc[i,j+inc] = 1
            else:
                d_port.iloc[i,j+inc] = -1
        d_port.iloc[0,j+inc] = labels[j]
    return d_port

# Calculates the Returns
# Must be Careful that we take previous signal * by today's return
def final(raw_data,inc,sigs):
    # raw data is returns
    # sigs is the signal
    scale = 1
    labels = ['Inflation', 'FX', '2yr', 'Equities']
    for j in range(0,4):    
        total_series.iloc[116,j+inc] = 100
        for i in range(117,l): # was 456
            # The signal must be one time step backward
            # This also must be amended, so that we have the proper asset being multiplied.
            d_ret.iloc[i,j+inc] = raw_data.iloc[i,j+inc] * (sigs.iloc[i-1,j] * scale)
            total_series.iloc[i,j+inc] = total_series.iloc[i-1,j+inc] * (1+d_ret.iloc[i,j+inc])
        d_ret.iloc[0,j+inc] = labels[j]

        # Calculate the Summary Statistics
        tot_ret = ((total_series.iloc[l-1,j+inc]/total_series.iloc[118,j+inc])**(1/28))-1  # was 455
        avg  = d_ret.iloc[1:,j+inc].mean()
        stdev = d_ret.iloc[1:,j+inc].std()

        sum1.iloc[0,j+inc] = avg
        sum1.iloc[1,j+inc] = tot_ret
        sum1.iloc[2,j+inc] = stdev
        sum1.iloc[3,j+inc] = (avg/stdev)
    return d_ret

def getData():
    from django.core.cache import cache
    import reports.modules.hashtool as hashtool
    
    data_file_md5 = hashtool.md5file(data_file_path)
    
    cache_key = 'rb_aqr_macro' + ':' + data_file_md5
    cached_result = cache.get(cache_key)

    if cached_result is not None:
        print('used cache')
        return cached_result
    else:
        print('did not use cache')
    
    inc = 0
    for i in range(0,5):
        ret1, ret2 = cleaned(i,inc)
        sigs = dir_portfolio(ret1,inc)
        ret = final(ret2,inc,sigs)
        inc = inc + 4

    #head = list(sigs.columns.values)
    #sigs.columns = sigs.iloc[0,:]
    #sigs.reindex(idx)
    sigs2 = sigs.iloc[116:,:]
    sigs2.set_index('Date')
    #sigs2.iloc[0,0] = 'Date'

    result = [(sigs2.iloc[-2:,0:4],'<h3>Macro Momentum US</h3>'),  # -24 displays the last 2 years of data
                   (sigs2.iloc[-2:,4:8],'<h3>Macro Momentum UK</h3>'),
                   (sigs2.iloc[-2:,8:12],'<h3>Macro Momentum Japan</h3>'),
                   (sigs2.iloc[-2:,12:16],'<h3>Macro Momentum Canada</h3>'),
                   (sigs2.iloc[-2:,16:20],'<h3>Macro Momentum Australia</h3>'),
                   (sigs2.iloc[-2:,20:24],'<h3>Macro Momentum Switzerland</h3>'),
                   (sigs2.iloc[-2:,24:28],'<h3>Macro Momentum Denamrk</h3>')]

    cache.set(cache_key, result, 604800)
                   
    return result