import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

temp = pd.DataFrame(np.zeros((456,4)), columns=['Inflation', 'FX', '2yr', 'Equities'])
temp2 = pd.DataFrame(np.zeros((456,4)), columns=['Inflation', 'FX', '2yr', 'Equities'])

t_port = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/rb_final_aqr_nets_v3_scaled.xls'))
idx = t_port.index

temp3 = pd.DataFrame(np.zeros((456,35)), index=idx)
d_port = pd.DataFrame(np.zeros((456,35)), index=idx)
d_ret = pd.DataFrame(np.zeros((456,35)), index=idx)
total_series = pd.DataFrame(np.zeros((456,35)), index=idx)
sum1 = pd.DataFrame(np.zeros((100,35)))

countries = ['US', 'UK', 'Japan', 'Canada', 'Australia', 'Switzerland', 'Denmark', 'HK', 'Sweden', 'NZ']
frames = [None]*7

def test():
    print('test')

def read(z):
    xl_dict = {}
    sheetname_list = ['US', 'UK', 'Japan', 'Canada', 'Australia', 'Switzerland', 'Denmark', 'HK', 'Sweden', 'NZ']
    for sheet in sheetname_list:
        xl_dict[sheet] = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/rb_final_aqr_nets_v3_scaled.xls'), sheet_name=sheet)
    return(xl_dict)

def transfer(xl_dict, k):
    c = countries[k]
    temp = xl_dict[c]
    return temp 

def returns(data,inc):
    for j in range(0,4):
        for i in range(116,456):
            if data.iloc[i-1,j] == 0:
                data.iloc[i-1,j] = data.iloc[i-1,j] + .0001
            else:
                pass
            temp2.iloc[i,j] = (data.iloc[i,j]/data.iloc[i-12,j]-1) # twelve month return
            temp3.iloc[i,j+inc] = (data.iloc[i,j]/data.iloc[i-1,j]-1) # twelve month return
    #temp2.to_csv("Temp_file_for_AQR_Model"+str(k)+".csv")
    return temp2, temp3

def cleaned(k,inc):
    xl_dict = read(k)
    temp = transfer(xl_dict, k)
    temp2,temp3 = returns(temp,inc)
    return(temp2,temp3)

def dir_portfolio(data,inc):
    labels = ['Inflation', 'FX', '2yr', 'Equities']
    for j in range(0,4):    
        for i in range(116,456):
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
        for i in range(117,456):
            # The signal must be one time step backward
            # This also must be amended, so that we have the proper asset being multiplied.
            d_ret.iloc[i,j+inc] = raw_data.iloc[i,j+inc] * (sigs.iloc[i-1,j] * scale)
            total_series.iloc[i,j+inc] = total_series.iloc[i-1,j+inc] * (1+d_ret.iloc[i,j+inc])
        d_ret.iloc[0,j+inc] = labels[j]

        # Calculate the Summary Statistics
        tot_ret = ((total_series.iloc[455,j+inc]/total_series.iloc[118,j+inc])**(1/28))-1
        avg  = d_ret.iloc[1:,j+inc].mean()
        stdev = d_ret.iloc[1:,j+inc].std()

        sum1.iloc[0,j+inc] = avg
        sum1.iloc[1,j+inc] = tot_ret
        sum1.iloc[2,j+inc] = stdev
        sum1.iloc[3,j+inc] = (avg/stdev)
    return d_ret

def getData():
    #a = 12 # number of months to look back in the data
    inc = 0
    for i in range(0,7):
        ret1, ret2 = cleaned(i,inc)
        sigs = dir_portfolio(ret1,inc)
        ret = final(ret2,inc,sigs)
        inc = inc + 4

    head = list(sigs.columns.values)
    sigs.columns = sigs.iloc[0,:]
    sigs.reindex(idx)
    sigs2 = sigs.iloc[116:,:]
    #print(sigs2)

    # ret.to_csv("returns_x_signals.csv")
    # sigs2.to_csv("final_system_signals.csv")
    # sum1.to_csv("summary_stats.csv")
    # total_series.to_csv("total_return_series.csv")

    # COUNTRIES = ["US", "UK", "JAPAN", "CANADA", "AUSTRALIA", "SWISS", "DENMARK"]
    #countries = ['US', 'UK', 'Japan', 'Canada', 'Australia', 'Switzerland', 'Denmark', 'HK', 'Sweden', 'NZ']

    return [(sigs2.iloc[-1:,0:4],'Macro Momentum US'),  # -24 displays the last 2 years of data
                   (sigs2.iloc[-1:,4:8],'Macro Momentum UK'),
                   (sigs2.iloc[-1:,8:12],'Macro Momentum Japan'),
                   (sigs2.iloc[-1:,12:16],'Macro Momentum Canada'),
                   (sigs2.iloc[-1:,16:20],'Macro Momentum Australia'),
                   (sigs2.iloc[-1:,20:24],'Macro Momentum Switzerland'),
                   (sigs2.iloc[-1:,24:28],'Macro Momentum Denamrk')]