import json
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from math import log
import reports.modules.signal_gen as sg
#import report
import os
from django.core.cache import cache


## cosmetic change

EPSILON = 0.0001 # Floor for Log function

def min1(x):
    return min(1, x)

def log_func(x, center, slope):
    return slope * log(max(EPSILON, x) / center)

def linear_func(x, center, slope):
    return slope * (x - center)

VALUE_CALCULATORS = {"log_func" : log_func, "linear_func" : linear_func}

def v_delta_func(func, bound, x, center, slope):
    upper = max(-bound, func(x, center, slope))
    return min(bound, upper)

def vd(signals, resources):
    """ resources are parameters for calculators """
    d = {}
    for key, r in resources.items():
        f = VALUE_CALCULATORS[r[0]]
        s = signals[key].map(lambda x: v_delta_func(f, r[1], x, r[2], r[3]))
        d[key] = s
    return DataFrame(d)

def m_delta_func(x, delta, buffer):
    if x < -buffer:
        result = -delta
    elif x > buffer:
        result = delta
    else:
        result = x / buffer * delta
    return result

def md(signals, resources):
    d = {}
    for key, r in resources.items():
        s = signals[key].map(lambda x : m_delta_func(x, r[0], r[1]))
        d[key] = s
    return DataFrame(d)


def scale_factor(ws, hs):
    """ ws: a DataFrame of of deltas. hs: Series of hierarchical weights. """
    w = ws.mul(hs).sum(axis=1) + hs.sum()
    return w.map(lambda x: max(x, 1.0))


def combine_factors(v_delta, m_delta,time):
    c_delta = pd.DataFrame(np.zeros((time,17)))
    c_delta = c_delta.reindex_like(v_delta)
    for j in range(0,17):
        for i in range(0,time):
            c_delta.iloc[i,j] = v_delta.iloc[i,j] + m_delta.iloc[i,j]
    return c_delta


def reformat_factors(v_delta, m_delta, c_delta,time):
    stop = time-1
    vrf = pd.DataFrame(np.zeros((stop,17)))
    vrf = vrf.reindex_like(v_delta)
    mrf = pd.DataFrame(np.zeros((stop,17)))
    mrf = mrf.reindex_like(m_delta)
    crf = pd.DataFrame(np.zeros((stop,17)))
    crf = crf.reindex_like(c_delta)

    for j in range(0,17):
        for i in range(0,time):
            if v_delta.iloc[i,j] > 0:
                vrf.iloc[i,j] = 1
            else:
                vrf.iloc[i,j] = -1
            if m_delta.iloc[i,j] > 0:
                mrf.iloc[i,j] = 1
            else:
                mrf.iloc[i,j] = -1
            if c_delta.iloc[i,j] > 0:
                crf.iloc[i,j] = 1
            else:
                crf.iloc[i,j] = -1

    return(vrf,mrf,crf)


WEIGHTS = ["US_EQ", "EUR_EQ", "UK_EQ", "JP_EQ", "PXJ_EQ", "CA_EQ", "EM_EQ",
            "US_REIT", "XUS_REIT", "OIL1", "OIL2", "GOLD", "US_TIP10",
            "US_GOV10", "MUNI", "CORP", "CORP_HY"]
EQUITY = ["US_EQ", "EUR_EQ", "UK_EQ", "JP_EQ", "PXJ_EQ", "CA_EQ", "EM_EQ"]
NONEQUITY = ["US_REIT", "XUS_REIT", "OIL1", "OIL2", "GOLD", "US_TIP10",
             "US_GOV10", "MUNI", "CORP", "CORP_HY"]

def weights(mkt_data, json_data,time):
    (vs, ms) = sg.signals(mkt_data, json_data, 121)
    value_resource = json_data["W_VALUE"]
    momentum_resource = json_data["W_MOMENTUM"]
    v_delta = vd(vs, value_resource)
    m_delta = md(ms, momentum_resource)
    c_delta = combine_factors(v_delta, m_delta, time)
    (vrf,mrf,crf) = reformat_factors(v_delta, m_delta, c_delta, time)
    vf = v_delta.reindex(columns=WEIGHTS)
    mf = m_delta.reindex(columns=WEIGHTS)
    cf = c_delta.reindex(columns=WEIGHTS)
    vrf = vrf.reindex(columns=WEIGHTS)
    mrf = mrf.reindex(columns=WEIGHTS)
    crf = crf.reindex(columns=WEIGHTS)
    #vf.to_csv('vf.csv')
    #mf.to_csv('mf.csv')
    #cf.to_csv('cf.csv')
    #vrf.to_csv('vrf.csv')
    #mrf.to_csv('mrf.csv')
    #crf.to_csv('crf.csv')
    return (vs, ms, vf, mf, cf, vrf, mrf, crf)

def getData():
    import reports.modules.hashtool as hashtool

    mkt_data2_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/vm_system_pull_simple.xlsx')
    json_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/new_param.json')
    
    mkt_data2_file_md5 = hashtool.md5file(mkt_data2_file_path)
    json_file_md5 = hashtool.md5file(json_file_path)

    cache_key = 'weight_gen' + ':' + mkt_data2_file_md5 + ':' + json_file_md5;
    cached_result = cache.get(cache_key)
    
    if cached_result is not None:
        return cached_result
    

    #### ** Key item is that we remove index_col=0 from the read_excel file
    time = 238
    mkt_data2 = pd.read_excel(
        mkt_data2_file_path, # changed from m3_csv
        usecols=range(sg.DATAIN_COLUMNS), dayfirst=False, index_col=0,
        parse_dates=True, na_values=['#VALUE!', '#NAME?', '#DIV/0!', '#N/A'], 
        header=0, names=sg.data_columns())
    
    json_file = open(json_file_path)
    print(mkt_data2.shape)

    json_data = json.load(json_file)
    json_file.close()
    (vs, ms, vf, mf, cf, vrf, mrf, crf) = weights(mkt_data2, json_data,time)

    frames = (vs.ix[-1:,EQUITY].T, vf.ix[-1:,EQUITY].T, vrf.ix[-1:,EQUITY].T, mrf.ix[-1:,EQUITY].T, mf.ix[-1:,EQUITY].T, crf.ix[-1:,EQUITY].T)
    frames2 = (vs.ix[-1:,NONEQUITY].T, vf.ix[-1:,NONEQUITY].T, vrf.ix[-1:,NONEQUITY].T, mrf.ix[-1:,NONEQUITY].T, mf.ix[-1:,NONEQUITY].T, crf.ix[-1:,NONEQUITY].T)
    
    result = pd.concat(frames, axis=1)
    result2 = pd.concat(frames2, axis=1)

    result.columns = ['V_Signal', 'V_Delta', 'V_Weight', 'M_Weight','M_Delta', 'C_Weight']
    result2.columns = ['V_Signal', 'V_Delta', 'V_Weight', 'M_Weight', 'M_Delta','C_Weight']

    result = [(vrf.ix[-2:,EQUITY],'<h3>Value Equity Positions</h3>'),  # -24 displays the lat 2 years of data
                   (vrf.ix[-2:,NONEQUITY],'<h3>Value Non-Equity Positions</h3>'),
                   (mrf.ix[-2:,EQUITY],'<h3>Momentum Equity Positions</h3>'),
                   (mrf.ix[-2:,NONEQUITY],'<h3>Momentum Non-Equity Positions</h3>'),
                   (crf.ix[-2:,EQUITY],'<h3>V+M Equity Positions</h3>'),
                   (crf.ix[-2:,NONEQUITY],'<h3>V+M Non-Equity Positions</h3>')]
    
    cache.set(cache_key, result, 604800)
                   
    return result