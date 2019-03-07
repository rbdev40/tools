import json
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from math import log
import reports.modules.signal_gen as sg
import os
from django.core.cache import cache
# import report

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


def combine_factors(v_delta, m_delta):
    c_delta = pd.DataFrame(np.zeros((157,17)))
    c_delta = c_delta.reindex_like(v_delta)
    for j in range(0,17):
        for i in range(0,158):
            c_delta.iloc[i,j] = v_delta.iloc[i,j] + m_delta.iloc[i,j]
    return c_delta


def reformat_factors(v_delta, m_delta, c_delta):
    vrf = pd.DataFrame(np.zeros((220,17)))
    vrf = vrf.reindex_like(v_delta)
    mrf = pd.DataFrame(np.zeros((220,17)))
    mrf = mrf.reindex_like(m_delta)
    crf = pd.DataFrame(np.zeros((220,17)))
    crf = crf.reindex_like(c_delta)

    for j in range(0,17):
        for i in range(0,221):
            if v_delta.iloc[i,j] < 0:
                vrf.iloc[i,j] = 1
            else:
                vrf.iloc[i,j] = -1
            if m_delta.iloc[i,j] < 0:
                mrf.iloc[i,j] = 1
            else:
                mrf.iloc[i,j] = -1
            if c_delta.iloc[i,j] < 0:
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

def weights(mkt_data, json_data):
    (vs, ms) = sg.signals(mkt_data, json_data, 121)
    value_resource = json_data["W_VALUE"]
    momentum_resource = json_data["W_MOMENTUM"]
    v_delta = vd(vs, value_resource)
    m_delta = md(ms, momentum_resource)
    c_delta = combine_factors(v_delta, m_delta)
    (vrf,mrf,crf) = reformat_factors(v_delta, m_delta, c_delta)
    vf = v_delta.reindex(columns=WEIGHTS)
    mf = m_delta.reindex(columns=WEIGHTS)
    cf = c_delta.reindex(columns=WEIGHTS)
    vrf = vrf.reindex(columns=WEIGHTS)
    mrf = mrf.reindex(columns=WEIGHTS)
    crf = crf.reindex(columns=WEIGHTS)
    #vrf.to_csv('vrf.csv')
    #mrf.to_csv('mrf.csv')
    #crf.to_csv('crf.csv')
    return (vs, ms, vf, mf, cf, vrf, mrf, crf)

def getData():
    import reports.modules.hashtool as hashtool
    
    mkt_data_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/Datain0426_m3.csv')
    mkt_data2_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/Datain0426.xlsx')
    json_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/param.json')
    
    mkt_data_file_md5 = hashtool.md5file(mkt_data_file_path)
    mkt_data2_file_md5 = hashtool.md5file(mkt_data2_file_path)
    json_file_md5 = hashtool.md5file(json_file_path)

    cache_key = 'weight_gen' + ':' + mkt_data_file_md5 + ':' + mkt_data2_file_md5 + ':' + json_file_md5;
    cached_result = cache.get(cache_key)
    
    if cached_result is not None:
        return cached_result
    
    mkt_data = pd.read_csv(
        mkt_data_file_path, # changed from m3_csv
        index_col=0, usecols=range(sg.DATAIN_COLUMNS), dayfirst=False,
        parse_dates=True, na_values=['#VALUE!', '#NAME?', '#DIV/0!', '#N/A'],
        header=0, names=sg.data_columns())
    print(mkt_data.shape)

    #### ** Key item is that we remove index_col=0 from the read_excel file
    mkt_data2 = pd.read_excel(
        mkt_data2_file_path, # changed from m3_csv
        usecols=range(sg.DATAIN_COLUMNS), dayfirst=False,
        parse_dates=True, na_values=['#VALUE!', '#NAME?', '#DIV/0!', '#N/A'], 
        header=0, names=sg.data_columns())
    json_file = open(json_file_path)
    print(mkt_data2.shape)

    json_data = json.load(json_file)
    json_file.close()
    (vs, ms, vf, mf, cf, vrf, mrf, crf) = weights(mkt_data, json_data)

    result = [(vrf.ix[-24:,EQUITY],'Value Equity Positions'),  # -24 displays the lat 2 years of data
                   (vrf.ix[-24:,NONEQUITY],'Value Non-Equity Positions'),
                   (mrf.ix[-24:,EQUITY],'Momentum Equity Positions'),
                   (mrf.ix[-24:,NONEQUITY],'Momentum Non-Equity Positions'),
                   (crf.ix[-24:,EQUITY],'V+M Equity Positions'),
                   (crf.ix[-24:,NONEQUITY],'V+M Non-Equity Positions')]
    
    cache.set(cache_key, result)
                   
    return result