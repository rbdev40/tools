#-------------------------------------------------------------------------------
#  SIGNAL
#  Richard Dewey
#  July 28, 2018
#-------------------------------------------------------------------------------

import csv
import json
from math import log
import itertools
#from functools import update_wrapper
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

#-------------------------------------------------------------------------------
#  UTILITY FUNCTIONS
#-------------------------------------------------------------------------------

DATAIN_COLUMNS = 247

def data_columns():
    """ Use excel column headings for data keys """
    head = [x for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    tail = [x + y for x in head for y in head]
    all_cols = head + tail
    #print(all_cols)
    return all_cols[:DATAIN_COLUMNS]

def make_gen(size, vector, start=0, length=1):
    """ Return a generator that is either a sliding slice of vector if
            length > 1 or a value if length = 1 or a repeated constant"""
    try:
        len(vector) # if the vector does not have a length then its a scalar
        end = start + size
        if length > 1:
            return (vector
                [i:i+length] for i in range(start, end))
        else:
            return (vector[i] for i in range(start, end))
    except TypeError:
        return (itertools.repeat(vector, size))

def signal_series(data, f, size):
    """ Calculate a Series of signals from the data and function f.
            data is a sequence of triples that are arguments to make_gen """
    gs = [make_gen(size, *p) for p in data]
    xs = zip(*gs)
    return Series((f(*x) for x in xs))

#-------------------------------------------------------------------------------
#  VALUE
#  SIGNAL COMPUTATION FUNCIONS: f_sig
#-------------------------------------------------------------------------------

def infl_adj_yld_sig(qs, cpis, price):
    """ qs and cpis are vectors, price is a scalar.
            We try to use plurals for vectors, e.g. cpis, qs.
            Calulates the inflation adjusted qs yield. """
    return cpis[-1] * qs.div(cpis).sum() / (len(cpis) * price)

# def return_sig(returns1, returns2, b1):
#     """ 3 year annual returns (returns1) vs average total return (returns2)
#             over a period of 0 to b1 """
#     prod1 = returns1[:b1].prod()
#     prod2 = returns2[:b1].prod()
#     exp1 = returns1[-1] ** b1
#     exp2 = returns2[-1] ** b1
#     return (log(exp1 / prod1) - log(exp2 / prod2)) / (3 * b1)

def commodity_sig(prices, cpis):
    """ Inflation adjusted commodity price over it's avergage """
    return prices[-1] / cpis[-1] / (prices.div(cpis).sum() / len(cpis))

def backwardation_sig(rate, c1, c2):
    """ Annual 1 year backwardation """
    return rate / 100.0 - (c2 / c1 - 1)

def real_yield_sig(yld):
    return yld / 100.0

def muni_spread_sig(muni, tax_rate, tsy):
    return (muni / (1 - tax_rate) - tsy) / 100

def corp_spread_sig(corp):
    return corp / 10000.0

def em_spread_sig(em):
    return em

def corp_hy_sig(corp):
    return corp / 10000.0

def euro_hy_sig(hy):
    return hy / 10000.0

def ppp_sig(fx_rate, ppp):
    return fx_rate / (1/ppp)

VALUE_CALCULATORS = {
    "infl_adj_yld_sig" : infl_adj_yld_sig,    
    #"return_sig" : return_sig,
    "commodity_sig" : commodity_sig,
    "backwardation_sig" : backwardation_sig,
    "real_yield_sig" : real_yield_sig,
    "muni_spread_sig" : muni_spread_sig,
    "corp_spread_sig" : corp_spread_sig,
    "em_spread_sig" : em_spread_sig,
    "corp_hy_sig" : corp_hy_sig,
    "euro_hy_sig" : euro_hy_sig,
    "ppp_sig" : ppp_sig
}

#-------------------------------------------------------------------------------
#  MOMENTUM
#  SIGNAL COMPUTATION FUNCIONS: f_msig
#-------------------------------------------------------------------------------

def excess_return_msig(returns, cpis, risk_premium):
    n = len(cpis) - 1
    return (returns[n] / cpis[n] / returns[:n].div(cpis[:n]).sum()
                    * n - (1 + risk_premium))

def excess_return2_msig(returns1, returns2, cpis, risk_premium):
    n = len(cpis) - 1
    return ((returns1[n] / cpis[n] / returns1[:n].div(cpis[:n]).sum() * n)
            / (returns2[n] / cpis[n] / returns2[:n].div(cpis[:n]).sum() * n)
            - ( 1 + risk_premium))

def muni_spread_msig(muni, tax_rate, gov):
    n = len(muni) - 1
    return (muni[n] / (1 - tax_rate[n]) - gov[n] -
                    (muni[:n].mean() / (1 - tax_rate[:n].mean()) -
                    gov[:n].mean())) / 100.0

def corp_spread_msig(spreads):
    n = len(spreads) - 1
    return (spreads[n] - spreads[:n].mean()) / 10000.0

MOMENTUM_CALCULATORS = {
    "excess_return_msig" : excess_return_msig,
    "excess_return2_msig" : excess_return2_msig,
    "muni_spread_msig" : muni_spread_msig,
    "corp_spread_msig" : corp_spread_msig
}

#-------------------------------------------------------------------------------

def calc_signals(mkt_data, resource, calculators, drop_periods):
    def expand(v):
        return mkt_data[v] if v in mkt_data else v
    d = {}
    size = len(mkt_data[drop_periods:])
    for key, value in resource.items():
        data = [map(expand, item) for item in value[1:]]
        d[key] = signal_series(data, calculators[value[0]], size)
    return DataFrame(d)

def value_signals(mkt_data, resource, calculators, drop_periods):
    vs = calc_signals(mkt_data, resource, calculators, drop_periods)
    vs['EM_BOND'] = vs['EM_BOND'] + vs['US_TIP10']
    vs['EURO_HY'] = vs['EURO_HY'] + vs['FG_TIP10']
    vs['EURO_PPP'] = vs['EURO_PPP'] * ((1+vs['US_TIP10'])
                     / (1+ vs['FG_TIP10'])) ** 10
    vs['UK_PPP'] = vs['UK_PPP'] * ((1+vs['US_TIP10'])
                   / (1+ vs['UK_TIP10'])) ** 10
    return vs

def momentum_signals(mkt_data, resource, calculators, drop_periods):
    return calc_signals(mkt_data, resource, calculators, drop_periods)

def signals(mkt_data, json_data, drop_periods):
    """ mkt_data: a DataFrame keyed by excel style columns headings.
            json data: a dictionary of arguments for _signals functions.
            drop_periods: number of periods in to data to start calcs.
            return a pair of DataFrames value signals and momentum signals. """
    names = json_data["SIGNALS"]
    vs = value_signals(mkt_data, json_data["VALUE"], VALUE_CALCULATORS,
                       drop_periods)
    vs.index = mkt_data.index[drop_periods:]
    ms = momentum_signals(mkt_data, json_data["MOMENTUM"], MOMENTUM_CALCULATORS,
                          drop_periods)
    ms.index = mkt_data.index[drop_periods:]
    return vs.reindex(columns=names), ms.reindex(columns=names)

def main():
    mkt_data = pd.read_excel(
        'reports/data/Datain0426.xlsx', # changed from m3_csv
        index_col=0, usecols=range(sg.DATAIN_COLUMNS), dayfirst=False,
        parse_dates=True, na_values=['#VALUE!', '#NAME?', '#DIV/0!', '#N/A'],
        header=0, names=sg.data_columns())
    # mkt_data = pd.read_csv(
    #     '/Users/rdewey/Desktop/rb_vm_system/csv/Datain0426_m3.csv',
    #     index_col=0, usecols=range(DATAIN_COLUMNS), dayfirst=False,
    #     parse_dates=True, na_values=['#VALUE!', '#NAME?', '#DIV/0!', '#N/A'],
    #     header=0, names=data_columns())
    json_file = open('reports/data/param.json')
    json_data = json.load(json_file)
    (vs, ms) = signals(mkt_data, json_data, 121)

if __name__ == '__main__':
    main()