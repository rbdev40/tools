from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import pprint
import os

def drawdown(request):
    import reports.modules.correlport_v1 as correlport_v1
    
    variable = request.GET.get('variable')
    
    if not (variable is None):
        correlport_v1.generateImage()
    
    var = correlport_v1.getVar()
    drawdown = correlport_v1.getDrawdown()
    context = {'var': var, 'drawdown': drawdown, 'variable': variable}
    return render(request, 'reports/drawdown.html', context)

def index(request):
    context = {}
    return render(request, 'reports/index.html', context)
    
def dashboard(request):
    
    import reports.modules.hashtool as hashtool
    
    data_file_path = 'reports/data/port2.csv'
    full_data_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_file_path)
    
    data = pd.read_csv(full_data_file_path)
    
    data_file_md5 = hashtool.md5file(full_data_file_path)

    #new data pull
    lab = data.iloc[7:11,0]
    weights = data.iloc[7:11,1]
    risk = data.iloc[0:5,1]
    values2 = data.iloc[0:8,1]

    labels = lab
    values = weights
    colors = [ "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA"]
    
    context = {'set': list(zip(values, labels, colors)), 'values2': values2, 'risk': risk, 'labels': labels}
    return render(request, 'reports/dashboard.html', context)

def timeSeries(request):

    import reports.modules.sim_func_graph2 as rd
    
    drift = request.GET.get('drift')           #1
    mu = request.GET.get('mean_of_drift')      #2
    sigma = request.GET.get('volatility')      #3
    lamda = request.GET.get('jump_parameter')  #4
    steps = request.GET.get('time_steps')       #5
    proc = request.GET.get('define_process')   #6
    showchart = not (drift is None or drift == "")

    if mu is not None:
        series = rd.sim(mu,sigma,lamda,steps,proc)
        values = series.iloc[:,0]
        steps = int(steps)

        c = []
        for i in range(0,steps):
            if (i % 12 == 0):
                c.append(i) ## This could be changed to a different arrary with months or something
            else:
                c.append("")
        labels = c
    else:
        values = None
        labels = None
    
    context = {'values': values, 'labels': labels, 'showchart': showchart, 'mean_of_drift': mu, 'volatility': sigma, 'jump_parameter': lamda, 'time_steps': steps}
    
    return render(request, 'reports/time_series.html', context)
    
def rb_aqr_macro(request):
    
    import reports.modules.rb_aqr_macro as rb
    
    data = rb.getData()
    
    tables = [];
    
    for d in data:
        tables.append(d[0].to_html(float_format=lambda x: '%4.3f' % (x), classes="table table-striped table-hover table-condensed"))
    
    context = {'tables': tables}
    return render(request, 'reports/rb_aqr_macro.html', context)



def rb_vm_system(request):
    
    import reports.modules.weight_gen as wg
    
    data = wg.getData()
    
    tables = [];
    
    for d in data:
        tables.append(d[0].to_html(float_format=lambda x: '%4.3f' % (x), classes="table table-striped table-hover table-condensed"))
    
    context = {'tables': tables}
    return render(request, 'reports/rb_vm_system.html', context)


def vol_regime(request):
    import reports.modules.vol_regime as vol_regime
    var, drawdown = vol_regime.generateImage()
    #var = vol_regime.getStat1(last_p)
    #drawdown = vol_regime.getStat2()
    context = {'var': var, 'drawdown': drawdown}
    return render(request, 'reports/volatility.html', context)


def amzn(request):
    import reports.modules.amzn as amzn
    amzn.generateImage()
    #var = amzn.getStat1()
    #drawdown = amzn.getStat2()
    #context = {'var': var, 'drawdown': drawdown}
    return render(request, 'reports/amzn.html', context)

