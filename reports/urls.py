from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('time_series', views.timeSeries, name='time_series'),
    path('rb_aqr_macro', views.rb_aqr_macro, name='rb_aqr_macro'),
    path('dashboard', views.dashboard, name='dashboard'),
    path('drawdown', views.drawdown, name='drawdown'),
    path('rb_vm_system', views.rb_vm_system, name='rb_vm_system'),
    path('vol_regime', views.vol_regime, name='vol_regime'),
    path('amzn', views.amzn, name='amzn'),
]


# path ; what function inside the views ; naming the path for later reference