from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('time_series', views.timeSeries, name='time_series'),
    path('rb_aqr_macro', views.rb_aqr_macro, name='rb_aqr_macro'),
    path('dashboard', views.dashboard, name='dashboard'),
    path('drawdown', views.drawdown, name='drawdown'),
]