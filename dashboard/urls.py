from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/process/', views.process_file, name='process_file'),
    path('api/custom-chart/', views.custom_chart, name='custom_chart'),
]
