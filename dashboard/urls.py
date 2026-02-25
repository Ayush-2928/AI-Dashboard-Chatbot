from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/process/', views.process_file, name='process_file'),
    path('api/reapply-filters/', views.reapply_filters, name='reapply_filters'),
    path('api/custom-chart/', views.custom_chart, name='custom_chart'),
    path('api/filter-values/', views.filter_values, name='filter_values'),
]
