from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/process/', views.process_file, name='process_file'),
    path('api/export-powerbi/', views.export_powerbi, name='export_powerbi'),  # ADD THIS
]