from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard', views.sales_cockpit, name='sales_cockpit'),
    path('dashboard/', views.sales_cockpit, name='sales_cockpit_slash'),
    path('wireframe/', views.wireframe, name='wireframe'),
    path('api/process/', views.process_file, name='process_file'),
    path('api/reapply-filters/', views.reapply_filters, name='reapply_filters'),
    path('api/custom-chart/', views.custom_chart, name='custom_chart'),
    path('api/filter-values/', views.filter_values, name='filter_values'),
    path('api/wireframe/generate/', views.wireframe_generate, name='wireframe_generate'),
    path('api/wireframe/add-artifact/', views.wireframe_add_artifact, name='wireframe_add_artifact'),
    path('api/dashboard/upload/', views.sales_cockpit_upload, name='sales_cockpit_upload'),
    path('api/dashboard/metadata/', views.sales_cockpit_metadata, name='sales_cockpit_metadata'),
    path('api/dashboard/column-mapping/', views.sales_cockpit_column_mapping, name='sales_cockpit_column_mapping'),
    path('api/dashboard/databricks-init/', views.sales_cockpit_databricks_init, name='sales_cockpit_databricks_init'),
    path('api/dashboard/data/', views.sales_cockpit_data, name='sales_cockpit_data'),
    path('api/dashboard/custom-chart/', views.sales_cockpit_custom_chart, name='sales_cockpit_custom_chart'),
    path('api/dashboard/save/', views.dashboard_save, name='dashboard_save'),
    path('api/dashboard/history/', views.dashboard_history, name='dashboard_history'),
    path('api/dashboard/history/<uuid:dashboard_id>/', views.dashboard_history_detail, name='dashboard_history_detail'),
]
