from django.urls import path
from . import views

app_name = 'returns'

urlpatterns = [
    path('', views.ReturnRequestListView.as_view(), name='return_list'),
    path('<int:pk>/', views.ReturnRequestDetailView.as_view(), name='return_detail'),
    path('stats/', views.ReturnStatsView.as_view(), name='return_stats'),
]