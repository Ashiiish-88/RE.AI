from django.urls import path
from . import views

app_name = 'returns'

urlpatterns = [
    path('', views.ReturnRequestListView.as_view(), name='return_list'),
    path('<int:pk>/', views.ReturnRequestDetailView.as_view(), name='return_detail'),
    path('stats/', views.ReturnStatsView.as_view(), name='return_stats'),
    path('login/', views.staff_login, name='staff_login'),  # Updated URL pattern
    path('return-form/', views.return_form, name='return_form'),
]