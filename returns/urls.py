from django.urls import path
from . import views

app_name = 'returns'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),  # Dashboard: pending requests + Process All
    path('automated/', views.automated_orders, name='automated_orders'),  # Automated Orders page
    path('manual/', views.manual_review, name='manual_review'),  # Manual Review page
    path('process_all/', views.process_all, name='process_all'),  # Triggers automation script
    path('manual_review_decision/<int:request_id>/', views.manual_review_decision, name='manual_review_decision'),
    path('invoice-modal/<int:request_id>/', views.invoice_modal, name='invoice_modal'),
    # Existing views
    path('<int:pk>/', views.ReturnRequestDetailView.as_view(), name='return_detail'),
    path('stats/', views.ReturnStatsView.as_view(), name='return_stats'),
    path('login/', views.staff_login, name='staff_login'),
    path('return-form/', views.return_form, name='return_form'),
]