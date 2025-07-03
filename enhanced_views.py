"""
Django Views and APIs for Enhanced Instacart Dataset
==================================================

This file contains Django views and REST API endpoints for the enhanced
Instacart dataset with demand forecasting and inventory optimization.
"""

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q, Avg, Sum, Count, Max, Min
from django.utils import timezone
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Import our enhanced models
from .enhanced_models import (
    Product, DemandForecast, InventoryOptimization, 
    MLModel, DemandAlert, DemandTrend
)

# =============================================================================
# DASHBOARD VIEWS
# =============================================================================

def dashboard_view(request):
    """
    Main dashboard view for demand forecasting and inventory optimization
    """
    # Get summary statistics
    total_products = Product.objects.count()
    high_demand_products = Product.objects.filter(demand_category='Very High').count()
    active_alerts = DemandAlert.objects.filter(is_active=True).count()
    
    # Get recent forecasts
    recent_forecasts = DemandForecast.objects.select_related('product').order_by('-created_at')[:10]
    
    # Get top products by demand score
    top_products = Product.objects.order_by('-demand_score')[:10]
    
    # Get inventory optimization recommendations
    urgent_reorders = InventoryOptimization.objects.filter(
        recommended_action='urgent'
    ).select_related('product').order_by('-priority_level')[:5]
    
    context = {
        'total_products': total_products,
        'high_demand_products': high_demand_products,
        'active_alerts': active_alerts,
        'recent_forecasts': recent_forecasts,
        'top_products': top_products,
        'urgent_reorders': urgent_reorders,
    }
    
    return render(request, 'returns/dashboard.html', context)

def product_detail_view(request, product_id):
    """
    Detailed view for a specific product with demand analysis
    """
    product = get_object_or_404(Product, product_id=product_id)
    
    # Get demand forecasts for the last 30 days
    thirty_days_ago = timezone.now() - timedelta(days=30)
    forecasts = DemandForecast.objects.filter(
        product=product,
        date__gte=thirty_days_ago
    ).order_by('date')
    
    # Get inventory optimization data
    inventory_data = InventoryOptimization.objects.filter(
        product=product
    ).order_by('-date')[:30]
    
    # Get recent alerts
    alerts = DemandAlert.objects.filter(
        product=product,
        is_active=True
    ).order_by('-created_at')[:10]
    
    # Get demand trends
    trends = DemandTrend.objects.filter(
        product=product
    ).order_by('-date')[:10]
    
    context = {
        'product': product,
        'forecasts': forecasts,
        'inventory_data': inventory_data,
        'alerts': alerts,
        'trends': trends,
    }
    
    return render(request, 'returns/product_detail.html', context)

def analytics_view(request):
    """
    Analytics and reporting view
    """
    # Get demand analytics by department
    department_stats = Product.objects.values('department').annotate(
        total_products=Count('product_id'),
        avg_demand_score=Avg('demand_score'),
        total_orders=Sum('total_orders')
    ).order_by('-total_orders')
    
    # Get weekly demand trends
    weekly_trends = DemandForecast.objects.values('week').annotate(
        total_demand=Sum('actual_demand'),
        avg_demand=Avg('actual_demand')
    ).order_by('week')
    
    # Get model performance metrics
    model_performance = MLModel.objects.filter(status='active').values(
        'model_name', 'model_type', 'mae', 'mse', 'r2_score'
    )
    
    context = {
        'department_stats': department_stats,
        'weekly_trends': weekly_trends,
        'model_performance': model_performance,
    }
    
    return render(request, 'returns/analytics.html', context)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@csrf_exempt
@require_http_methods(["GET"])
def api_products(request):
    """
    API endpoint to get products with filtering and pagination
    """
    # Get query parameters
    page = int(request.GET.get('page', 1))
    page_size = int(request.GET.get('page_size', 20))
    category = request.GET.get('category', '')
    department = request.GET.get('department', '')
    search = request.GET.get('search', '')
    
    # Start with all products
    products = Product.objects.all()
    
    # Apply filters
    if category:
        products = products.filter(demand_category=category)
    
    if department:
        products = products.filter(department__icontains=department)
    
    if search:
        products = products.filter(
            Q(product_name__icontains=search) |
            Q(aisle__icontains=search)
        )
    
    # Order by demand score
    products = products.order_by('-demand_score')
    
    # Paginate
    paginator = Paginator(products, page_size)
    page_obj = paginator.get_page(page)
    
    # Serialize data
    products_data = []
    for product in page_obj:
        products_data.append({
            'product_id': product.product_id,
            'product_name': product.product_name,
            'aisle': product.aisle,
            'department': product.department,
            'demand_score': product.demand_score,
            'demand_category': product.demand_category,
            'total_orders': product.total_orders,
            'avg_reorder_rate': product.avg_reorder_rate,
            'unique_customers': product.unique_customers,
        })
    
    return JsonResponse({
        'products': products_data,
        'pagination': {
            'page': page,
            'page_size': page_size,
            'total_pages': paginator.num_pages,
            'total_items': paginator.count,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
        }
    })

@csrf_exempt
@require_http_methods(["GET"])
def api_demand_forecast(request, product_id):
    """
    API endpoint to get demand forecast for a specific product
    """
    product = get_object_or_404(Product, product_id=product_id)
    
    # Get date range from query parameters
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')
    
    # Default to last 30 days if no dates provided
    if not start_date:
        start_date = timezone.now() - timedelta(days=30)
    else:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    if not end_date:
        end_date = timezone.now()
    else:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Get forecasts
    forecasts = DemandForecast.objects.filter(
        product=product,
        date__gte=start_date,
        date__lte=end_date
    ).order_by('date')
    
    # Serialize data
    forecast_data = []
    for forecast in forecasts:
        forecast_data.append({
            'date': forecast.date.strftime('%Y-%m-%d %H:%M'),
            'actual_demand': forecast.actual_demand,
            'predicted_demand': forecast.predicted_demand,
            'confidence_interval_lower': forecast.confidence_interval_lower,
            'confidence_interval_upper': forecast.confidence_interval_upper,
            'hour': forecast.hour,
            'day_of_week': forecast.day_of_week,
            'is_weekend': forecast.is_weekend,
        })
    
    return JsonResponse({
        'product': {
            'product_id': product.product_id,
            'product_name': product.product_name,
            'department': product.department,
        },
        'forecasts': forecast_data,
        'date_range': {
            'start_date': start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date.strftime('%Y-%m-%d'),
        }
    })

@csrf_exempt
@require_http_methods(["GET"])
def api_inventory_optimization(request):
    """
    API endpoint to get inventory optimization recommendations
    """
    # Get query parameters
    action = request.GET.get('action', '')
    priority = request.GET.get('priority', '')
    page = int(request.GET.get('page', 1))
    page_size = int(request.GET.get('page_size', 20))
    
    # Start with all optimization records
    optimizations = InventoryOptimization.objects.select_related('product').all()
    
    # Apply filters
    if action:
        optimizations = optimizations.filter(recommended_action=action)
    
    if priority:
        optimizations = optimizations.filter(priority_level=int(priority))
    
    # Order by priority and date
    optimizations = optimizations.order_by('-priority_level', '-date')
    
    # Paginate
    paginator = Paginator(optimizations, page_size)
    page_obj = paginator.get_page(page)
    
    # Serialize data
    optimization_data = []
    for opt in page_obj:
        optimization_data.append({
            'product_id': opt.product.product_id,
            'product_name': opt.product.product_name,
            'date': opt.date.strftime('%Y-%m-%d'),
            'current_stock': opt.current_stock,
            'reorder_point': opt.reorder_point,
            'optimal_stock_level': opt.optimal_stock_level,
            'predicted_daily_demand': opt.predicted_daily_demand,
            'recommended_action': opt.recommended_action,
            'recommended_quantity': opt.recommended_quantity,
            'priority_level': opt.priority_level,
            'total_cost': opt.total_cost,
        })
    
    return JsonResponse({
        'optimizations': optimization_data,
        'pagination': {
            'page': page,
            'page_size': page_size,
            'total_pages': paginator.num_pages,
            'total_items': paginator.count,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous(),
        }
    })

@csrf_exempt
@require_http_methods(["GET"])
def api_demand_alerts(request):
    """
    API endpoint to get demand alerts
    """
    # Get query parameters
    severity = request.GET.get('severity', '')
    alert_type = request.GET.get('alert_type', '')
    is_active = request.GET.get('is_active', 'true').lower() == 'true'
    
    # Get alerts
    alerts = DemandAlert.objects.select_related('product').filter(
        is_active=is_active
    )
    
    # Apply filters
    if severity:
        alerts = alerts.filter(severity=severity)
    
    if alert_type:
        alerts = alerts.filter(alert_type=alert_type)
    
    # Order by severity and creation date
    severity_order = ['critical', 'high', 'medium', 'low']
    alerts = alerts.order_by('-created_at')
    
    # Serialize data
    alert_data = []
    for alert in alerts:
        alert_data.append({
            'id': alert.id,
            'product_id': alert.product.product_id,
            'product_name': alert.product.product_name,
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'message': alert.message,
            'current_value': alert.current_value,
            'threshold_value': alert.threshold_value,
            'expected_value': alert.expected_value,
            'is_acknowledged': alert.is_acknowledged,
            'created_at': alert.created_at.strftime('%Y-%m-%d %H:%M'),
        })
    
    return JsonResponse({
        'alerts': alert_data,
        'summary': {
            'total_alerts': len(alert_data),
            'critical': len([a for a in alert_data if a['severity'] == 'critical']),
            'high': len([a for a in alert_data if a['severity'] == 'high']),
            'medium': len([a for a in alert_data if a['severity'] == 'medium']),
            'low': len([a for a in alert_data if a['severity'] == 'low']),
        }
    })

@csrf_exempt
@require_http_methods(["POST"])
def api_acknowledge_alert(request, alert_id):
    """
    API endpoint to acknowledge an alert
    """
    try:
        alert = get_object_or_404(DemandAlert, id=alert_id)
        
        # Get request data
        data = json.loads(request.body)
        acknowledged_by = data.get('acknowledged_by', 'System')
        
        # Update alert
        alert.is_acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = timezone.now()
        alert.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Alert acknowledged successfully',
            'alert_id': alert_id,
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
        }, status=400)

@csrf_exempt
@require_http_methods(["GET"])
def api_analytics_summary(request):
    """
    API endpoint to get analytics summary
    """
    # Get date range
    days = int(request.GET.get('days', 30))
    start_date = timezone.now() - timedelta(days=days)
    
    # Overall statistics
    total_products = Product.objects.count()
    total_forecasts = DemandForecast.objects.filter(date__gte=start_date).count()
    
    # Demand categories distribution
    demand_categories = Product.objects.values('demand_category').annotate(
        count=Count('product_id')
    )
    
    # Department performance
    department_stats = Product.objects.values('department').annotate(
        product_count=Count('product_id'),
        avg_demand_score=Avg('demand_score'),
        total_orders=Sum('total_orders')
    ).order_by('-total_orders')[:10]
    
    # Alert statistics
    alert_stats = DemandAlert.objects.filter(
        created_at__gte=start_date,
        is_active=True
    ).values('severity').annotate(count=Count('id'))
    
    # Model performance
    active_models = MLModel.objects.filter(status='active').count()
    
    # Recent trends
    recent_trends = DemandTrend.objects.filter(
        date__gte=start_date
    ).values('trend_direction').annotate(count=Count('id'))
    
    return JsonResponse({
        'summary': {
            'total_products': total_products,
            'total_forecasts': total_forecasts,
            'active_models': active_models,
            'date_range_days': days,
        },
        'demand_categories': list(demand_categories),
        'department_stats': list(department_stats),
        'alert_stats': list(alert_stats),
        'trend_stats': list(recent_trends),
    })

@csrf_exempt
@require_http_methods(["GET"])
def api_export_data(request):
    """
    API endpoint to export data in CSV format
    """
    data_type = request.GET.get('type', 'products')
    
    if data_type == 'products':
        # Export products data
        products = Product.objects.all()
        data = []
        
        for product in products:
            data.append({
                'product_id': product.product_id,
                'product_name': product.product_name,
                'aisle': product.aisle,
                'department': product.department,
                'demand_score': product.demand_score,
                'demand_category': product.demand_category,
                'total_orders': product.total_orders,
                'avg_reorder_rate': product.avg_reorder_rate,
                'unique_customers': product.unique_customers,
            })
        
        # Create CSV response
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
        
        response = HttpResponse(csv_data, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="products_export.csv"'
        return response
    
    elif data_type == 'forecasts':
        # Export forecast data
        forecasts = DemandForecast.objects.select_related('product').all()
        data = []
        
        for forecast in forecasts:
            data.append({
                'product_id': forecast.product.product_id,
                'product_name': forecast.product.product_name,
                'date': forecast.date.strftime('%Y-%m-%d %H:%M'),
                'actual_demand': forecast.actual_demand,
                'predicted_demand': forecast.predicted_demand,
                'hour': forecast.hour,
                'day_of_week': forecast.day_of_week,
                'is_weekend': forecast.is_weekend,
            })
        
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
        
        response = HttpResponse(csv_data, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="forecasts_export.csv"'
        return response
    
    else:
        return JsonResponse({'error': 'Invalid data type'}, status=400)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_product_demand_patterns(product_id):
    """
    Utility function to get demand patterns for a specific product
    """
    forecasts = DemandForecast.objects.filter(product_id=product_id)
    
    # Daily patterns
    daily_patterns = forecasts.values('day_of_week').annotate(
        avg_demand=Avg('actual_demand')
    ).order_by('day_of_week')
    
    # Hourly patterns
    hourly_patterns = forecasts.values('hour').annotate(
        avg_demand=Avg('actual_demand')
    ).order_by('hour')
    
    # Monthly patterns
    monthly_patterns = forecasts.values('month').annotate(
        avg_demand=Avg('actual_demand')
    ).order_by('month')
    
    return {
        'daily_patterns': list(daily_patterns),
        'hourly_patterns': list(hourly_patterns),
        'monthly_patterns': list(monthly_patterns),
    }

def calculate_forecast_accuracy(model_id=None):
    """
    Utility function to calculate forecast accuracy
    """
    forecasts = DemandForecast.objects.all()
    
    if model_id:
        forecasts = forecasts.filter(model_version=model_id)
    
    # Calculate metrics
    actual_values = [f.actual_demand for f in forecasts if f.predicted_demand is not None]
    predicted_values = [f.predicted_demand for f in forecasts if f.predicted_demand is not None]
    
    if not actual_values:
        return None
    
    # Calculate MAE, MSE, MAPE
    mae = np.mean(np.abs(np.array(actual_values) - np.array(predicted_values)))
    mse = np.mean((np.array(actual_values) - np.array(predicted_values)) ** 2)
    mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mape': mape,
        'sample_size': len(actual_values),
    }
