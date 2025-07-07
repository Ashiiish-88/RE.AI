from django.views.generic import ListView, UpdateView, TemplateView
from django.utils import timezone
from django.core.cache import cache
from .models import ReturnRequest, ReturnAction, Product, Store, ReturnReason
from .forms import ReturnActionForm
from django.views.decorators.http import require_GET, require_POST
from django.shortcuts import render, redirect, get_object_or_404
from django.core.management import call_command
from returns.management.commands.automate_returns import mock_demand_model
from django.template.loader import render_to_string
from django.http import JsonResponse, Http404
from datetime import date


import os
import joblib
import datetime
import numpy as np
import random


# Cache key for models
MODEL_CACHE_KEY = 'ml_models_cache'
MODEL_CACHE_TIMEOUT = 3600  # 1 hour

def load_ml_models():
    """Load ML models with caching"""
    models = cache.get(MODEL_CACHE_KEY)
    
    if models is None:
        print("[CACHE] Loading ML models from disk...")
        BASE_DIR = os.path.dirname(__file__)
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
        
        try:
            models = {
                'demand_model': joblib.load(os.path.join(MODELS_DIR, 'demand_model.pkl')),
                'profit_model': joblib.load(os.path.join(MODELS_DIR, 'profit_model.pkl')),
                'feature_columns': joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl')),
                'label_encoders': joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl')),
                'scaler': joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
            }
            
            cache.set(MODEL_CACHE_KEY, models, MODEL_CACHE_TIMEOUT)
            print(f"[CACHE] Models loaded and cached for {MODEL_CACHE_TIMEOUT} seconds")
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            return None
    else:
        print("[CACHE] Using cached ML models")
    
    return models

def run_demand_and_profit_models(product_id, current_store_location, date_str):
    """Optimized ML model prediction with caching and proper input handling"""
    
    print(f"\n[🎯 PREDICTION] ========== PROCESSING REQUEST ==========")
    print(f"[📝 INPUT] Product ID: {product_id}")
    print(f"[📍 INPUT] Current Store: {current_store_location}")
    print(f"[📅 INPUT] Date: {date_str}")

    # Load models with caching
    models = load_ml_models()
    if models is None:
        return 0, current_store_location, -100, "Error: Models unavailable"

    # Extract models from cache
    demand_model = models['demand_model']
    profit_model = models['profit_model'] 
    feature_columns = models['feature_columns']
    label_encoders = models['label_encoders']
    scaler = models['scaler']
    
    print(f"[🧠 MODEL] Features required: {len(feature_columns)}")
    
    # Get product details for realistic pricing
    try:
        product = Product.objects.get(id=product_id)
        product_price = float(product.item_value) if product.item_value else 29.99
        print(f"[🛍️ PRODUCT] {product.name} (${product_price:.2f})")
    except Product.DoesNotExist:
        print(f"[❌ ERROR] Product ID {product_id} not found")
        return 0, current_store_location, -100, "Error: Product not found"

    # Parse date
    try:
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"[❌ ERROR] Invalid date format")
        return 0, current_store_location, -100, "Error: Invalid date"

    # Get valid stores for prediction
    all_stores = Store.objects.all().values_list('location', flat=True)
    valid_stores = [s for s in all_stores if s in label_encoders['store_location'].classes_]
    print(f"[🏪 STORES] Testing {len(valid_stores)} valid stores")

    best_store = current_store_location
    best_profit = -999999
    best_demand = 0
    results = []

    # Test each store
    for i, store_location in enumerate(valid_stores):
        try:
            print(f"\n[🔄 {i+1}/{len(valid_stores)}] Testing: {store_location}")
            
            # Build complete feature vector for DEMAND model (25 features)
            demand_features = {}
            
            # Core features
            demand_features['product_id'] = int(product_id)
            demand_features['product_price'] = product_price
            demand_features['cost_price'] = product_price * 0.75  # 75% of retail
            
            # Date features
            demand_features['hour'] = 14  # 2 PM
            demand_features['day_of_week'] = date_obj.weekday()
            demand_features['week'] = date_obj.isocalendar()[1]
            demand_features['month'] = date_obj.month
            demand_features['day_of_month'] = date_obj.day
            demand_features['quarter'] = (date_obj.month - 1) // 3 + 1
            demand_features['is_weekend'] = 1 if date_obj.weekday() >= 5 else 0
            demand_features['is_morning'] = 0
            demand_features['is_evening'] = 0
            
            # Lag features (historical demand)
            demand_features['demand_lag_1'] = 12
            demand_features['demand_lag_2'] = 11  
            demand_features['demand_lag_3'] = 13
            demand_features['demand_lag_7'] = 15
            demand_features['demand_lag_14'] = 18
            
            # Business features
            demand_features['unique_customers'] = 42
            demand_features['distance_miles'] = 38.5
            demand_features['logistics_cost_per_unit'] = 4.25
            
            # Categorical encodings
            product_name = f'Product_{product_id}'
            if product_name in label_encoders['product_name'].classes_:
                demand_features['product_name_encoded'] = label_encoders['product_name'].transform([product_name])[0]
            else:
                demand_features['product_name_encoded'] = 0  # Default encoding
                
            demand_features['aisle_encoded'] = label_encoders['aisle'].transform(['Aisle_1'])[0]
            demand_features['department_encoded'] = label_encoders['department'].transform(['Department_1'])[0]
            demand_features['store_location_encoded'] = label_encoders['store_location'].transform([store_location])[0]
            demand_features['distribution_center_encoded'] = label_encoders['distribution_center'].transform(['Atlanta_GA'])[0]
            
            # Create and scale feature vector for DEMAND prediction
            X_demand = np.array([[demand_features.get(col, 0) for col in feature_columns]])
            X_demand_scaled = scaler.transform(X_demand)
            
            # Predict DEMAND first
            predicted_demand = float(demand_model.predict(X_demand_scaled)[0])
            
            # Now build feature vector for PROFIT model (9 features)
            profit_features = {
                'predicted_demand': predicted_demand,
                'distance_miles': demand_features['distance_miles'],
                'product_id': demand_features['product_id'],
                'store_location_encoded': demand_features['store_location_encoded'],
                'distribution_center_encoded': demand_features['distribution_center_encoded'],
                'hour': demand_features['hour'],
                'day_of_week': demand_features['day_of_week'],
                'month': demand_features['month'],
                'is_weekend': demand_features['is_weekend']
            }
            
            # Profit model feature order
            profit_feature_order = ['predicted_demand', 'distance_miles', 'product_id', 'store_location_encoded',
                                  'distribution_center_encoded', 'hour', 'day_of_week', 'month', 'is_weekend']
            
            # Create feature vector for PROFIT prediction
            X_profit = np.array([[profit_features[col] for col in profit_feature_order]])
            
            # Predict PROFIT (no scaling needed for profit model)
            predicted_profit = float(profit_model.predict(X_profit)[0])
            
            results.append({
                'store': store_location,
                'demand': predicted_demand,
                'profit': predicted_profit
            })
            
            print(f"[📊 RESULT] Demand: {predicted_demand:.2f}, Profit: ${predicted_profit:.2f}")
            
            if predicted_profit > best_profit:
                best_profit = predicted_profit
                best_store = store_location
                best_demand = predicted_demand
                print(f"[🏆 NEW BEST] {store_location}")
                
        except Exception as e:
            print(f"[❌ ERROR] {store_location}: {str(e)}")
            continue
    
    # Final results
    results.sort(key=lambda x: x['profit'], reverse=True)
    top_stores = results[:3]  # Get top 3 recommendations
    
    print(f"\n[🎯 FINAL RESULTS]")
    print(f"[🏆 BEST] Store: {best_store}")
    print(f"[📈 DEMAND] {best_demand:.2f}")
    print(f"[💰 PROFIT] ${best_profit:.2f}")
    print(f"[📊 TOP 3] {[(r['store'], r['profit']) for r in top_stores]}")
    
    classification = "profitable_transfer" if best_profit > 0 else "keep_current"
    
    return best_demand, best_store, best_profit, classification, top_stores

def return_form(request):
    """Main form view with ML integration and caching"""
    products = Product.objects.all().order_by('name')
    reasons = ReturnReason.objects.all()
    stores = Store.objects.all().order_by('name')
    
    # Date constraints based on training data (2023-01-01 to 2023-12-30)
    min_date = "2023-01-01"
    max_date = "2023-12-30"
    default_date = "2023-06-15"  # Middle of the range for better predictions
    
    classification = None
    demand_result = None
    invoice_number = None

    if request.method == 'POST':
        # Get form data
        product_id = int(request.POST.get('product_id'))
        current_store_location = request.POST.get('location')
        date = request.POST.get('date')

        print(f"\n[FORM INPUT] ========== NEW RETURN REQUEST ==========")
        print(f"[FORM INPUT] Product ID: {product_id}")
        print(f"[FORM INPUT] Current Store Location: {current_store_location}")
        print(f"[FORM INPUT] Date: {date}")
        print(f"[FORM INPUT] ============================================")

        # Validate date is within training data range
        try:
            input_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            min_date_obj = datetime.datetime.strptime(min_date, "%Y-%m-%d").date()
            max_date_obj = datetime.datetime.strptime(max_date, "%Y-%m-%d").date()
            
            if input_date < min_date_obj or input_date > max_date_obj:
                print(f"[WARNING] Date {date} is outside training range {min_date} to {max_date}")
                # Still process but log the warning
        except ValueError:
            print(f"[ERROR] Invalid date format: {date}")

        # Run ML model to get predictions and classification
        demand, best_store, profit, classification, top_stores = run_demand_and_profit_models(
            product_id, current_store_location, date
        )
        
        print(f"[ML OUTPUT] Classification: {classification}")
        print(f"[ML OUTPUT] Demand: {demand}")
        print(f"[ML OUTPUT] Best Store: {best_store}")
        print(f"[ML OUTPUT] Profit: ${profit}")
        
        # Determine recommendation based on profit
        if profit > 0:
            recommendation = f"Transfer to {best_store}"
            recommendation_type = "transfer"
            print(f"[RECOMMENDATION] Transfer recommended - Profit: ${profit}")
        else:
            recommendation = "Keep in same store"
            recommendation_type = "keep"
            best_store = current_store_location
            print(f"[RECOMMENDATION] Keep in same store - Profit would be negative: ${profit}")
        
        demand_result = {
            'demand': round(demand, 2),
            'best_store': best_store,
            'current_store': current_store_location,
            'profit': round(profit, 2),
            'recommendation': recommendation,
            'recommendation_type': recommendation_type,
            'threshold_passed': demand > 4 and profit > 0,
            'classification': classification,
            'top_stores': top_stores
        }
        
        if demand_result['threshold_passed']:
            invoice_number = f"INV-{random.randint(10000,99999)}"
            print(f"[INVOICE] Generated invoice: {invoice_number}")

        return render(request, 'returns/form.html', {
            'products': products,
            'reasons': reasons,
            'stores': stores,
            'min_date': min_date,
            'max_date': max_date,
            'default_date': default_date,
            'classification': classification,
            'submitted': True,
            'demand_result': demand_result,
            'invoice_number': invoice_number,
        })

    return render(request, 'returns/form.html', {
        'products': products,
        'reasons': reasons,
        'stores': stores,
        'min_date': min_date,
        'max_date': max_date,
        'default_date': default_date,
    })

def staff_login(request):
    """Simple staff login view"""
    if request.method == 'POST':
        return redirect('returns:return_form')
    return render(request, 'returns/login.html')

class ReturnRequestListView(ListView):
    """Dashboard view for return requests"""
    model = ReturnRequest
    template_name = 'returns/dashboard.html'
    context_object_name = 'returns'

    def get_queryset(self):
        return ReturnRequest.objects.none()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['manual_returns'] = ReturnRequest.objects.filter(
            manual_review_flag=True, auto_processed=False
        ).select_related('product', 'reason', 'predicted_action', 'final_action', 'recommended_store').order_by('-created_at')
        context['automated_returns'] = ReturnRequest.objects.filter(
            auto_processed=True
        ).select_related('product', 'reason', 'predicted_action', 'final_action', 'recommended_store').order_by('-created_at')
        context['manual_review_count'] = context['manual_returns'].count()
        context['auto_processed_count'] = context['automated_returns'].count()
        return context

class ReturnRequestDetailView(UpdateView):
    """Detail view for individual return requests"""
    model = ReturnRequest
    form_class = ReturnActionForm
    template_name = 'returns/return_detail.html'
    context_object_name = 'return_request'

    def form_valid(self, form):
        obj = form.save(commit=False)
        obj.status = 'Processed'
        obj.final_action_by = self.request.user
        obj.final_action_at = timezone.now()
        obj.save()
        return redirect('returns:return_list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['images'] = self.object.images.all() if hasattr(self.object, 'images') else []
        return context

class ReturnStatsView(TemplateView):
    """Statistics view for return actions"""
    template_name = 'returns/return_stats.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        actions = ReturnAction.objects.all()
        stats = []
        for action in actions:
            count = ReturnRequest.objects.filter(final_action=action).count()
            stats.append({'action': action.name, 'count': count})
        context['stats'] = stats
        return context


@require_GET
def dashboard(request):
    pending_requests = ReturnRequest.objects.filter(status='pending').order_by('-created_at')
    manual_returns = ReturnRequest.objects.filter(
        manual_review_flag=True, auto_processed=False
    ).order_by('-created_at')
    automated_returns = ReturnRequest.objects.filter(
        auto_processed=True
    ).order_by('-created_at')
    manual_review_count = manual_returns.count()
    auto_processed_count = automated_returns.count()
    rejected_count = ReturnRequest.objects.filter(status='rejected').count()
    return render(request, 'returns/dashboard.html', {
        'pending_requests': pending_requests,
        'manual_review_count': manual_review_count,
        'auto_processed_count': auto_processed_count,
        'rejected_count': rejected_count,
    })

@require_GET
def automated_orders(request):
    automated_returns = ReturnRequest.objects.filter(auto_processed=True).order_by('-created_at')
    return render(request, 'returns/automated_orders.html', {
        'automated_returns': automated_returns,
    })

@require_GET
def manual_review(request):
    manual_returns = ReturnRequest.objects.filter(manual_review_flag=True).order_by('-created_at')
    return render(request, 'returns/manual_review.html', {
        'manual_returns': manual_returns,
    })

@require_POST
def process_all(request):
    call_command('automate_returns')
    return redirect('returns:dashboard')

@require_POST
def manual_review_decision(request, request_id):
    r = get_object_or_404(ReturnRequest, pk=request_id)
    final_action = request.POST.get('final_action')
    staff_note = request.POST.get('staff_note', '')
    r.final_action = ReturnAction.objects.filter(name__iexact=final_action).first()
    r.status = 'reviewed'
    r.manual_review_flag = False
    r.manual_review_reason = staff_note
    r.save()
    return redirect('returns:manual_review')

@require_GET
def invoice_modal(request, request_id):
    """
    Returns invoice details for a ReturnRequest as HTML (for modal display).
    Uses mock_demand_model to get profit and shipping location.
    """
    try:
        r = ReturnRequest.objects.select_related('product', 'store').get(pk=request_id)
    except ReturnRequest.DoesNotExist:
        raise Http404("ReturnRequest not found")

    # Generate a random invoice id for demonstration
    random_invoice_id = f"INV-{random.randint(100000, 999999)}"

    # Use mock_demand_model to get profit and best_store (shipping location)
    # Use today's date if created_at is missing
    created_date = r.created_at.date() if r.created_at else date.today()
    demand, best_store, profit = mock_demand_model(
        r.product.id,
        r.store.location if r.store else "Unknown",
        created_date
    )

    # Render a template fragment for the modal
    html = render_to_string('returns/invoice_modal.html', {
        'invoice_id': random_invoice_id,
        'product_name': r.product.name,
        'shipping_location': best_store,
        'profit': profit,
    }, request=request)

    return JsonResponse({'html': html})