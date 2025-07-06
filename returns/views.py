from django.views.generic import ListView, UpdateView, TemplateView
from django.shortcuts import redirect, render
from django.utils import timezone
from .models import ReturnRequest, ReturnAction, Product, Store, ReturnReason
from .forms import ReturnActionForm

import os
import joblib
import datetime
import numpy as np
import random

# --- Manual classifier mapping for reasons ---
REASON_CLASSIFICATION = {
    1: 'recycle',   # Defective Item
    2: 'restock',   # Wrong Size
    3: 'restock',   # Unwanted
    4: 'restock',   # Late Delivery
    5: 'refurbish', # Damaged Packaging
    6: 'recycle',   # Defective
    7: 'restock',   # Wrong Item
    8: 'restock',   # Changed Mind
    9: 'refurbish', # Damaged
}

def run_demand_and_profit_models(product_id, store_location, date_str):
    import os
    import joblib
    import datetime
    import numpy as np

    BASE_DIR = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    print(f"[INFO] Loading models from: {MODELS_DIR}")

    demand_model = joblib.load(os.path.join(MODELS_DIR, 'demand_model.pkl'))
    profit_model = joblib.load(os.path.join(MODELS_DIR, 'profit_model.pkl'))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

    print(f"[INFO] feature_columns: {feature_columns}")
    print(f"[INFO] label_encoders keys: {list(label_encoders.keys())}")

    features = {}
    for col in feature_columns:
        if col == 'product_id':
            features[col] = product_id
        elif col == 'store_location_encoded':
            # Use the correct encoder key
            features[col] = label_encoders['store_location'].transform([store_location])[0]
        elif col == 'product_name_encoded':
            product_name = 'Product_{}'.format(product_id)
            if product_name not in label_encoders['product_name'].classes_:
                print(f"[WARNING] {product_name} not in label encoder classes, using default.")
            product_name = label_encoders['product_name'].classes_[0]
            features[col] = label_encoders['product_name'].transform([product_name])[0]
        elif col == 'aisle_encoded':
            features[col] = label_encoders['aisle'].transform(['Aisle_1'])[0]  # Replace as needed
        elif col == 'department_encoded':
            features[col] = label_encoders['department'].transform(['Department_1'])[0]  # Replace as needed
        elif col == 'distribution_center_encoded':
            features[col] = label_encoders['distribution_center'].transform(['DC_1'])[0]  # Replace as needed
        elif col == 'date':
            features[col] = datetime.datetime.strptime(date_str, "%Y-%m-%d").toordinal()
        elif col == 'hour':
            features[col] = 12  # or extract from date/time if you have it
        elif col == 'day_of_week':
            features[col] = 1   # or extract from date
        elif col == 'month':
            features[col] = 7   # or extract from date
        elif col == 'is_weekend':
            features[col] = 0   # or calculate from date
        elif col == 'product_price':
            features[col] = 100 # or get from your Product model
        elif col == 'cost_price':
            features[col] = 80  # or get from your Product model
        else:
            features[col] = 0   # default for any other feature

    print(f"[DEBUG] Features after encoding: {features}")

    X = np.array([[features[col] for col in feature_columns]])
    print(f"[DEBUG] X shape: {X.shape}, X: {X}")

    X_scaled = scaler.transform(X)
    print(f"[DEBUG] X_scaled: {X_scaled}")

    demand = float(demand_model.predict(X_scaled)[0])
    profit = float(profit_model.predict(X_scaled)[0])
    best_store = store_location

    print(f"[RESULT] Demand: {demand}, Profit: {profit}, Best Store: {best_store}")

    return demand, best_store, profit

def return_form(request):
    products = Product.objects.all()
    reasons = ReturnReason.objects.all()
    stores = Store.objects.all()
    today = timezone.now().date().isoformat()
    classification = None
    demand_result = None
    invoice_number = None

    if request.method == 'POST':
        reason_id = int(request.POST.get('reason'))
        classification = REASON_CLASSIFICATION.get(reason_id, 'refurbish')
        product_id = int(request.POST.get('product_id'))
        store_location = request.POST.get('location')
        date = request.POST.get('date')
    
        # Defensive: ensure store_location is in the encoder's classes
        if store_location not in label_encoders['store_location'].classes_:
            print(f"[WARNING] {store_location} not in label encoder classes, using default.")
            store_location = label_encoders['store_location'].classes_[0]
        print(label_encoders['store_location'].classes_)


        if classification == 'restock' and store:
            demand, best_store, profit = run_demand_and_profit_models(product_id, store_location, date)
            demand_result = {
                'demand': round(demand, 2),
                'best_store': best_store,
                'profit': round(profit, 2),
                'threshold_passed': demand > 4 and profit > 0
            }
            if demand_result['threshold_passed']:
                invoice_number = f"INV-{random.randint(10000,99999)}"

        return render(request, 'returns/form.html', {
            'products': products,
            'reasons': reasons,
            'stores': stores,
            'today': today,
            'classification': classification,
            'submitted': True,
            'demand_result': demand_result,
            'invoice_number': invoice_number,
        })

    return render(request, 'returns/form.html', {
        'products': products,
        'reasons': reasons,
        'stores': stores,
        'today': today,
    })

def staff_login(request):
    if request.method == 'POST':
        return redirect('returns:return_form')
    return render(request, 'returns/login.html')

class ReturnRequestListView(ListView):
    model = ReturnRequest
    template_name = 'returns/dashboard.html'
    context_object_name = 'returns'  # Not used, but required by ListView

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