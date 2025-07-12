from django.views.generic import ListView, UpdateView, TemplateView
from django.utils import timezone
from .models import ReturnRequest, ReturnAction, Product, Store, ReturnReason
from .forms import ReturnActionForm
from django.views.decorators.http import require_GET, require_POST
from django.shortcuts import render, redirect, get_object_or_404
from django.core.management import call_command
from django.template.loader import render_to_string
from django.http import JsonResponse, Http404
from datetime import date
import random

# Import your ML utils
from returns.ml_utils import predict_demand

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
    status = request.GET.get('status', 'all')
    qs = ReturnRequest.objects.filter(manual_review_reason__isnull=False)
    if status == 'pending':
        qs = qs.filter(status='pending')
    elif status == 'reviewed':
        qs = qs.filter(status='reviewed')
    manual_returns = qs.order_by('-created_at')
    return render(request, 'returns/manual_review.html', {
        'manual_returns': manual_returns,
        'current_status': status,
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
    Uses real demand model to get profit and shipping location.
    """
    try:
        r = ReturnRequest.objects.select_related('product', 'store').get(pk=request_id)
    except ReturnRequest.DoesNotExist:
        raise Http404("ReturnRequest not found")

    # Generate a random invoice id for demonstration
    random_invoice_id = f"INV-{random.randint(100000, 999999)}"

    # Use demand model to get demand score and profit
    from django.db import connection
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT * FROM demands
            WHERE product_id = %s AND store_id = %s
            LIMIT 1
        """, [r.product_id, r.store_id])
        row = cursor.fetchone()
        columns = [col[0] for col in cursor.description]
        if row:
            demand_row = dict(zip(columns, row))
        else:
            demand_row = None

    if demand_row:
        demand_score = predict_demand(r, demand_row)
        profit = float(demand_row.get('profit_margin', 0))
        shipping_location = r.store.location if r.store else "Unknown"
    else:
        demand_score = 0
        profit = 0
        shipping_location = r.store.location if r.store else "Unknown"

    html = render_to_string('returns/invoice_modal.html', {
        'invoice_id': random_invoice_id,
        'product_name': r.product.product_name,
        'shipping_location': shipping_location,
        'profit': profit,
        'demand_score': demand_score,
    }, request=request)

    return JsonResponse({'html': html})

# Optionally, keep your class-based views if you use them elsewhere:

class ReturnRequestListView(ListView):
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
    


def staff_login(request):
    """Simple staff login view (placeholder)"""
    if request.method == 'POST':
        # You can add real authentication here if needed
        return redirect('returns:dashboard')
    return render(request, 'returns/login.html')

from django.shortcuts import render
from .models import Product, ReturnReason, Store
import datetime
import random

def return_form(request):
    """Main form view for submitting a return request."""
    products = Product.objects.all().order_by('name')
    reasons = ReturnReason.objects.all()
    stores = Store.objects.all().order_by('name')

    # Date constraints (adjust as needed)
    min_date = "2023-01-01"
    max_date = "2023-12-30"
    default_date = "2023-06-15"

    if request.method == 'POST':
        # Get form data
        product_id = int(request.POST.get('product_id'))
        store_id = int(request.POST.get('store_id'))
        reason_id = int(request.POST.get('reason_id'))
        date_str = request.POST.get('date')

        # Create a new ReturnRequest (status = pending)
        from .models import ReturnRequest
        r = ReturnRequest.objects.create(
            product_id=product_id,
            store_id=store_id,
            reason_id=reason_id,
            created_at=date_str,
            status='pending'
        )
        # Optionally, add more fields as needed

        # Redirect to dashboard or show a success message
        return render(request, 'returns/form.html', {
            'products': products,
            'reasons': reasons,
            'stores': stores,
            'min_date': min_date,
            'max_date': max_date,
            'default_date': default_date,
            'submitted': True,
            'return_request': r,
        })

    return render(request, 'returns/form.html', {
        'products': products,
        'reasons': reasons,
        'stores': stores,
        'min_date': min_date,
        'max_date': max_date,
        'default_date': default_date,
    })