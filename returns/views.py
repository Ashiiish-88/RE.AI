from django.views.generic import ListView, UpdateView, TemplateView
from django.shortcuts import redirect
from django.utils import timezone
from .models import ReturnRequest, ReturnAction
from .forms import ReturnActionForm

class ReturnRequestListView(ListView):
    model = ReturnRequest
    template_name = 'returns/dashboard.html'
    context_object_name = 'returns'  # Not used, but required by ListView

    def get_queryset(self):
        # Not used, as we pass custom querysets in get_context_data
        return ReturnRequest.objects.none()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Manual review: needs staff attention
        context['manual_returns'] = ReturnRequest.objects.filter(
            manual_review_flag=True, auto_processed=False
        ).select_related('product', 'reason', 'predicted_action', 'final_action', 'recommended_store').order_by('-created_at')
        # Automated: handled by automation
        context['automated_returns'] = ReturnRequest.objects.filter(
            auto_processed=True
        ).select_related('product', 'reason', 'predicted_action', 'final_action', 'recommended_store').order_by('-created_at')
        # For analytics cards (optional)
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