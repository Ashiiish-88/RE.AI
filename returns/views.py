from django.views.generic import ListView, UpdateView, TemplateView
from django.shortcuts import redirect
from django.utils import timezone
from .models import ReturnRequest, ReturnAction
from .forms import ReturnActionForm

# Dashboard: List all pending returns
class ReturnRequestListView(ListView):
    model = ReturnRequest
    template_name = 'returns/dashboard.html'
    context_object_name = 'returns'

    def get_queryset(self):
        return ReturnRequest.objects.filter(status='Pending').select_related(
            'product', 'reason', 'predicted_action', 'final_action'
        ).order_by('-created_at')

# Detail: View and process a single return
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
        # If you have images related_name='images'
        context['images'] = self.object.images.all() if hasattr(self.object, 'images') else []
        return context

# Analytics: Show stats for actions taken
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