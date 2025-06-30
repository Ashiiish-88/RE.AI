from django import forms
from .models import ReturnRequest, ReturnAction

class ReturnActionForm(forms.ModelForm):
    final_action = forms.ModelChoiceField(
        queryset=ReturnAction.objects.all(),
        required=True,
        label="Override/Confirm Action"
    )

    class Meta:
        model = ReturnRequest
        fields = ['final_action']