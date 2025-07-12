from django.contrib import admin

from .models import (
    Product,
    Store,
    ReturnAction,
    ReturnReason,
    ReturnRequest,
)

admin.site.register(Product)
admin.site.register(Store)
admin.site.register(ReturnAction)
admin.site.register(ReturnReason)
admin.site.register(ReturnRequest)