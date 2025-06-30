from django.contrib import admin

# Register your models here.
from .models import (
    Category, Product, Store, ReturnAction, ReturnReason,
    Customer, ProductInstance, ReturnRequest, ReturnImage,
    Attachment, ReturnStatusHistory
)

admin.site.register(Category)
admin.site.register(Product)
admin.site.register(Store)
admin.site.register(ReturnAction)
admin.site.register(ReturnReason)
admin.site.register(Customer)
admin.site.register(ProductInstance)
admin.site.register(ReturnRequest)
admin.site.register(ReturnImage)
admin.site.register(Attachment)
admin.site.register(ReturnStatusHistory)