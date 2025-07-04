from django.db import models
from django.contrib.auth import get_user_model

class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

class Product(models.Model):
    name = models.CharField(max_length=255)
    sku = models.CharField(max_length=100, unique=True)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True)
    description = models.TextField(blank=True)
    brand = models.CharField(max_length=100, blank=True)
    item_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)  # Added

class Store(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    code = models.CharField(max_length=50, unique=True)
    current_inventory = models.IntegerField(null=True, blank=True)  # Added
    forecasted_sales = models.FloatField(null=True, blank=True)     # Added

class ReturnAction(models.Model):
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)

class ReturnReason(models.Model):
    reason = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

class Customer(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True)
    address = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class ProductInstance(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    serial_number = models.CharField(max_length=100, unique=True)
    purchase_date = models.DateField(null=True, blank=True)
    customer = models.ForeignKey(Customer, on_delete=models.SET_NULL, null=True, blank=True)

class ReturnRequest(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    product_instance = models.ForeignKey(ProductInstance, on_delete=models.SET_NULL, null=True, blank=True)
    store = models.ForeignKey(Store, on_delete=models.SET_NULL, null=True)
    requested_by = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True, blank=True)
    customer = models.ForeignKey(Customer, on_delete=models.SET_NULL, null=True, blank=True)
    reason = models.ForeignKey(ReturnReason, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    predicted_action = models.ForeignKey(ReturnAction, on_delete=models.SET_NULL, null=True, blank=True, related_name='predicted_returns')
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    final_action = models.ForeignKey(ReturnAction, on_delete=models.SET_NULL, null=True, blank=True, related_name='final_returns')
    final_action_by = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True, blank=True, related_name='final_action_by')
    final_action_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=50, default='pending')
    auto_processed = models.BooleanField(default=False)
    auto_action_type = models.CharField(max_length=30, null=True, blank=True)  # e.g., 'restock', 'refurbish'
    recommended_store = models.ForeignKey('Store', null=True, blank=True, on_delete=models.SET_NULL)
    manual_review_flag = models.BooleanField(default=False)
    manual_review_reason = models.TextField(null=True, blank=True)
    # Optional for future:
    refund_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    exchange_date = models.DateField(null=True, blank=True)
    invoice_number = models.CharField(max_length=50, null=True, blank=True)

    # AI/Forecasting fields
    resale_probability_score = models.FloatField(null=True, blank=True)
    manual_review_flag = models.BooleanField(default=False)
    image_quality_score = models.FloatField(null=True, blank=True)
    recommended_store = models.ForeignKey(Store, on_delete=models.SET_NULL, null=True, blank=True, related_name='recommended_returns')
    channel_fit = models.CharField(
        max_length=20,
        choices=[('online', 'Online'), ('offline', 'Offline'), ('both', 'Both')],
        null=True, blank=True
    )
    demand_score = models.FloatField(null=True, blank=True)
    forecasted_sales = models.FloatField(null=True, blank=True)
    logistics_cost = models.FloatField(null=True, blank=True)
    item_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)  # For economic feasibility

class ReturnImage(models.Model):
    return_request = models.ForeignKey(ReturnRequest, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='return_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Attachment(models.Model):
    return_request = models.ForeignKey(ReturnRequest, on_delete=models.CASCADE, related_name='attachments')
    file = models.FileField(upload_to='return_attachments/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    description = models.CharField(max_length=255, blank=True)

class ReturnStatusHistory(models.Model):
    return_request = models.ForeignKey(ReturnRequest, on_delete=models.CASCADE, related_name='status_history')
    status = models.CharField(max_length=50)
    changed_by = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True, blank=True)
    changed_at = models.DateTimeField(auto_now_add=True)
    note = models.TextField(blank=True)