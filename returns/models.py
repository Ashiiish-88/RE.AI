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
    item_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

class Store(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    code = models.CharField(max_length=50, unique=True)
    current_inventory = models.IntegerField(null=True, blank=True)
    forecasted_sales = models.FloatField(null=True, blank=True)

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
    auto_action_type = models.CharField(max_length=30, null=True, blank=True)
    recommended_store = models.ForeignKey('Store', null=True, blank=True, on_delete=models.SET_NULL, related_name='recommended_returns')
    manual_review_flag = models.BooleanField(default=False)
    manual_review_reason = models.TextField(null=True, blank=True)
    refund_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    exchange_date = models.DateField(null=True, blank=True)
    invoice_number = models.CharField(max_length=50, null=True, blank=True)
    resale_probability_score = models.FloatField(null=True, blank=True)
    image_quality_score = models.FloatField(null=True, blank=True)
    channel_fit = models.CharField(
        max_length=20,
        choices=[('online', 'Online'), ('offline', 'Offline'), ('both', 'Both')],
        null=True, blank=True
    )
    demand_score = models.FloatField(null=True, blank=True)
    forecasted_sales = models.FloatField(null=True, blank=True)
    logistics_cost = models.FloatField(null=True, blank=True)
    item_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

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

# --- MAPPING MODELS ---

class ProductMapping(models.Model):
    app_product = models.ForeignKey(Product, on_delete=models.CASCADE)
    csv_product_id = models.IntegerField()
    csv_product_name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.app_product.name} ↔ {self.csv_product_name} ({self.csv_product_id})"

class StoreMapping(models.Model):
    app_store = models.ForeignKey(Store, on_delete=models.CASCADE)
    csv_store_location = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.app_store.name} ↔ {self.csv_store_location}"

# --- DEMAND TABLE (from CSV) ---

class ProductDemand(models.Model):
    product_id = models.IntegerField()
    product_name = models.CharField(max_length=255)
    store_location = models.CharField(max_length=255)
    demand = models.FloatField()
    product_price = models.FloatField()
    cost_price = models.FloatField()
    logistics_cost_per_unit = models.FloatField()
    profit_margin = models.FloatField()
    profit_margin_pct = models.FloatField()
    # Add more fields if you need them for analytics

    class Meta:
        unique_together = ('product_id', 'store_location')

    def __str__(self):
        return f"{self.product_name} at {self.store_location}"