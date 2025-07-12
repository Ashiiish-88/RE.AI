from django.db import models

class Product(models.Model):
    product_id = models.BigIntegerField(primary_key=True)
    product_name = models.TextField()

    class Meta:
        db_table = 'products'

    def __str__(self):
        return self.product_name

class Store(models.Model):
    store_id = models.BigIntegerField(primary_key=True)
    store_name = models.TextField()
    location = models.TextField()

    class Meta:
        db_table = 'stores'

    def __str__(self):
        return self.store_name

class ReturnReason(models.Model):
    id = models.BigIntegerField(primary_key=True)
    reason = models.TextField()
    description = models.TextField()

    class Meta:
        db_table = 'returns_returnreason'

    def __str__(self):
        return self.reason

class ReturnAction(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.TextField()
    description = models.TextField()

    class Meta:
        db_table = 'returns_returnaction'

    def __str__(self):
        return self.name

class ReturnRequest(models.Model):
    id = models.BigIntegerField(primary_key=True)
    created_at = models.DateTimeField()
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    final_action_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=50)
    customer_id = models.BigIntegerField(null=True, blank=True)
    final_action_id = models.BigIntegerField(null=True, blank=True)
    final_action_by_id = models.IntegerField(null=True, blank=True)
    predicted_action = models.ForeignKey(ReturnAction, on_delete=models.SET_NULL, null=True, db_column='predicted_action_id', related_name='predicted_requests')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, db_column='product_id')
    product_instance_id = models.BigIntegerField(null=True, blank=True)
    reason = models.ForeignKey(ReturnReason, on_delete=models.SET_NULL, null=True, db_column='reason_id')
    requested_by_id = models.IntegerField(null=True, blank=True)
    store = models.ForeignKey(Store, on_delete=models.SET_NULL, null=True, db_column='store_id')
    channel_fit = models.CharField(max_length=50, null=True, blank=True)
    demand_score = models.FloatField(null=True, blank=True)
    forecasted_sales = models.FloatField(null=True, blank=True)
    image_quality_score = models.FloatField(null=True, blank=True)
    item_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    logistics_cost = models.FloatField(null=True, blank=True)
    manual_review_flag = models.BooleanField(default=False)
    profit = models.FloatField(null=True, blank=True)
    recommended_store_id = models.BigIntegerField(null=True, blank=True)
    resale_probability_score = models.FloatField(null=True, blank=True)
    auto_action_type = models.CharField(max_length=50, null=True, blank=True)
    auto_processed = models.BooleanField(default=False)
    exchange_date = models.DateField(null=True, blank=True)
    invoice_number = models.CharField(max_length=50, null=True, blank=True)
    manual_review_reason = models.TextField(null=True, blank=True)
    refund_amount = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    return_reason = models.TextField(null=True, blank=True)

    class Meta:
        db_table = 'returns_returnrequest'

    def __str__(self):
        return f"ReturnRequest {self.id}"

# You can add similar Meta/db_table for other models as needed.