"""
Django Models for Enhanced Instacart Dataset
===========================================

This file defines Django models for storing and managing the enhanced Instacart
dataset for demand forecasting and inventory optimization.
"""

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
import json

class Product(models.Model):
    """
    Model for storing product information from Instacart dataset
    """
    product_id = models.IntegerField(unique=True, primary_key=True)
    product_name = models.CharField(max_length=255)
    aisle_id = models.IntegerField()
    aisle = models.CharField(max_length=100)
    department_id = models.IntegerField()
    department = models.CharField(max_length=100)
    
    # Enhanced demand metrics
    total_orders = models.IntegerField(default=0)
    avg_reorder_rate = models.FloatField(default=0.0)
    unique_customers = models.IntegerField(default=0)
    avg_reorder_days = models.FloatField(default=0.0)
    demand_score = models.FloatField(default=0.0)
    
    DEMAND_CATEGORIES = [
        ('Low', 'Low Demand'),
        ('Medium', 'Medium Demand'),
        ('High', 'High Demand'),
        ('Very High', 'Very High Demand'),
    ]
    demand_category = models.CharField(max_length=20, choices=DEMAND_CATEGORIES, default='Low')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'products'
        indexes = [
            models.Index(fields=['demand_category']),
            models.Index(fields=['department']),
            models.Index(fields=['aisle']),
            models.Index(fields=['-demand_score']),
        ]
    
    def __str__(self):
        return f"{self.product_name} (ID: {self.product_id})"

class DemandForecast(models.Model):
    """
    Model for storing demand forecasts and time series data
    """
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='demand_forecasts')
    date = models.DateTimeField()
    hour = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(23)])
    day_of_week = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(6)])
    week = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(53)])
    month = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(12)])
    quarter = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(4)])
    
    # Demand data
    actual_demand = models.IntegerField(default=0)
    predicted_demand = models.FloatField(null=True, blank=True)
    confidence_interval_lower = models.FloatField(null=True, blank=True)
    confidence_interval_upper = models.FloatField(null=True, blank=True)
    
    # Time-based features
    is_weekend = models.BooleanField(default=False)
    is_morning = models.BooleanField(default=False)
    is_evening = models.BooleanField(default=False)
    
    # Lag features (stored as JSON for flexibility)
    lag_features = models.JSONField(default=dict, blank=True)
    rolling_features = models.JSONField(default=dict, blank=True)
    
    # Model metadata
    model_version = models.CharField(max_length=50, default='v1.0')
    prediction_accuracy = models.FloatField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'demand_forecasts'
        unique_together = ['product', 'date', 'hour']
        indexes = [
            models.Index(fields=['product', 'date']),
            models.Index(fields=['date', 'hour']),
            models.Index(fields=['day_of_week']),
            models.Index(fields=['is_weekend']),
            models.Index(fields=['-predicted_demand']),
        ]
    
    def __str__(self):
        return f"{self.product.product_name} - {self.date.strftime('%Y-%m-%d %H:%M')}"

class InventoryOptimization(models.Model):
    """
    Model for storing inventory optimization recommendations
    """
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='inventory_optimizations')
    date = models.DateField()
    
    # Current inventory levels
    current_stock = models.IntegerField(default=0)
    reorder_point = models.IntegerField(default=0)
    optimal_stock_level = models.IntegerField(default=0)
    safety_stock = models.IntegerField(default=0)
    
    # Demand predictions
    predicted_daily_demand = models.FloatField(default=0.0)
    predicted_weekly_demand = models.FloatField(default=0.0)
    predicted_monthly_demand = models.FloatField(default=0.0)
    
    # Optimization metrics
    service_level = models.FloatField(default=0.95)  # 95% service level
    carrying_cost = models.FloatField(default=0.0)
    stockout_cost = models.FloatField(default=0.0)
    total_cost = models.FloatField(default=0.0)
    
    # Recommendations
    ACTIONS = [
        ('maintain', 'Maintain Current Level'),
        ('reorder', 'Reorder Required'),
        ('reduce', 'Reduce Inventory'),
        ('increase', 'Increase Inventory'),
        ('urgent', 'Urgent Reorder'),
    ]
    recommended_action = models.CharField(max_length=20, choices=ACTIONS, default='maintain')
    recommended_quantity = models.IntegerField(default=0)
    priority_level = models.IntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(5)])
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'inventory_optimizations'
        unique_together = ['product', 'date']
        indexes = [
            models.Index(fields=['date']),
            models.Index(fields=['recommended_action']),
            models.Index(fields=['-priority_level']),
        ]
    
    def __str__(self):
        return f"{self.product.product_name} - {self.date} - {self.recommended_action}"

class MLModel(models.Model):
    """
    Model for storing machine learning model metadata and performance
    """
    model_name = models.CharField(max_length=100)
    model_version = models.CharField(max_length=50)
    model_type = models.CharField(max_length=50)  # ARIMA, LSTM, Prophet, XGBoost, etc.
    
    # Model parameters (stored as JSON)
    hyperparameters = models.JSONField(default=dict)
    
    # Performance metrics
    mae = models.FloatField(null=True, blank=True)  # Mean Absolute Error
    mse = models.FloatField(null=True, blank=True)  # Mean Squared Error
    rmse = models.FloatField(null=True, blank=True)  # Root Mean Squared Error
    mape = models.FloatField(null=True, blank=True)  # Mean Absolute Percentage Error
    r2_score = models.FloatField(null=True, blank=True)  # R-squared
    
    # Training metadata
    training_start_date = models.DateTimeField()
    training_end_date = models.DateTimeField()
    training_samples = models.IntegerField(default=0)
    feature_count = models.IntegerField(default=0)
    
    # Model status
    STATUS_CHOICES = [
        ('training', 'Training'),
        ('active', 'Active'),
        ('retired', 'Retired'),
        ('failed', 'Failed'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='training')
    
    # File path (if model is saved to disk)
    model_file_path = models.CharField(max_length=255, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'ml_models'
        unique_together = ['model_name', 'model_version']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['-created_at']),
        ]
    
    def __str__(self):
        return f"{self.model_name} v{self.model_version} ({self.model_type})"

class DemandAlert(models.Model):
    """
    Model for storing demand alerts and notifications
    """
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='demand_alerts')
    
    ALERT_TYPES = [
        ('stockout', 'Stockout Risk'),
        ('overstock', 'Overstock Risk'),
        ('demand_spike', 'Demand Spike'),
        ('demand_drop', 'Demand Drop'),
        ('forecast_error', 'Forecast Error'),
    ]
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    
    SEVERITY_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS, default='medium')
    
    # Alert details
    message = models.TextField()
    current_value = models.FloatField()
    threshold_value = models.FloatField()
    expected_value = models.FloatField()
    
    # Status
    is_active = models.BooleanField(default=True)
    is_acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.CharField(max_length=100, blank=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'demand_alerts'
        indexes = [
            models.Index(fields=['is_active']),
            models.Index(fields=['severity']),
            models.Index(fields=['alert_type']),
            models.Index(fields=['-created_at']),
        ]
    
    def __str__(self):
        return f"{self.alert_type} - {self.product.product_name} ({self.severity})"

class DemandTrend(models.Model):
    """
    Model for storing demand trends and patterns
    """
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='demand_trends')
    date = models.DateField()
    
    # Trend analysis
    TREND_DIRECTIONS = [
        ('increasing', 'Increasing'),
        ('decreasing', 'Decreasing'),
        ('stable', 'Stable'),
        ('volatile', 'Volatile'),
    ]
    trend_direction = models.CharField(max_length=20, choices=TREND_DIRECTIONS)
    trend_strength = models.FloatField(default=0.0)  # 0-1 scale
    
    # Seasonality
    has_seasonality = models.BooleanField(default=False)
    seasonality_period = models.IntegerField(null=True, blank=True)  # Days
    seasonality_strength = models.FloatField(default=0.0)
    
    # Statistical measures
    average_demand = models.FloatField(default=0.0)
    demand_variance = models.FloatField(default=0.0)
    demand_cv = models.FloatField(default=0.0)  # Coefficient of Variation
    
    # Growth metrics
    growth_rate = models.FloatField(default=0.0)  # Monthly growth rate
    growth_acceleration = models.FloatField(default=0.0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'demand_trends'
        unique_together = ['product', 'date']
        indexes = [
            models.Index(fields=['date']),
            models.Index(fields=['trend_direction']),
            models.Index(fields=['has_seasonality']),
        ]
    
    def __str__(self):
        return f"{self.product.product_name} - {self.date} - {self.trend_direction}"
