"""
Model Usage Example
==================

This script demonstrates how to use the trained models for predictions.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from return_classification_model import ProductReturnClassifier
from demand_prediction_model import DemandPredictor
from optimal_restock_location_model import OptimalRestockLocationModel

def demonstrate_model_usage():
    """Demonstrate how to use all three trained models"""
    print("=" * 60)
    print("MODEL USAGE DEMONSTRATION")
    print("=" * 60)
    
    # Load the trained models
    print("Loading trained models...")
    
    try:
        # Load Return Classification Model
        classifier = ProductReturnClassifier()
        classifier.load_model()
        
        # Load Demand Prediction Model
        predictor = DemandPredictor()
        predictor.load_model()
        
        # Load Optimal Restock Location Model
        optimizer = OptimalRestockLocationModel()
        optimizer.load_model()
        
        print("✓ All models loaded successfully!\n")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        print("Please run 'train_all_models.py' first to train the models.")
        return
    
    # Example 1: Return Classification
    print("1. RETURN CLASSIFICATION EXAMPLE")
    print("-" * 40)
    
    # Create sample return data
    sample_return = pd.DataFrame({
        'product_name': ['Sony WH-1000XM4 Wireless Bluetooth Headphones'],
        'return_reason': ['Minor cosmetic damage'],
        'inspector_notes': ['Small scratch on surface, otherwise good condition'],
        'customer_notes': ['Arrived with minor scratches'],
        'warranty_valid': [True],
        'age_days': [15],
        'packaging_intact': [True],
        'category': ['Electronics']
    })
    
    predictions, probabilities = classifier.predict(sample_return)
    
    print("Sample Return:")
    print(f"  Product: {sample_return['product_name'].iloc[0]}")
    print(f"  Reason: {sample_return['return_reason'].iloc[0]}")
    print(f"  Age: {sample_return['age_days'].iloc[0]} days")
    print(f"  Prediction: {predictions[0]}")
    print(f"  Confidence: {max(probabilities[0]):.3f}")
    print()
    
    # Example 2: Demand Prediction
    print("2. DEMAND PREDICTION EXAMPLE")
    print("-" * 40)
    
    # Create sample product data
    sample_product = pd.DataFrame({
        'product_id': [1],
        'store_id': [5],
        'base_price': [299.99],
        'selling_price': [399.99],
        'cost_price': [199.99],
        'profit_margin': [0.5],
        'quantity_sold': [25],
        'stock_level': [50],
        'restock_threshold': [20],
        'trend_factor': [1.2],
        'customer_rating': [4.5],
        'return_rate': [0.05],
        'order_dow': [1],
        'order_hour_of_day': [14],
        'days_since_prior_order': [7],
        'needs_restock': [0],
        'reordered': [1],
        'season': ['Spring']
    })
    
    demand_prediction = predictor.predict(sample_product)
    
    print("Sample Product:")
    print(f"  Product ID: {sample_product['product_id'].iloc[0]}")
    print(f"  Store ID: {sample_product['store_id'].iloc[0]}")
    print(f"  Current Stock: {sample_product['stock_level'].iloc[0]}")
    print(f"  Customer Rating: {sample_product['customer_rating'].iloc[0]}")
    print(f"  Predicted Demand: {demand_prediction[0]:.2f}")
    print()
    
    # Example 3: Optimal Restock Location
    print("3. OPTIMAL RESTOCK LOCATION EXAMPLE")
    print("-" * 40)
    
    # Create sample location data for a product
    sample_locations = pd.DataFrame({
        'product_id': [1, 1, 1, 1, 1],
        'store_id': [1, 2, 3, 4, 5],
        'store_name': [
            'Walmart Supercenter Downtown',
            'Walmart Neighborhood Market',
            'Walmart Supercenter Suburban',
            'Walmart Express',
            'Walmart Supercenter Mall'
        ],
        'store_location': [
            'Downtown Seattle, WA',
            'Mall Plaza Portland, OR',
            'Suburban Phoenix, AZ',
            'Airport Denver, CO',
            'Mall Miami, FL'
        ],
        'logistics_cost_per_mile': [1.2, 1.5, 0.8, 2.0, 1.8],
        'logistics_cost_per_km': [0.75, 0.93, 0.50, 1.24, 1.12],
        'demand_score': [8.5, 6.2, 9.1, 4.8, 7.3],
        'stock_level': [15, 25, 8, 35, 20],
        'restock_threshold': [20, 20, 20, 20, 20],
        'quantity_sold': [30, 18, 35, 12, 22],
        'base_price': [299.99] * 5,
        'selling_price': [399.99] * 5,
        'profit_margin': [0.5] * 5,
        'customer_rating': [4.5, 4.2, 4.7, 4.1, 4.4],
        'return_rate': [0.05] * 5,
        'trend_factor': [1.2] * 5,
        'needs_restock': [1, 0, 1, 0, 1],
        'reordered': [1] * 5
    })
    
    optimal_locations = optimizer.predict_optimal_locations(sample_locations, top_n=3)
    
    print("Top 3 Optimal Restock Locations:")
    for i, row in optimal_locations.iterrows():
        print(f"  {i+1}. {row['store_name']}")
        print(f"     Location: {row['store_location']}")
        print(f"     Probability: {row['optimal_probability']:.3f}")
        print(f"     Demand Score: {row['demand_score']:.1f}")
        print(f"     Current Stock: {row['stock_level']}")
        print(f"     Logistics Cost: ${row['logistics_cost']:.2f}/mile")
        print()
    
    print("=" * 60)
    print("INTEGRATION EXAMPLE: COMPLETE WORKFLOW")
    print("=" * 60)
    
    # Complete workflow example
    print("Scenario: A returned product needs to be processed and restocked")
    print()
    
    # Step 1: Classify the return
    print("Step 1: Classify the return")
    return_classification = predictions[0]
    print(f"  Return Classification: {return_classification}")
    
    if return_classification == 'restock':
        print("  → Product can be restocked directly")
        
        # Step 2: Predict demand
        print("\nStep 2: Predict demand for restocking")
        demand = demand_prediction[0]
        print(f"  Predicted Demand: {demand:.2f}")
        
        # Step 3: Find optimal location
        print("\nStep 3: Find optimal restock location")
        best_location = optimal_locations.iloc[0]
        print(f"  Best Location: {best_location['store_name']}")
        print(f"  Confidence: {best_location['optimal_probability']:.3f}")
        
        print("\nRecommendation:")
        print(f"  ✓ Restock {sample_return['product_name'].iloc[0]}")
        print(f"  ✓ Expected demand: {demand:.2f} units")
        print(f"  ✓ Optimal location: {best_location['store_name']}")
        print(f"  ✓ Logistics cost: ${best_location['logistics_cost']:.2f}/mile")
        
    elif return_classification == 'refurbish':
        print("  → Product needs refurbishment before restocking")
        print("  → Send to refurbishment center first")
        
    else:  # recycle
        print("  → Product should be recycled")
        print("  → Do not restock")

if __name__ == "__main__":
    demonstrate_model_usage()
