"""
Master Training Script
=====================

This script trains all three models:
1. Product Return Classification Model
2. Demand Prediction Model
3. Optimal Restock Location Model

Run this script to train all models at once.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from return_classification_model import ProductReturnClassifier
from demand_prediction_model import DemandPredictor
from optimal_restock_location_model import OptimalRestockLocationModel

def train_all_models():
    """Train all three models"""
    print("=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Data paths
    classification_data = "data/product_classification_dataset_realistic_with_id.csv"
    demand_data = "/Users/ommohite/Documents/RE.AI/data/train.csv"
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Training results
    results = {}
    
    # 1. Train Return Classification Model
    print("\n1. TRAINING RETURN CLASSIFICATION MODEL")
    print("-" * 50)
    try:
        classifier = ProductReturnClassifier()
        classification_results = classifier.train(classification_data)
        classifier.save_model()
        results['classification'] = classification_results
        print("✓ Return Classification Model trained successfully!")
    except Exception as e:
        print(f"✗ Error training Return Classification Model: {e}")
        results['classification'] = {'error': str(e)}
    
    # 2. Train Demand Prediction Model
    print("\n2. TRAINING DEMAND PREDICTION MODEL")
    print("-" * 50)
    try:
        predictor = DemandPredictor()
        demand_results = predictor.train(demand_data)
        predictor.save_model()
        results['demand'] = demand_results
        print("✓ Demand Prediction Model trained successfully!")
    except Exception as e:
        print(f"✗ Error training Demand Prediction Model: {e}")
        results['demand'] = {'error': str(e)}
    
    # 3. Train Optimal Restock Location Model
    print("\n3. TRAINING OPTIMAL RESTOCK LOCATION MODEL")
    print("-" * 50)
    try:
        optimizer = OptimalRestockLocationModel()
        location_results = optimizer.train(demand_data)
        optimizer.save_model()
        results['location'] = location_results
        print("✓ Optimal Restock Location Model trained successfully!")
    except Exception as e:
        print(f"✗ Error training Optimal Restock Location Model: {e}")
        results['location'] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} MODEL:")
        if 'error' in result:
            print(f"  Status: ✗ FAILED")
            print(f"  Error: {result['error']}")
        else:
            print(f"  Status: ✓ SUCCESS")
            if model_name == 'classification':
                print(f"  Accuracy: {result['accuracy']:.4f}")
            elif model_name == 'demand':
                print(f"  RMSE: {result['rmse']:.4f}")
                print(f"  R² Score: {result['r2']:.4f}")
            elif model_name == 'location':
                print(f"  Accuracy: {result['accuracy']:.4f}")
    
    print(f"\nCompleted at: {datetime.now()}")
    print("All models have been trained and saved to 'models/' directory")
    
    return results

def test_models():
    """Test all trained models with sample data"""
    print("\n" + "=" * 60)
    print("TESTING TRAINED MODELS")
    print("=" * 60)
    
    try:
        # Load sample data for testing
        classification_data = pd.read_csv("data/product_classification_dataset_realistic_with_id.csv")
        demand_data = pd.read_csv("/Users/ommohite/Documents/RE.AI/data/train.csv")
        
        # Test Classification Model
        print("\n1. TESTING RETURN CLASSIFICATION MODEL")
        print("-" * 50)
        classifier = ProductReturnClassifier()
        classifier.load_model()
        
        sample_returns = classification_data.head(5)
        predictions, probabilities = classifier.predict(sample_returns)
        
        print("Sample Predictions:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"  Return {i+1}: {pred} (confidence: {max(prob):.3f})")
        
        # Test Demand Prediction Model
        print("\n2. TESTING DEMAND PREDICTION MODEL")
        print("-" * 50)
        predictor = DemandPredictor()
        predictor.load_model()
        
        sample_products = demand_data.head(5)
        demand_predictions = predictor.predict(sample_products)
        
        print("Sample Demand Predictions:")
        for i, pred in enumerate(demand_predictions):
            print(f"  Product {i+1}: {pred:.2f}")
        
        # Test Location Optimization Model
        print("\n3. TESTING OPTIMAL RESTOCK LOCATION MODEL")
        print("-" * 50)
        optimizer = OptimalRestockLocationModel()
        optimizer.load_model()
        
        sample_locations = demand_data.head(20)  # More samples for location optimization
        optimal_locations = optimizer.predict_optimal_locations(sample_locations, top_n=3)
        
        print("Sample Optimal Locations:")
        print(optimal_locations[['product_id', 'store_name', 'optimal_probability']].head(10))
        
        print("\n✓ All models tested successfully!")
        
    except Exception as e:
        print(f"✗ Error testing models: {e}")

if __name__ == "__main__":
    # Train all models
    results = train_all_models()
    
    # Test models if training was successful
    if all('error' not in result for result in results.values()):
        test_models()
    else:
        print("\nSkipping model testing due to training errors.")
