#!/usr/bin/env python3
"""
Model Training Script
====================

Train and evaluate demand prediction and profit optimization models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.demand_system import DemandPredictionSystem
from src.utils.data_processor import DataProcessor


def main():
    """Main training pipeline."""
    print("🚀 Starting Walmart Demand Prediction Model Training")
    print("=" * 60)
    
    # Initialize components
    processor = DataProcessor()
    system = DemandPredictionSystem()
    
    # Load data
    print("\n📊 Loading and preprocessing data...")
    df = processor.load_train_data()
    processor.print_data_summary(df)
    
    # Train demand model
    print("\n🤖 Training demand prediction models...")
    demand_results = system.train_demand_model(df)
    
    # Print model comparison
    print("\n📈 Model Performance Comparison:")
    print("-" * 50)
    for model_name, metrics in demand_results.items():
        print(f"{model_name}:")
        print(f"  Train R²: {metrics['train_r2']:.4f}")
        print(f"  Test R²:  {metrics['test_r2']:.4f}")
        print(f"  Overfitting: {metrics['overfitting']:.4f}")
        print()
    
    # Train profit model
    print("💰 Training profit optimization model...")
    profit_results = system.train_profit_model(df)
    
    print(f"Profit Model Performance:")
    print(f"  Train R²: {profit_results['train_r2']:.4f}")
    print(f"  Test R²:  {profit_results['test_r2']:.4f}")
    print(f"  Overfitting: {profit_results['overfitting']:.4f}")
    
    # Save models
    print("\n💾 Saving trained models...")
    system.save_models()
    
    print("\n✅ Training completed successfully!")
    print("🎯 Models are ready for production use!")


if __name__ == "__main__":
    main()
