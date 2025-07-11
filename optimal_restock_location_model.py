"""
Optimal Restock Location Model
=============================

This model determines the best location to restock products based on:
- Predicted demand at each location
- Logistics costs (shipping, distance, handling)
- Current stock levels
- Store capacity and characteristics
- Product-specific factors

The model uses store IDs and product IDs to recommend optimal restocking locations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class OptimalRestockLocationModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.product_encoder = LabelEncoder()
        self.store_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_features(self, df):
        """Prepare features for training/prediction"""
        features = pd.DataFrame()
        
        # ID-based features
        if hasattr(self.product_encoder, 'classes_'):
            features['product_id_encoded'] = self.product_encoder.transform(df['product_id'])
        else:
            features['product_id_encoded'] = self.product_encoder.fit_transform(df['product_id'])
            
        if hasattr(self.store_encoder, 'classes_'):
            features['store_id_encoded'] = self.store_encoder.transform(df['store_id'])
        else:
            features['store_id_encoded'] = self.store_encoder.fit_transform(df['store_id'])
        
        # Logistics features
        features['logistics_cost_per_mile'] = df['logistics_cost_per_mile'].fillna(df['logistics_cost_per_mile'].median())
        features['logistics_cost_per_km'] = df['logistics_cost_per_km'].fillna(df['logistics_cost_per_km'].median())
        
        # Demand and stock features
        features['demand_score'] = df['demand_score'].fillna(df['demand_score'].median())
        features['stock_level'] = df['stock_level'].fillna(df['stock_level'].median())
        features['restock_threshold'] = df['restock_threshold'].fillna(df['restock_threshold'].median())
        features['quantity_sold'] = df['quantity_sold'].fillna(df['quantity_sold'].median())
        
        # Price features
        features['base_price'] = df['base_price'].fillna(df['base_price'].median())
        features['selling_price'] = df['selling_price'].fillna(df['selling_price'].median())
        features['profit_margin'] = df['profit_margin'].fillna(df['profit_margin'].median())
        
        # Customer features
        features['customer_rating'] = df['customer_rating'].fillna(df['customer_rating'].median())
        features['return_rate'] = df['return_rate'].fillna(df['return_rate'].median())
        features['trend_factor'] = df['trend_factor'].fillna(df['trend_factor'].median())
        
        # Binary features
        features['needs_restock'] = df['needs_restock'].astype(int)
        features['reordered'] = df['reordered'].astype(int)
        
        # Derived features for location optimization
        features['stock_shortage'] = np.maximum(0, features['restock_threshold'] - features['stock_level'])
        features['demand_to_stock_ratio'] = features['demand_score'] / (features['stock_level'] + 1)
        features['profit_per_logistics_cost'] = features['profit_margin'] / (features['logistics_cost_per_mile'] + 0.1)
        features['sales_velocity'] = features['quantity_sold'] / (features['stock_level'] + 1)
        
        # Location priority score (higher = better location)
        features['location_priority'] = (
            features['demand_score'] * 0.3 +
            features['stock_shortage'] * 0.2 +
            features['customer_rating'] * 0.2 +
            (1 / (features['logistics_cost_per_mile'] + 0.1)) * 0.2 +
            features['profit_margin'] * 0.1
        )
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        return features
    
    def create_target_variable(self, df):
        """Create target variable for optimal restock location"""
        # Group by product and create optimal location labels
        optimal_locations = []
        
        for product_id in df['product_id'].unique():
            product_data = df[df['product_id'] == product_id].copy()
            
            # Calculate location score for each store
            product_data['location_score'] = (
                product_data['demand_score'] * 0.3 +
                np.maximum(0, product_data['restock_threshold'] - product_data['stock_level']) * 0.25 +
                product_data['customer_rating'] * 0.2 +
                (1 / (product_data['logistics_cost_per_mile'] + 0.1)) * 0.15 +
                product_data['profit_margin'] * 0.1
            )
            
            # Mark top 20% locations as optimal (1), rest as suboptimal (0)
            threshold = product_data['location_score'].quantile(0.8)
            product_data['optimal_location'] = (product_data['location_score'] >= threshold).astype(int)
            
            optimal_locations.extend(product_data['optimal_location'].tolist())
        
        return np.array(optimal_locations)
    
    def train(self, csv_path):
        """Train the optimal restock location model"""
        print("Loading data...")
        df = pd.read_csv(csv_path)
        
        # Prepare features
        print("Preparing features...")
        X = self.prepare_features(df)
        y = self.create_target_variable(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': feature_importance
        }
    
    def predict_optimal_locations(self, df, top_n=5):
        """Predict optimal restock locations for products"""
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities for being optimal location
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create results dataframe
        results = df[['product_id', 'store_id', 'store_name', 'store_location']].copy()
        results['optimal_probability'] = probabilities
        results['logistics_cost'] = df['logistics_cost_per_mile']
        results['demand_score'] = df['demand_score']
        results['stock_level'] = df['stock_level']
        results['restock_threshold'] = df['restock_threshold']
        
        # Get top N locations for each product
        optimal_locations = []
        for product_id in df['product_id'].unique():
            product_results = results[results['product_id'] == product_id].copy()
            top_locations = product_results.nlargest(top_n, 'optimal_probability')
            optimal_locations.append(top_locations)
        
        return pd.concat(optimal_locations, ignore_index=True)
    
    def save_model(self, model_dir='models'):
        """Save the trained model and encoders"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, f'{model_dir}/location_optimizer.pkl')
        joblib.dump(self.product_encoder, f'{model_dir}/location_product_encoder.pkl')
        joblib.dump(self.store_encoder, f'{model_dir}/location_store_encoder.pkl')
        joblib.dump(self.scaler, f'{model_dir}/location_scaler.pkl')
        joblib.dump(self.feature_names, f'{model_dir}/location_feature_names.pkl')
        
        print(f"Location optimization model saved to {model_dir}/")
    
    def load_model(self, model_dir='models'):
        """Load the trained model and encoders"""
        self.model = joblib.load(f'{model_dir}/location_optimizer.pkl')
        self.product_encoder = joblib.load(f'{model_dir}/location_product_encoder.pkl')
        self.store_encoder = joblib.load(f'{model_dir}/location_store_encoder.pkl')
        self.scaler = joblib.load(f'{model_dir}/location_scaler.pkl')
        self.feature_names = joblib.load(f'{model_dir}/location_feature_names.pkl')
        
        print(f"Location optimization model loaded from {model_dir}/")

def main():
    """Main function to train and save the location optimization model"""
    print("Optimal Restock Location Model")
    print("=" * 40)
    
    # Initialize model
    optimizer = OptimalRestockLocationModel()
    
    # Train model
    data_path = "/Users/ommohite/Documents/RE.AI/data/train.csv"
    results = optimizer.train(data_path)
    
    # Save model
    optimizer.save_model()
    
    print("\nLocation optimization model training completed successfully!")
    print("Model saved to 'models/' directory")

if __name__ == "__main__":
    main()
