import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class DemandPredictionSystem:
    def __init__(self):
        self.demand_model = None
        self.profit_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def prepare_features(self, df, fit_encoders=True):
        """Prepare features for model training"""
        df = df.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Categorical columns to encode
        categorical_cols = ['product_name', 'aisle', 'department', 'store_location', 'distribution_center']
        
        # Encode categorical variables
        for col in categorical_cols:
            if fit_encoders:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].map(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ else -1
                    )
                else:
                    df[f'{col}_encoded'] = -1
        
        # Feature columns for demand prediction
        self.feature_columns = [
            'product_id', 'hour', 'day_of_week', 'week', 'month', 'day_of_month', 
            'quarter', 'is_weekend', 'is_morning', 'is_evening', 'unique_customers',
            'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_7', 'demand_lag_14',
            'product_price', 'cost_price', 'distance_miles', 'logistics_cost_per_unit',
            'product_name_encoded', 'aisle_encoded', 'department_encoded', 
            'store_location_encoded', 'distribution_center_encoded'
        ]
        
        return df
    
    def train_demand_model(self, df):
        """Train the demand prediction model"""
        print("Training demand prediction model...")
        
        # Prepare features
        df_processed = self.prepare_features(df, fit_encoders=True)
        
        # Features and target
        X = df_processed[self.feature_columns]
        y = df_processed['demand']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models with optimized parameters to reduce overfitting
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=50,           # Reduced from 100
                max_depth=10,              # Limit tree depth to prevent overfitting
                min_samples_split=10,      # Require more samples to split
                min_samples_leaf=5,        # Require more samples in leaves
                max_features=0.7,          # Use 70% of features
                random_state=42, 
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300,          # Increased for even better performance
                learning_rate=0.03,        # Further reduced learning rate
                max_depth=8,               # Slightly increased depth
                min_samples_split=8,       # Fine-tuned
                min_samples_leaf=4,        # Fine-tuned
                subsample=0.85,           # Slightly increased subsample
                max_features=0.8,         # Use 80% of features
                random_state=42
            ),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[name] = {
                'Train MSE': train_mse,
                'Test MSE': test_mse,
                'Train R2': train_r2,
                'Test R2': test_r2,
                'Train MAE': train_mae,
                'Test MAE': test_mae
            }
            
            trained_models[name] = model
        
        # Select best model based on test R2
        best_model_name = max(results.keys(), key=lambda x: results[x]['Test R2'])
        self.demand_model = trained_models[best_model_name]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Test R2 Score: {results[best_model_name]['Test R2']:.4f}")
        
        return results, best_model_name
    
    def train_profit_model(self, df):
        """Train the profit prediction model WITHOUT data leakage"""
        print("\nTraining profit optimization model...")
        
        # Prepare features for profit prediction
        df_processed = self.prepare_features(df, fit_encoders=False)
        
        # Add predicted demand as feature
        X_demand = df_processed[self.feature_columns]
        if isinstance(self.demand_model, LinearRegression):
            X_demand_scaled = self.scaler.transform(X_demand)
            df_processed['predicted_demand'] = self.demand_model.predict(X_demand_scaled)
        else:
            df_processed['predicted_demand'] = self.demand_model.predict(X_demand)
        
        # Features for profit prediction (REMOVED LEAKAGE: no product_price, cost_price, logistics_cost_per_unit)
        profit_features = [
            'predicted_demand', 'distance_miles', 'product_id', 
            'store_location_encoded', 'distribution_center_encoded',
            'hour', 'day_of_week', 'month', 'is_weekend'
        ]
        
        X_profit = df_processed[profit_features]
        y_profit = df_processed['profit_margin']
        
        # Split data to properly evaluate profit model
        from sklearn.model_selection import train_test_split
        X_train_profit, X_test_profit, y_train_profit, y_test_profit = train_test_split(
            X_profit, y_profit, test_size=0.2, random_state=42
        )
        
        # Train profit model
        self.profit_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=10,
            random_state=42, 
            n_jobs=-1
        )
        self.profit_model.fit(X_train_profit, y_train_profit)
        
        # Calculate profit model performance on both train and test
        y_pred_train_profit = self.profit_model.predict(X_train_profit)
        y_pred_test_profit = self.profit_model.predict(X_test_profit)
        
        train_profit_r2 = r2_score(y_train_profit, y_pred_train_profit)
        test_profit_r2 = r2_score(y_test_profit, y_pred_test_profit)
        
        print(f"Profit Model Train R2 Score: {train_profit_r2:.4f}")
        print(f"Profit Model Test R2 Score: {test_profit_r2:.4f}")
        print(f"Profit Model Generalization Gap: {abs(train_profit_r2 - test_profit_r2):.4f}")
        
        return test_profit_r2
    
    def predict_demand(self, product_id, date, store_location):
        """Predict demand for given inputs"""
        if self.demand_model is None:
            raise ValueError("Demand model not trained yet!")
        
        # Create a sample row with the given inputs
        # We'll use average values for other features
        sample_data = {
            'product_id': product_id,
            'date': pd.to_datetime(date),
            'store_location': store_location,
            # Add default values for other required features
            'product_name': f'Product_{product_id}',
            'aisle': 'Aisle_1',
            'department': 'Department_1',
            'hour': 12,
            'day_of_week': pd.to_datetime(date).dayofweek,
            'week': pd.to_datetime(date).isocalendar().week,
            'month': pd.to_datetime(date).month,
            'day_of_month': pd.to_datetime(date).day,
            'quarter': (pd.to_datetime(date).month - 1) // 3 + 1,
            'is_weekend': 1 if pd.to_datetime(date).dayofweek >= 5 else 0,
            'is_morning': 0,
            'is_evening': 0,
            'unique_customers': 1000,
            'demand_lag_1': 15.0,
            'demand_lag_2': 15.0,
            'demand_lag_3': 15.0,
            'demand_lag_7': 15.0,
            'demand_lag_14': 15.0,
            'product_price': 10.0,
            'cost_price': 7.0,
            'distance_miles': 100,
            'logistics_cost_per_unit': 1.0,
            'distribution_center': 'Chicago_IL'
        }
        
        sample_df = pd.DataFrame([sample_data])
        sample_processed = self.prepare_features(sample_df, fit_encoders=False)
        
        X_sample = sample_processed[self.feature_columns]
        
        if isinstance(self.demand_model, LinearRegression):
            X_sample_scaled = self.scaler.transform(X_sample)
            predicted_demand = self.demand_model.predict(X_sample_scaled)[0]
        else:
            predicted_demand = self.demand_model.predict(X_sample)[0]
        
        return max(0, predicted_demand)
    
    def find_optimal_store(self, predicted_demand, product_id, store_locations, distances_df):
        """Find the most profitable store for transportation"""
        if self.profit_model is None:
            raise ValueError("Profit model not trained yet!")
        
        results = []
        
        for store in store_locations:
            # Get distance from current location to target store
            # For now, use a default distance calculation
            distance = 200  # Default distance
            
            # Create features for profit prediction (matching the updated profit model)
            profit_features = {
                'predicted_demand': predicted_demand,
                'distance_miles': distance,
                'product_id': product_id,
                'store_location_encoded': self.label_encoders.get('store_location', LabelEncoder()).transform([store])[0] if store in self.label_encoders.get('store_location', LabelEncoder()).classes_ else -1,
                'distribution_center_encoded': 0,  # Default
                'hour': 12,  # Default noon
                'day_of_week': 1,  # Default Monday
                'month': 6,  # Default June
                'is_weekend': 0  # Default weekday
            }
            
            profit_df = pd.DataFrame([profit_features])
            predicted_profit = self.profit_model.predict(profit_df)[0]
            
            results.append({
                'store': store,
                'predicted_profit': predicted_profit,
                'distance': distance,
                'calculated_logistics_cost': distance * 0.01  # Calculate logistics cost here
            })
        
        # Sort by predicted profit (descending)
        results.sort(key=lambda x: x['predicted_profit'], reverse=True)
        
        return results
    
    def save_models(self):
        """Save trained models"""
        joblib.dump(self.demand_model, 'models/demand_model.pkl')
        joblib.dump(self.profit_model, 'models/profit_model.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.feature_columns, 'models/feature_columns.pkl')
        print("Models saved successfully!")
    
    def load_models(self):
        """Load trained models"""
        self.demand_model = joblib.load('models/demand_model.pkl')
        self.profit_model = joblib.load('models/profit_model.pkl')
        self.label_encoders = joblib.load('models/label_encoders.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.feature_columns = joblib.load('models/feature_columns.pkl')
        print("Models loaded successfully!")

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/train.csv')
    
    # Initialize system
    system = DemandPredictionSystem()
    
    # Train models
    demand_results, best_model = system.train_demand_model(df)
    profit_score = system.train_profit_model(df)
    
    # Create results table
    results_df = pd.DataFrame(demand_results).T
    
    # Display results
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(results_df.round(4))
    
    # Create models directory and save models
    import os
    os.makedirs('models', exist_ok=True)
    system.save_models()
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Model comparison
    plt.subplot(2, 3, 1)
    models = results_df.index
    train_r2 = results_df['Train R2']
    test_r2 = results_df['Test R2']
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, train_r2, width, label='Train R2', alpha=0.8)
    plt.bar(x + width/2, test_r2, width, label='Test R2', alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('R2 Score')
    plt.title('Model Performance Comparison (R2 Score)')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: MSE comparison
    plt.subplot(2, 3, 2)
    train_mse = results_df['Train MSE']
    test_mse = results_df['Test MSE']
    
    plt.bar(x - width/2, train_mse, width, label='Train MSE', alpha=0.8)
    plt.bar(x + width/2, test_mse, width, label='Test MSE', alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('MSE')
    plt.title('Model Performance Comparison (MSE)')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: MAE comparison
    plt.subplot(2, 3, 3)
    train_mae = results_df['Train MAE']
    test_mae = results_df['Test MAE']
    
    plt.bar(x - width/2, train_mae, width, label='Train MAE', alpha=0.8)
    plt.bar(x + width/2, test_mae, width, label='Test MAE', alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('MAE')
    plt.title('Model Performance Comparison (MAE)')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Demand distribution
    plt.subplot(2, 3, 4)
    plt.hist(df['demand'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.title('Distribution of Demand')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Profit vs Distance
    plt.subplot(2, 3, 5)
    plt.scatter(df['distance_miles'], df['profit_margin'], alpha=0.5, color='green')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Profit Margin')
    plt.title('Profit vs Distance Relationship')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Store locations performance
    plt.subplot(2, 3, 6)
    store_profit = df.groupby('store_location')['profit_margin'].mean().sort_values(ascending=False)
    top_stores = store_profit.head(10)
    plt.barh(range(len(top_stores)), top_stores.values, color='orange', alpha=0.7)
    plt.yticks(range(len(top_stores)), [loc.replace('_', ' ') for loc in top_stores.index])
    plt.xlabel('Average Profit Margin')
    plt.title('Top 10 Most Profitable Store Locations')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nBest performing model: {best_model}")
    print(f"Test R2 Score: {results_df.loc[best_model, 'Test R2']:.4f}")
    print(f"Profit Model R2 Score: {profit_score:.4f}")

if __name__ == "__main__":
    main()
