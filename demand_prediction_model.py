import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class DemandPredictor:
    def __init__(self):
        """
        Initializes the model with more aggressive hyperparameters to combat overfitting.
        - max_depth=6: Further limits tree depth.
        - min_samples_leaf=15: Requires a larger number of samples to form a leaf.
        - max_features='sqrt': Uses a random subset of features at each split to decorrelate trees.
        """
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=6,              # Reduced further to simplify trees
            min_samples_leaf=15,      # Increased significantly for regularization
            max_features='sqrt',      # Key change for reducing variance
            n_jobs=-1
        )
        # Encoders for essential ID and categorical columns
        self.product_encoder = LabelEncoder()
        self.store_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []

    def _encode_column(self, df, features, col_name, encoder):
        """Helper function to fit/transform a column with a LabelEncoder."""
        # Create a copy to avoid SettingWithCopyWarning
        series = df[col_name].copy()
        
        if hasattr(encoder, 'classes_'):
            known_labels = set(encoder.classes_)
            # Map unknown labels to a placeholder. Add placeholder to encoder if it's new.
            unknown_label = 'unknown'
            if unknown_label not in known_labels:
                encoder.classes_ = np.append(encoder.classes_, unknown_label)
            
            series[~series.isin(known_labels)] = unknown_label
            features[f'{col_name}_encoded'] = encoder.transform(series)
        else:
            features[f'{col_name}_encoded'] = encoder.fit_transform(series)
        return features

    def prepare_features(self, df):
        """
        Prepares a simplified and robust feature set for training or prediction.
        - Focuses on the most impactful IDs and numerical data.
        - Removes high-cardinality IDs (user, aisle, department) to reduce noise.
        """
        features = pd.DataFrame(index=df.index)

        # Essential ID-based features
        features = self._encode_column(df, features, 'product_id', self.product_encoder)
        features = self._encode_column(df, features, 'store_id', self.store_encoder)
        
        # Categorical feature encoding
        features = self._encode_column(df, features, 'season', self.season_encoder)

        # Core numerical features
        numerical_features = [
            'base_price', 'selling_price', 'cost_price', 'profit_margin',
            'quantity_sold', 'stock_level', 'restock_threshold', 'trend_factor',
            'customer_rating', 'return_rate', 'order_dow', 'order_hour_of_day',
            'days_since_prior_order', 'order_number', 'add_to_cart_order'
        ]
        
        for feature in numerical_features:
            if feature in df.columns:
                features[feature] = df[feature].fillna(df[feature].median())

        # Binary and derived features
        features['needs_restock'] = df['needs_restock'].astype(int)
        features['reordered'] = df['reordered'].astype(int)
        
        # Robustly calculate derived features to avoid division by zero
        features['price_ratio'] = features['selling_price'] / np.maximum(1, features.get('base_price', 1))
        features['stock_ratio'] = features['stock_level'] / np.maximum(1, features.get('restock_threshold', 1))
        features['profit_per_unit'] = features['selling_price'] - features['cost_price']
        features['is_weekend'] = (df['order_dow'].isin([0, 6])).astype(int)
        
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(0, inplace=True)

        if not self.feature_names:
            self.feature_names = features.columns.tolist()

        return features[self.feature_names]

    def train(self, csv_path):
        """Trains the demand prediction model and evaluates it for overfitting."""
        print("Loading data...")
        df = pd.read_csv(csv_path)

        print("Preparing features...")
        X = self.prepare_features(df)
        y = df['demand_score']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Training model...")
        self.model.fit(X_train_scaled, y_train)

        print("Evaluating model...")
        y_pred_test = self.model.predict(X_test_scaled)
        y_pred_train = self.model.predict(X_train_scaled)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print("\n--- Model Performance ---")
        print(f"Training R² Score: {r2_train:.4f}")
        print(f"Testing R² Score: {r2_test:.4f}")
        print(f"OVERFITTING GAP (Train R² - Test R²): {r2_train - r2_test:.4f}")
        print("-" * 25)

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self

    def predict(self, df):
        """Predicts demand for new data."""
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def save_model(self, model_dir='models'):
        """Saves the trained model and associated processors."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'demand_predictor.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'demand_scaler.pkl'))
        joblib.dump(self.feature_names, os.path.join(model_dir, 'demand_feature_names.pkl'))
        joblib.dump(self.product_encoder, os.path.join(model_dir, 'demand_product_encoder.pkl'))
        joblib.dump(self.store_encoder, os.path.join(model_dir, 'demand_store_encoder.pkl'))
        joblib.dump(self.season_encoder, os.path.join(model_dir, 'demand_season_encoder.pkl'))
        print(f"\nDemand model and processors saved to '{model_dir}/'")

def main():
    """Main function to train, save, and test the demand prediction model."""
    print("Demand Prediction Model Training")
    print("=" * 30)
    
    # --- IMPORTANT ---
    # Replace this path with the actual path to your training CSV file.
    data_path = "/Users/ommohite/Documents/RE.AI/data/train.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'")
        return

    # Initialize and train the model
    predictor = DemandPredictor().train(data_path)
    
    # Save the trained model
    predictor.save_model()

    # Example of loading the model and making a prediction
    print("\n--- Prediction Example ---")
    df_full = pd.read_csv(data_path)
    sample_data = df_full.head()  # Use the first 5 rows as new data
    
    predictor_loaded = DemandPredictor()
    predictor_loaded.load_model()
    
    predictions = predictor_loaded.predict(sample_data)
    print("Predicted demand scores for the first 5 rows:")
    print(predictions)

if __name__ == "__main__":
    main()