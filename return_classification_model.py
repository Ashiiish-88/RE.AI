"""
Product Return Classification Model
==================================

This model classifies product returns into three categories:
- restock: Items that can be resold as-is
- refurbish: Items that need minor repairs before resale
- recycle: Items that cannot be resold and should be recycled

This script trains a model by EXCLUDING the highly predictive text fields
to get a more realistic measure of performance based on other attributes.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class ProductReturnClassifier:
    """
    A classifier to determine the destination of a returned product.
    """
    def __init__(self):
        # We no longer need the text vectorizer
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.product_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.feature_names = []

    def prepare_features(self, df, is_training=True):
        """
        Prepares the feature set for model training or prediction.
        This version EXCLUDES text features to prevent data leakage.
        """
        features = pd.DataFrame(index=df.index)

        # Categorical features
        if is_training:
            features['product_encoded'] = self.product_encoder.fit_transform(df['product_name'])
            features['category_encoded'] = self.category_encoder.fit_transform(df['category'])
        else:
            features['product_encoded'] = self.product_encoder.transform(df['product_name'])
            features['category_encoded'] = self.category_encoder.transform(df['category'])

        # Numerical and boolean features
        features['age_days'] = df['age_days']
        features['warranty_valid'] = df['warranty_valid'].astype(int)
        features['packaging_intact'] = df['packaging_intact'].astype(int)

        if is_training:
            self.feature_names = features.columns.tolist()

        return features

    def train(self, csv_path):
        """
        Trains the classification model using data from a CSV file.
        """
        print("Loading data...")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Define all columns that could leak information
        leaky_columns = [
            'inspector_notes', 'condition_score', 'return_value',
            'processing_time_hours', 'return_reason', 'customer_notes'
        ]
        df_cleaned = df.drop(columns=leaky_columns, errors='ignore')

        print("Preparing features (excluding text fields)...")
        X = self.prepare_features(df_cleaned, is_training=True)
        y = df_cleaned['classification']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training model...")
        self.model.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        print(f"\n--- HONEST & REALISTIC MODEL PERFORMANCE ---")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nMost Important Features (Non-Text):")
        print(feature_importance.head())

    def save_model(self, model_dir='models'):
        """Saves the trained model and associated components."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'return_classifier_no_text.pkl'))
        joblib.dump(self.product_encoder, os.path.join(model_dir, 'product_encoder.pkl'))
        joblib.dump(self.category_encoder, os.path.join(model_dir, 'category_encoder.pkl'))
        joblib.dump(self.feature_names, os.path.join(model_dir, 'feature_names.pkl'))
        print(f"\nModel saved successfully to '{model_dir}/'")


def main():
    """Main function to run the model training and saving process."""
    print("Product Return Classification Model (No-Text Version)")
    print("=" * 50)

    data_path = "data/product_classification_dataset.csv"

    if not os.path.exists(data_path):
        print(f"Error: The file was not found at '{data_path}'")
        return

    classifier = ProductReturnClassifier()
    classifier.train(data_path)
    classifier.save_model()
    print("\nProcess completed.")

if __name__ == "__main__":
    main()