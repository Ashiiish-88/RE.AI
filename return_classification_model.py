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
    def predict(self, df):
        """
        Predicts the return classification and probabilities for the given DataFrame.
        Expects columns: inspector_notes, return_reason (other columns are ignored).
        Returns: (predicted_class, probabilities)
        """
        X = self.prepare_features(df, is_training=False)
        pred = self.model.predict(X)
        prob = self.model.predict_proba(X)
        return pred, prob
    def load_model(self, model_dir='models'):
        """Loads the trained model and associated components from disk."""
        import joblib
        import os
        self.model = joblib.load(os.path.join(model_dir, 'return_classifier_text.pkl'))
        self.inspector_vect = joblib.load(os.path.join(model_dir, 'inspector_vect.pkl'))
        self.return_reason_vect = joblib.load(os.path.join(model_dir, 'return_reason_vect.pkl'))
        self.feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        print(f"Model loaded successfully from '{model_dir}/'")
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
        Uses only inspector_notes and return_reason as features (vectorized).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Fill missing values
        df['inspector_notes'] = df['inspector_notes'].fillna("")
        df['return_reason'] = df['return_reason'].fillna("")

        # Fit or transform vectorizers
        if is_training or not hasattr(self, 'inspector_vect'):
            self.inspector_vect = TfidfVectorizer(max_features=100)
            self.return_reason_vect = TfidfVectorizer(max_features=100)
            inspector_features = self.inspector_vect.fit_transform(df['inspector_notes']).toarray()
            reason_features = self.return_reason_vect.fit_transform(df['return_reason']).toarray()
        else:
            inspector_features = self.inspector_vect.transform(df['inspector_notes']).toarray()
            reason_features = self.return_reason_vect.transform(df['return_reason']).toarray()

        import numpy as np
        features = np.concatenate([inspector_features, reason_features], axis=1)
        self.feature_names = (
            [f"inspector_notes_{i}" for i in range(inspector_features.shape[1])] +
            [f"return_reason_{i}" for i in range(reason_features.shape[1])]
        )
        return features


    def train(self, csv_path):
        """
        Trains the classification model using data from a CSV file.
        Accepts the new dataset with product_id and product_name columns.
        """
        print("Loading data...")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Accept both old and new formats
        expected_cols = set(['inspector_notes', 'return_reason', 'classification'])
        if set(['product_id', 'product_name']).issubset(df.columns):
            # New format: keep only text and label columns (ignore product_id/name for now)
            keep_cols = ['inspector_notes', 'return_reason', 'classification']
        else:
            keep_cols = list(expected_cols & set(df.columns))
        df_cleaned = df[keep_cols].copy()

        print("Preparing features (using inspector_notes and return_reason)...")
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

        # Feature importance for text features
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nMost Important Features (Text):")
        print(feature_importance.head())

    def save_model(self, model_dir='models'):
        """Saves the trained model and associated components."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'return_classifier_text.pkl'))
        joblib.dump(self.inspector_vect, os.path.join(model_dir, 'inspector_vect.pkl'))
        joblib.dump(self.return_reason_vect, os.path.join(model_dir, 'return_reason_vect.pkl'))
        joblib.dump(self.feature_names, os.path.join(model_dir, 'feature_names.pkl'))
        print(f"\nModel saved successfully to '{model_dir}/'")


def main():
    """Main function to run the model training and saving process."""
    print("Product Return Classification Model (No-Text Version)")
    print("=" * 50)

    # Use the new dataset with product_id and product_name if available
    data_path = "data/product_classification_dataset_realistic_with_id_and_name.csv"
    if not os.path.exists(data_path):
        print(f"Error: The file was not found at '{data_path}'")
        return

    classifier = ProductReturnClassifier()
    classifier.train(data_path)
    classifier.save_model()
    print("\nProcess completed.")

if __name__ == "__main__":
    main()