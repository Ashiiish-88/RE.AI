import os
import joblib
import pandas as pd
import numpy as np
import scipy.sparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'returns', 'models')

# ---------------------------
# Demand Model: Load Artifacts
# ---------------------------
demand_model = joblib.load(os.path.join(MODEL_DIR, 'demand_predictor.pkl'))
demand_scaler = joblib.load(os.path.join(MODEL_DIR, 'demand_scaler.pkl'))
demand_feature_names = joblib.load(os.path.join(MODEL_DIR, 'demand_feature_names.pkl'))
demand_product_encoder = joblib.load(os.path.join(MODEL_DIR, 'demand_product_encoder.pkl'))
demand_store_encoder = joblib.load(os.path.join(MODEL_DIR, 'demand_store_encoder.pkl'))
demand_season_encoder = joblib.load(os.path.join(MODEL_DIR, 'demand_season_encoder.pkl'))

def build_demand_features(req, demand_row=None):
    """
    Build a DataFrame row for the demand model from a ReturnRequest and (optionally) a demand_row dict.
    """
    # 1. Build the base features dict
    if demand_row:
        features = dict(demand_row)
    else:
        product = req.product
        store = req.store
        features = {
            'product_id': product.id,
            'store_id': store.id,
            'season': 'Spring',  # Default, or fetch from your logic
            'base_price': getattr(product, 'base_price', 1.0),
            'selling_price': getattr(product, 'selling_price', 1.0),
            'cost_price': getattr(product, 'cost_price', 1.0),
            'profit_margin': getattr(product, 'profit_margin', 1.0),
            'quantity_sold': 10,
            'stock_level': 10,
            'restock_threshold': 5,
            'trend_factor': 1.0,
            'customer_rating': 4.0,
            'return_rate': 0.1,
            'order_dow': 1,
            'order_hour_of_day': 12,
            'days_since_prior_order': 7,
            'order_number': 1,
            'add_to_cart_order': 1,
            'needs_restock': 1,
            'reordered': 0,
        }
        for col in demand_feature_names:
            if col not in features:
                features[col] = 0

    # 2. Add derived/encoded features
    features['product_id_encoded'] = demand_product_encoder.transform([features['product_id']])[0]
    features['store_id_encoded'] = demand_store_encoder.transform([features['store_id']])[0]
    features['season_encoded'] = demand_season_encoder.transform([features['season']])[0]
    features['price_ratio'] = features['selling_price'] / max(1, features['base_price'])
    features['stock_ratio'] = features['stock_level'] / max(1, features['restock_threshold'])
    features['profit_per_unit'] = features['selling_price'] - features['cost_price']
    features['is_weekend'] = int(str(features['order_dow']) in ['0', '6'])
    print("Demand features:", features)

    # 3. Build DataFrame and select features
    import pandas as pd
    df = pd.DataFrame([features])
    df = df.fillna(0)
    X = df[demand_feature_names]
    X_scaled = demand_scaler.transform(X)
    return X_scaled

def predict_demand(req, demand_row=None):
    """
    Predict demand score for a ReturnRequest (optionally using a demand_row dict).
    """
    X_scaled = build_demand_features(req, demand_row)
    pred = demand_model.predict(X_scaled)
    return float(pred[0])

# ---------------------------------
# Classification Model: Load Artifacts
# ---------------------------------
classifier_model = joblib.load(os.path.join(MODEL_DIR, 'return_classifier_text.pkl'))
inspector_vect = joblib.load(os.path.join(MODEL_DIR, 'inspector_vect.pkl'))
reason_vect = joblib.load(os.path.join(MODEL_DIR, 'return_reason_vect.pkl'))

# Try to load label encoder for output classes (optional)
try:
    classifier_label_encoder = joblib.load(os.path.join(MODEL_DIR, 'classifier_label_encoder.pkl'))
except Exception:
    classifier_label_encoder = None

def build_classification_features(req):
    inspector_notes = getattr(req, 'inspector_notes', '')
    return_reason = getattr(req, 'return_reason', '')  # Use the text field!
    X_notes = inspector_vect.transform([inspector_notes]).toarray()
    X_reason = reason_vect.transform([return_reason]).toarray()
    import numpy as np
    X = np.concatenate([X_notes, X_reason], axis=1)
    return X

def predict_classification(req):
    X = build_classification_features(req)
    print("X shape:", X.shape)
    print("Model expects:", classifier_model.n_features_in_)
    pred = classifier_model.predict(X)
    prob = classifier_model.predict_proba(X)
    predicted_action_name = pred[0]
    confidence = max(prob[0]) if prob is not None else None
    return predicted_action_name, confidence