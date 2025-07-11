import streamlit as st
import pandas as pd
import numpy as np
import os
from return_classification_model import ProductReturnClassifier
from demand_prediction_model import DemandPredictor
from optimal_restock_location_model import OptimalRestockLocationModel

st.set_page_config(page_title="RE.AI Model Tester", layout="wide")
st.title("RE.AI Model Testing Dashboard")

# Load models
@st.cache_resource
def load_models():
    classifier = ProductReturnClassifier()
    classifier.load_model()
    predictor = DemandPredictor()
    predictor.load_model()
    optimizer = OptimalRestockLocationModel()
    optimizer.load_model()
    return classifier, predictor, optimizer

classifier, predictor, optimizer = load_models()

st.sidebar.header("Choose Model to Test")
model_choice = st.sidebar.radio("Select a model:", [
    "Return Classification", "Demand Prediction", "Optimal Restock Location"
])

if model_choice == "Return Classification":
    st.header("End-to-End Workflow: Return → Demand → Restock Location")
    st.write("Input only product_id, return_reason, and store_id. All other fields are auto-filled from the dataset. Date is set to today.")
    # Load full dataset
    full_data = pd.read_csv("data/product_classification_dataset_realistic_with_id_and_name.csv")
    # Unique product_id and product_name mapping
    product_map = full_data[['product_id', 'product_name']].drop_duplicates().sort_values('product_id')
    product_id = st.selectbox("Select Product ID", product_map['product_id'].unique())
    product_name = product_map[product_map['product_id'] == product_id]['product_name'].values[0]
    # All return reasons for this product_id
    reasons = full_data[full_data['product_id'] == product_id]['return_reason'].unique()
    return_reason = st.selectbox("Select Return Reason", reasons)
    # Inspector notes: always allow user to write anything
    inspector_notes = st.text_area("Inspector Notes", "")
    # Store selection (from train.csv)
    demand_data = pd.read_csv("data/train.csv")
    store_ids = demand_data['store_id'].unique() if 'store_id' in demand_data.columns else []
    store_id = st.selectbox("Select Store ID", store_ids)
    # Today's date
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    # Build row for prediction
    row_dict = {
        'product_id': product_id,
        'product_name': product_name,
        'inspector_notes': inspector_notes,
        'return_reason': return_reason,
        'store_id': store_id,
        'date': today
    }
    row = pd.DataFrame([row_dict])
    st.write("**Auto-filled Row for Prediction:**", row)
    pred, prob = classifier.predict(row)
    st.write(f"**Step 1 Prediction:** {pred[0]}")
    st.write(f"**Probabilities:** {dict(zip(classifier.model.classes_, np.round(prob[0], 3)))}")

    if pred[0] == 'restock':
        st.success("This return is suitable for restocking. Proceeding to demand prediction and optimal location...")
        # Try to find a matching row in train.csv for demand/location models
        demand_row = demand_data[(demand_data['product_id'] == product_id) & (demand_data['store_id'] == store_id)]
        if demand_row.empty:
            demand_row = demand_data[demand_data['product_id'] == product_id]
        if demand_row.empty:
            demand_row = demand_data.head(1)
        st.write("\n---\n**Step 2: Demand Prediction**")
        st.write("Sample Demand Data:", demand_row)
        demand_pred = predictor.predict(demand_row)
        st.write(f"**Predicted Demand:** {demand_pred[0]:.2f}")

        st.write("\n---\n**Step 3: Optimal Restock Location**")
        # For location, use all rows for this product_id if possible
        location_rows = demand_data[demand_data['product_id'] == product_id]
        if location_rows.empty:
            location_rows = demand_data.head(10)
        top_n = st.slider("Top N Locations", min_value=1, max_value=5, value=3)
        results = optimizer.predict_optimal_locations(location_rows, top_n=top_n)
        st.write("**Top Locations:**")
        st.dataframe(results[[col for col in results.columns if col in ["product_id", "store_name", "optimal_probability"]]].head(10))
    else:
        st.info("This return is not suitable for restocking. Demand and location prediction skipped.")

elif model_choice == "Demand Prediction":
    st.header("Demand Prediction Model")
    st.write("Predicts demand for a product at a store.")
    demand_data = pd.read_csv("data/train.csv").head(10)
    st.write("### Sample Data", demand_data)
    idx = st.number_input("Select row to test", min_value=0, max_value=len(demand_data)-1, value=0)
    row = demand_data.iloc[[idx]]
    pred = predictor.predict(row)
    st.write(f"**Predicted Demand:** {pred[0]:.2f}")

elif model_choice == "Optimal Restock Location":
    st.header("Optimal Restock Location Model")
    st.write("Predicts the best store(s) to restock a product.")
    demand_data = pd.read_csv("data/train.csv").head(20)
    st.write("### Sample Data", demand_data)
    top_n = st.slider("Top N Locations", min_value=1, max_value=5, value=3)
    results = optimizer.predict_optimal_locations(demand_data, top_n=top_n)
    st.write("**Top Locations:**")
    st.dataframe(results[[col for col in results.columns if col in ["product_id", "store_name", "optimal_probability"]]].head(10))

st.sidebar.markdown("---")
st.sidebar.info("RE.AI Model Testing App\n\nSelect a model and test with sample data.")
