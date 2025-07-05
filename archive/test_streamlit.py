import streamlit as st
import pandas as pd
import numpy as np

st.title("🛒 Test Streamlit App")
st.write("Hello! This is a test to see if Streamlit is working.")

# Test data loading
try:
    df = pd.read_csv('data/train.csv')
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    st.write("First few rows:")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Error loading data: {e}")

# Test model loading
try:
    from train_models import DemandPredictionSystem
    system = DemandPredictionSystem()
    system.load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    import traceback
    st.code(traceback.format_exc())
