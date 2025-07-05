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

# Load the data
print("Loading data...")
df = pd.read_csv('data/train.csv')

print(f"Dataset shape: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique values in key columns:")
print(f"Unique products: {df['product_id'].nunique()}")
print(f"Unique store locations: {df['store_location'].nunique()}")
print(f"Unique distribution centers: {df['distribution_center'].nunique()}")

print("\nStore locations:")
print(df['store_location'].unique())

print("\nDistribution centers:")
print(df['distribution_center'].unique())

print("\nTarget variable statistics:")
print(df['demand'].describe())

# Check date range
df['date'] = pd.to_datetime(df['date'])
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
