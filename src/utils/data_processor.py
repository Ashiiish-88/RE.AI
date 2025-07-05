"""
Data Processing Utilities
=========================

Utilities for data loading, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class DataProcessor:
    """
    Data processing utilities for the Walmart demand prediction system.
    
    Handles:
    - Data loading and validation
    - Store distance calculations
    - Product information management
    - Data quality checks
    """
    
    @staticmethod
    def load_train_data(filepath: str = "data/train.csv") -> pd.DataFrame:
        """
        Load and validate training data.
        
        Args:
            filepath (str): Path to the training CSV file
            
        Returns:
            pd.DataFrame: Cleaned training data
        """
        print(f"📊 Loading training data from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Basic validation
        required_columns = [
            'product_id', 'product_name', 'aisle', 'department', 
            'product_price', 'store_location', 'distribution_center',
            'date', 'demand'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Data quality checks
        print(f"  📈 Dataset shape: {df.shape}")
        print(f"  🏪 Unique stores: {df['store_location'].nunique()}")
        print(f"  📦 Unique products: {df['product_id'].nunique()}")
        print(f"  📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            print(f"  ⚠️ Missing values found:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    @staticmethod
    def load_store_distances(filepath: str = "data/store_distances.csv") -> pd.DataFrame:
        """
        Load store distance matrix.
        
        Args:
            filepath (str): Path to the store distances CSV
            
        Returns:
            pd.DataFrame: Store distance matrix
        """
        print(f"🗺️ Loading store distances from {filepath}...")
        
        try:
            distances_df = pd.read_csv(filepath)
            print(f"  📏 Distance matrix: {distances_df.shape}")
            return distances_df
        except FileNotFoundError:
            print(f"  ⚠️ Distance file not found. Creating default distances...")
            return DataProcessor.create_default_distances()
    
    @staticmethod
    def create_default_distances() -> pd.DataFrame:
        """
        Create a default distance matrix for demo purposes.
        
        Returns:
            pd.DataFrame: Default distance matrix
        """
        # Sample store locations
        stores = [
            "Alabama_Montgomery", "Arizona_Phoenix", "Arkansas_Little_Rock",
            "California_Los_Angeles", "Colorado_Denver", "Connecticut_Hartford",
            "Florida_Miami", "Georgia_Atlanta", "Illinois_Chicago",
            "Texas_Houston", "New_York_New_York", "Nevada_Las_Vegas"
        ]
        
        distances = []
        for origin in stores:
            for destination in stores:
                if origin != destination:
                    # Random distances between 50-500 miles
                    distance = np.random.uniform(50, 500)
                    distances.append({
                        'origin': origin,
                        'destination': destination,
                        'distance_miles': round(distance, 1)
                    })
        
        return pd.DataFrame(distances)
    
    @staticmethod
    def get_unique_locations(df: pd.DataFrame) -> List[str]:
        """
        Get list of unique store locations.
        
        Args:
            df (pd.DataFrame): Training dataframe
            
        Returns:
            List[str]: Sorted list of unique store locations
        """
        return sorted(df['store_location'].unique().tolist())
    
    @staticmethod
    def get_unique_products(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get unique products with their details.
        
        Args:
            df (pd.DataFrame): Training dataframe
            
        Returns:
            pd.DataFrame: Unique products with metadata
        """
        return df.groupby('product_id').agg({
            'product_name': 'first',
            'aisle': 'first',
            'department': 'first',
            'product_price': 'mean',
            'demand': 'mean'
        }).reset_index().sort_values('product_name')
    
    @staticmethod
    def calculate_distance(origin: str, destination: str, distances_df: pd.DataFrame) -> float:
        """
        Calculate distance between two locations.
        
        Args:
            origin (str): Origin location
            destination (str): Destination location
            distances_df (pd.DataFrame): Distance matrix
            
        Returns:
            float: Distance in miles
        """
        if origin == destination:
            return 0.0
        
        distance_row = distances_df[
            (distances_df['origin'] == origin) & 
            (distances_df['destination'] == destination)
        ]
        
        if not distance_row.empty:
            return distance_row['distance_miles'].iloc[0]
        else:
            # Default distance if not found
            return 100.0
    
    @staticmethod
    def analyze_data_quality(df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data quality analysis.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            Dict: Data quality report
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numerical_stats': {},
            'categorical_stats': {}
        }
        
        # Numerical column statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            report['numerical_stats'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'zeros': (df[col] == 0).sum(),
                'negatives': (df[col] < 0).sum()
            }
        
        # Categorical column statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            report['categorical_stats'][col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'most_frequent_count': df[col].value_counts().iloc[0] if not df[col].empty else 0
            }
        
        return report
    
    @staticmethod
    def print_data_summary(df: pd.DataFrame):
        """
        Print a comprehensive data summary.
        
        Args:
            df (pd.DataFrame): Dataset to summarize
        """
        print("📊 Data Summary")
        print("=" * 50)
        
        # Basic info
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            print("\n❌ Missing Values:")
            for col, count in missing[missing > 0].items():
                print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print("\n✅ No missing values")
        
        # Data types
        print(f"\n📋 Data Types:")
        for dtype, cols in df.dtypes.groupby(df.dtypes).items():
            print(f"  {dtype}: {len(cols)} columns")
        
        # Unique values for key columns
        key_columns = ['product_id', 'store_location', 'department', 'aisle']
        print(f"\n🔑 Key Columns:")
        for col in key_columns:
            if col in df.columns:
                print(f"  {col}: {df[col].nunique()} unique values")


# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Load and analyze data
    df = processor.load_train_data()
    processor.print_data_summary(df)
    
    # Load distances
    distances = processor.load_store_distances()
    print(f"\nDistance matrix loaded: {distances.shape}")
