import pandas as pd
import os

# Input and output file paths
INPUT_FILE = 'data/store_distances.csv'  # Use the correct file with store info
OUTPUT_FILE = 'data/stores.csv'

# Try to extract store_id, store_name, location if present

def extract_stores(input_file, output_file):
    df = pd.read_csv(input_file)
    print('Columns found:', df.columns.tolist())
    required = ['store_id', 'store_name', 'store_location']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}. Extraction aborted.")
        return
    stores = df[required].drop_duplicates().sort_values('store_id')
    stores = stores.rename(columns={'store_location': 'location'})
    stores.to_csv(output_file, index=False)
    print(f"Extracted {len(stores)} unique stores to {output_file}")

if __name__ == "__main__":
    extract_stores(INPUT_FILE, OUTPUT_FILE)
