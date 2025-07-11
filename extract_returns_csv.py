import pandas as pd
import uuid

INPUT_FILE = 'data/product_classification_dataset_realistic_with_id_and_name.csv'
OUTPUT_FILE = 'data/returns.csv'

# Optionally, set STORE_ID_COLUMN if available in your dataset
STORE_ID_COLUMN = None  # e.g., 'store_id' if present

def generate_returns(input_file, output_file, store_id_column=None):
    df = pd.read_csv(input_file)
    # Add a UUID for each row
    df['return_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    # Add a placeholder date if not present
    if 'date' not in df.columns:
        df['date'] = ''
    # Select columns for returns.csv
    cols = ['return_id', 'product_id', 'inspector_notes', 'return_reason', 'classification', 'date']
    if store_id_column and store_id_column in df.columns:
        cols.append(store_id_column)
    # Reorder columns if store_id is present
    returns = df[cols] if store_id_column else df[cols]
    returns.to_csv(output_file, index=False)
    print(f"Generated {len(returns)} returns in {output_file}")

if __name__ == "__main__":
    generate_returns(INPUT_FILE, OUTPUT_FILE, STORE_ID_COLUMN)
