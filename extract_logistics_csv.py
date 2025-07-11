import pandas as pd
import os

INPUT_FILE = 'data/train.csv'  # Change this if your logistics data is in a different file
OUTPUT_FILE = 'data/logistics.csv'

COLUMNS = [
    'logistics_id', 'product_id', 'store_id', 'logistics_cost_per_mile', 'logistics_cost_per_km'
]

def extract_logistics(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        # Create a template for manual entry
        template = pd.DataFrame(columns=COLUMNS)
        template.to_csv(output_file, index=False)
        print(f"Created template {output_file} for manual entry.")
        return
    df = pd.read_csv(input_file)
    # Add logistics_id if not present
    if 'logistics_id' not in df.columns:
        df['logistics_id'] = range(1, len(df) + 1)
    # Only keep columns that exist in the file or fill missing ones with blanks
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ''
    logistics = df[COLUMNS]
    logistics.to_csv(output_file, index=False)
    print(f"Generated {len(logistics)} rows in {output_file}")

if __name__ == "__main__":
    extract_logistics(INPUT_FILE, OUTPUT_FILE)
