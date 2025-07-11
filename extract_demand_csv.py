import pandas as pd
import os

INPUT_FILE = 'data/train.csv'  # Change this if your demand data is in a different file
OUTPUT_FILE = 'data/demand.csv'

COLUMNS = [
    'demand_id', 'product_id', 'store_id', 'season', 'base_price', 'selling_price', 'cost_price',
    'profit_margin', 'quantity_sold', 'stock_level', 'restock_threshold', 'trend_factor',
    'customer_rating', 'return_rate', 'order_dow', 'order_hour_of_day', 'days_since_prior_order',
    'order_number', 'add_to_cart_order', 'needs_restock', 'reordered'
]

def extract_demand(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        # Create a template for manual entry
        template = pd.DataFrame(columns=COLUMNS)
        template.to_csv(output_file, index=False)
        print(f"Created template {output_file} for manual entry.")
        return
    df = pd.read_csv(input_file)
    # Add demand_id if not present
    if 'demand_id' not in df.columns:
        df['demand_id'] = range(1, len(df) + 1)
    # Only keep columns that exist in the file or fill missing ones with blanks
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ''
    demand = df[COLUMNS]
    demand.to_csv(output_file, index=False)
    print(f"Generated {len(demand)} rows in {output_file}")

if __name__ == "__main__":
    extract_demand(INPUT_FILE, OUTPUT_FILE)
