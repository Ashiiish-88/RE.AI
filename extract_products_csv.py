import pandas as pd

# Input and output file paths
INPUT_FILE = 'data/product_classification_dataset_realistic_with_id_and_name.csv'
OUTPUT_FILE = 'data/products.csv'

def extract_unique_products(input_file, output_file):
    df = pd.read_csv(input_file)
    products = df[['product_id', 'product_name']].drop_duplicates().sort_values('product_id')
    products.to_csv(output_file, index=False)
    print(f"Extracted {len(products)} unique products to {output_file}")

if __name__ == "__main__":
    extract_unique_products(INPUT_FILE, OUTPUT_FILE)
