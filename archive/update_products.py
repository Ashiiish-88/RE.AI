import pandas as pd
import numpy as np

# Load the current dataset
df = pd.read_csv('data/train.csv')

# Create realistic product mapping based on price ranges and departments
product_mapping = {
    # Low-price items ($2-4) - typically food/household basics
    4: {'name': 'Bananas (per lb)', 'price_range': (2.50, 3.50)},
    15: {'name': 'White Bread', 'price_range': (2.25, 3.25)},
    18: {'name': 'Milk (1 gallon)', 'price_range': (2.75, 3.75)},
    19: {'name': 'Eggs (dozen)', 'price_range': (2.50, 3.50)},
    24: {'name': 'Orange Juice (64oz)', 'price_range': (2.75, 3.75)},
    26: {'name': 'Yogurt (6-pack)', 'price_range': (2.50, 3.50)},
    
    # Medium-low price ($5-10) - packaged foods, household items
    7: {'name': 'Cheerios Cereal', 'price_range': (4.50, 6.50)},
    8: {'name': 'Pasta Sauce (24oz)', 'price_range': (3.50, 5.50)},
    11: {'name': 'Peanut Butter (18oz)', 'price_range': (4.25, 6.25)},
    14: {'name': 'Canned Tomatoes (4-pack)', 'price_range': (4.75, 6.75)},
    25: {'name': 'Dish Soap (32oz)', 'price_range': (5.25, 7.25)},
    
    # Medium price ($10-16) - meat, frozen foods, household products
    10: {'name': 'Ground Beef (1 lb)', 'price_range': (8.50, 12.50)},
    16: {'name': 'Chicken Breast (2 lbs)', 'price_range': (11.50, 15.50)},
    21: {'name': 'Frozen Pizza', 'price_range': (6.50, 9.50)},
    
    # Higher price items ($16+) - premium products
    9: {'name': 'Salmon Fillet (1 lb)', 'price_range': (14.50, 18.50)},
    
    # Add more products to reach 50 total
    27: {'name': 'Apples (3 lb bag)', 'price_range': (3.25, 4.75)},
    28: {'name': 'Ground Turkey (1 lb)', 'price_range': (5.50, 7.50)},
    29: {'name': 'Shampoo (12oz)', 'price_range': (6.25, 9.25)},
    30: {'name': 'Toilet Paper (12-pack)', 'price_range': (8.75, 12.75)},
    31: {'name': 'Laundry Detergent (64oz)', 'price_range': (7.50, 11.50)},
    32: {'name': 'Cheese Slices (8oz)', 'price_range': (3.75, 5.75)},
    33: {'name': 'Crackers', 'price_range': (2.25, 4.25)},
    34: {'name': 'Ice Cream (1.5 qt)', 'price_range': (4.50, 7.50)},
    35: {'name': 'Steak (1 lb)', 'price_range': (12.50, 18.50)},
    36: {'name': 'Paper Towels (6-pack)', 'price_range': (6.25, 9.25)},
    37: {'name': 'Olive Oil (16.9oz)', 'price_range': (8.25, 12.25)},
    38: {'name': 'Avocados (4-pack)', 'price_range': (3.50, 5.50)},
    39: {'name': 'Spinach (5oz)', 'price_range': (2.75, 4.25)},
    40: {'name': 'Granola Bars (6-pack)', 'price_range': (4.25, 6.75)},
    41: {'name': 'Coffee (12oz)', 'price_range': (7.50, 11.50)},
    42: {'name': 'Tea Bags (20-count)', 'price_range': (3.25, 5.25)},
    43: {'name': 'Frozen Vegetables (1 lb)', 'price_range': (2.50, 4.50)},
    44: {'name': 'Canned Soup (4-pack)', 'price_range': (5.25, 7.75)},
    45: {'name': 'Chips (family size)', 'price_range': (3.75, 5.75)},
    46: {'name': 'Soda (12-pack)', 'price_range': (4.50, 6.50)},
    47: {'name': 'Energy Drinks (4-pack)', 'price_range': (6.25, 9.25)},
    48: {'name': 'Protein Bars (6-pack)', 'price_range': (8.75, 12.75)},
    49: {'name': 'Vitamins (60-count)', 'price_range': (12.50, 18.50)},
    50: {'name': 'Baby Formula (12.4oz)', 'price_range': (15.75, 22.75)},
    51: {'name': 'Diapers (size 3, 32-count)', 'price_range': (11.25, 16.25)},
    52: {'name': 'Body Wash (18oz)', 'price_range': (4.75, 7.75)},
    53: {'name': 'Conditioner (12oz)', 'price_range': (5.25, 8.25)}
}

print("Creating realistic product names and prices...")

# Get unique products in the dataset
unique_products = df[['product_id', 'product_name']].drop_duplicates().sort_values('product_id')
print(f"Found {len(unique_products)} unique products")

# Create a mapping for all products (fill missing ones with generic names)
for _, row in unique_products.iterrows():
    pid = row['product_id']
    if pid not in product_mapping:
        # Create generic mapping for missing products
        avg_price = df[df['product_id'] == pid]['product_price'].mean()
        if avg_price < 5:
            product_mapping[pid] = {'name': f'Household Item #{pid}', 'price_range': (avg_price*0.8, avg_price*1.2)}
        elif avg_price < 10:
            product_mapping[pid] = {'name': f'Food Item #{pid}', 'price_range': (avg_price*0.8, avg_price*1.2)}
        else:
            product_mapping[pid] = {'name': f'Premium Item #{pid}', 'price_range': (avg_price*0.8, avg_price*1.2)}

print("Updating dataset with realistic product names...")

# Update product names in the dataset
name_mapping = {pid: info['name'] for pid, info in product_mapping.items()}
df['product_name'] = df['product_id'].map(name_mapping)

# Adjust prices to be more realistic while maintaining some variation
np.random.seed(42)  # For reproducible results
for pid, info in product_mapping.items():
    mask = df['product_id'] == pid
    if mask.any():
        min_price, max_price = info['price_range']
        # Create price variation around the range
        n_records = mask.sum()
        new_prices = np.random.uniform(min_price, max_price, n_records)
        df.loc[mask, 'product_price'] = new_prices

# Update cost_price to maintain reasonable profit margins (60-80% of product_price)
df['cost_price'] = df['product_price'] * np.random.uniform(0.6, 0.8, len(df))

# Recalculate profit_margin
df['profit_margin'] = df['product_price'] - df['cost_price'] - df['logistics_cost_per_unit']
df['profit_margin_pct'] = (df['profit_margin'] / df['product_price']) * 100

print("Saving updated dataset...")

# Save the updated dataset
df.to_csv('data/train.csv', index=False)

print("✅ Dataset updated successfully!")
print("\nSample of updated products:")
sample = df.groupby(['product_id', 'product_name']).agg({
    'product_price': 'mean',
    'cost_price': 'mean',
    'profit_margin': 'mean'
}).round(2).head(10)
print(sample)
