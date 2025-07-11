import csv
import random

# Load product_id and product_name mapping
product_map = []
with open('product_id_name_map.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        product_map.append({'product_id': row['product_id'], 'product_name': row['product_name']})

# Read the original realistic dataset
with open('data/product_classification_dataset_realistic.csv', newline='') as infile:
    reader = list(csv.reader(infile))
    header = reader[0]
    rows = reader[1:]

# Add product_id column to header
new_header = ['product_id'] + header

# Assign a random product_id to each row
new_rows = []
for row in rows:
    product = random.choice(product_map)
    new_row = [product['product_id']] + row
    new_rows.append(new_row)

# Write the new dataset with product_id
with open('data/product_classification_dataset_realistic_with_id.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(new_header)
    writer.writerows(new_rows)

print('Done: data/product_classification_dataset_realistic_with_id.csv created.')
