import csv

# Load product_id to product_name mapping
id_to_name = {}
with open('product_id_name_map.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        id_to_name[row['product_id']] = row['product_name']

# Read the dataset with product_id
with open('data/product_classification_dataset_realistic_with_id.csv', newline='') as infile:
    reader = list(csv.reader(infile))
    header = reader[0]
    rows = reader[1:]

# Add product_name column after product_id
new_header = header[:]
if 'product_name' not in new_header:
    new_header.insert(1, 'product_name')

new_rows = []
for row in rows:
    product_id = row[0]
    product_name = id_to_name.get(product_id, 'UNKNOWN')
    new_row = row[:]
    new_row.insert(1, product_name)
    new_rows.append(new_row)

# Write the new dataset with product_name
with open('data/product_classification_dataset_realistic_with_id_and_name.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(new_header)
    writer.writerows(new_rows)

print('Done: data/product_classification_dataset_realistic_with_id_and_name.csv created.')
