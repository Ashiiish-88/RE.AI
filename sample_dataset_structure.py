"""
REAL PRODUCT DEMAND DATASETS - NO SYNTHETIC DATA

This file contains information about genuine, real-world datasets 
that contain actual product demand data from real retailers.
"""

# =============================================================================
# 1. WALMART SALES DATASET (KAGGLE) - REAL RETAIL DATA
# =============================================================================
"""
Dataset: Walmart Store Sales Forecasting
Source: Kaggle (https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-dataset)
Type: REAL historical sales data from Walmart stores

ALL FIELDS/COLUMNS:
1. Store (int): Store number (1-45)
2. Date (date): Week ending date (YYYY-MM-DD)
3. Weekly_Sales (float): Actual sales for that week (in USD)
4. Holiday_Flag (int): 1 if holiday week, 0 otherwise
5. Temperature (float): Average temperature in the region (Fahrenheit)
6. Fuel_Price (float): Cost of fuel in the region (per gallon)
7. CPI (float): Consumer Price Index
8. Unemployment (float): Unemployment rate in the region (percentage)

Additional Files in Dataset:
- stores.csv: Store, Type, Size
- features.csv: Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday

Sample Data Points (REAL):
Store 1, 2010-02-05, $1,643,690.90, 0, 42.31, 2.572, 211.096, 8.106
Store 1, 2010-02-12, $1,641,957.44, 1, 38.51, 2.548, 211.242, 8.106
Store 1, 2010-02-19, $1,611,968.17, 0, 39.93, 2.514, 211.289, 8.106

Products: Department-wise sales (not individual SKUs but product categories)
Time Period: 2010-2012 (3 years)
Records: 421,570 real transactions
"""

# =============================================================================
# 2. INSTACART MARKET BASKET ANALYSIS - REAL GROCERY DATA
# =============================================================================
"""
Dataset: Instacart Market Basket Analysis
Source: Kaggle (https://www.kaggle.com/c/instacart-market-basket-analysis)
Type: REAL customer purchase data from Instacart

ALL FIELDS/COLUMNS:

File 1: orders.csv
1. order_id (int): Unique order identifier
2. user_id (int): Unique customer identifier
3. eval_set (str): 'prior', 'train', or 'test'
4. order_number (int): Sequence number for this user (1=first order)
5. order_dow (int): Day of week order was placed (0=Sunday, 6=Saturday)
6. order_hour_of_day (int): Hour of day order was placed (0-23)
7. days_since_prior_order (float): Days since customer's last order

File 2: products.csv
1. product_id (int): Unique product identifier
2. product_name (str): REAL product names (e.g., "Banana", "Bag of Organic Bananas")
3. aisle_id (int): Aisle identifier
4. department_id (int): Department identifier

File 3: aisles.csv
1. aisle_id (int): Aisle identifier
2. aisle (str): Aisle name (e.g., "fresh fruits", "dairy eggs")

File 4: departments.csv
1. department_id (int): Department identifier
2. department (str): Department name (e.g., "produce", "dairy eggs")

File 5: order_products_prior.csv
1. order_id (int): Order identifier
2. product_id (int): Product identifier
3. add_to_cart_order (int): Sequence in which product was added to cart
4. reordered (int): 1 if product was reordered by customer, 0 otherwise

File 6: order_products_train.csv
1. order_id (int): Order identifier
2. product_id (int): Product identifier
3. add_to_cart_order (int): Sequence in which product was added to cart
4. reordered (int): 1 if product was reordered by customer, 0 otherwise

Sample REAL Products with Demand Patterns:
1. Banana (product_id: 24852) - High reorder rate, consistent demand
2. Bag of Organic Bananas (product_id: 13176) - Growing demand
3. Organic Strawberries (product_id: 21137) - Seasonal demand
4. Organic Baby Spinach (product_id: 21903) - Steady demand
5. Organic Hass Avocado (product_id: 47209) - High demand

Dataset Size: 3M+ orders, 200k+ users, 50k+ real products
Time Period: Real historical data
"""

# =============================================================================
# 3. ONLINE RETAIL DATASET (UCI) - REAL E-COMMERCE DATA
# =============================================================================
"""
Dataset: Online Retail Dataset
Source: UCI Machine Learning Repository
Type: REAL transactional data from UK-based online retail

ALL FIELDS/COLUMNS:
1. InvoiceNo (str): Invoice number (6-digit number or starts with 'c' for cancellation)
2. StockCode (str): Product code (5-digit number or special codes)
3. Description (str): REAL product description
4. Quantity (int): Quantity purchased (negative for returns)
5. InvoiceDate (datetime): Date and time of purchase (MM/DD/YYYY HH:MM)
6. UnitPrice (float): Unit price in GBP (British Pounds)
7. CustomerID (float): Customer identifier (some missing values)
8. Country (str): Customer country (mainly UK, but includes other countries)

Sample REAL Products with StockCodes:
- WHITE HANGING HEART T-LIGHT HOLDER (StockCode: 85123A)
- REGENCY CAKESTAND 3 TIER (StockCode: 23084)
- JUMBO BAG RED RETROSPOT (StockCode: 47566)
- PARTY BUNTING (StockCode: 84879)
- PINK REGENCY TEACUP AND SAUCER (StockCode: 21730)
- GREEN REGENCY TEACUP AND SAUCER (StockCode: 21731)
- RED WOOLLY HOTTIE WHITE HEART (StockCode: 22752)
- BLUE POLKADOT BUNTING (StockCode: 21774)

Dataset Size: 1M+ transactions
Products: 4,000+ unique real products
Time Period: 2010-2011 (1 year)
Countries: 38 different countries
"""

# =============================================================================
# 4. ROSSMANN STORE SALES - REAL PHARMACY/DRUGSTORE DATA
# =============================================================================
"""
Dataset: Rossmann Store Sales
Source: Kaggle (https://www.kaggle.com/c/rossmann-store-sales)
Type: REAL sales data from Rossmann drugstore chain

ALL FIELDS/COLUMNS:

File 1: train.csv
1. Store (int): Store identifier (1-1115)
2. DayOfWeek (int): Day of week (1=Monday, 7=Sunday)
3. Date (date): Date of sales (YYYY-MM-DD)
4. Sales (float): Sales amount (in EUR)
5. Customers (int): Number of customers on that day
6. Open (int): Whether store was open (1=open, 0=closed)
7. Promo (int): Whether store was running a promotion (1=yes, 0=no)
8. StateHoliday (str): State holiday ('a'=public holiday, 'b'=Easter, 'c'=Christmas, '0'=none)
9. SchoolHoliday (int): School holiday indicator (1=yes, 0=no)

File 2: store.csv
1. Store (int): Store identifier
2. StoreType (str): Store type (a, b, c, d)
3. Assortment (str): Assortment level (a=basic, b=extra, c=extended)
4. CompetitionDistance (float): Distance to nearest competitor (meters)
5. CompetitionOpenSinceMonth (float): Month competitor opened
6. CompetitionOpenSinceYear (float): Year competitor opened
7. Promo2 (int): Participating in continuing promotion (1=yes, 0=no)
8. Promo2SinceWeek (float): Week when Promo2 started
9. Promo2SinceYear (float): Year when Promo2 started
10. PromoInterval (str): Consecutive intervals Promo2 is started

Products: Drugstore/pharmacy products (real retail chain)
Stores: 1,115 real stores across Germany
Time Period: 2013-2015 (2.5 years)
Records: 1M+ real sales records
"""

# =============================================================================
# 5. FAVORITA GROCERY SALES - REAL SUPERMARKET DATA
# =============================================================================
"""
Dataset: Corporación Favorita Grocery Sales Forecasting
Source: Kaggle (https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
Type: REAL sales data from Favorita supermarket chain in Ecuador

ALL FIELDS/COLUMNS:

File 1: train.csv
1. id (int): Unique identifier
2. date (date): Date of sales (YYYY-MM-DD)
3. store_nbr (int): Store number (1-54)
4. family (str): Product family (e.g., AUTOMOTIVE, BABY CARE, BEAUTY)
5. sales (float): Sales amount (target variable)
6. onpromotion (int): Number of items on promotion

File 2: test.csv
1. id (int): Unique identifier
2. date (date): Date for prediction
3. store_nbr (int): Store number
4. family (str): Product family
5. onpromotion (int): Number of items on promotion

File 3: stores.csv
1. store_nbr (int): Store number
2. city (str): City name
3. state (str): State name
4. type (str): Store type (A, B, C, D, E)
5. cluster (int): Store cluster (1-17)

File 4: items.csv
1. item_nbr (int): Item number
2. family (str): Product family
3. class (int): Item class
4. perishable (int): 1 if perishable, 0 otherwise

File 5: transactions.csv
1. date (date): Date
2. store_nbr (int): Store number
3. transactions (int): Number of transactions

File 6: oil.csv
1. date (date): Date
2. dcoilwtico (float): Oil price (daily)

File 7: holidays_events.csv
1. date (date): Date
2. type (str): Holiday type (Holiday, Additional, Transfer, etc.)
3. locale (str): Locale (National, Regional, Local)
4. locale_name (str): Name of locale
5. description (str): Holiday description
6. transferred (bool): Whether holiday was transferred

Sample REAL Product Families (33 total):
- GROCERY I (basic groceries)
- BEVERAGES (drinks)
- PRODUCE (fruits/vegetables)
- CLEANING (household cleaning)
- DAIRY (milk, cheese, yogurt)
- BREAD/BAKERY (baked goods)
- MEATS (meat products)
- PREPARED FOODS (ready-to-eat)
- DELI (delicatessen)
- POULTRY (chicken, turkey)
- SEAFOOD (fish, shellfish)
- PERSONAL CARE (hygiene products)
- BABY CARE (baby products)
- AUTOMOTIVE (car accessories)
- BEAUTY (cosmetics)
- FROZEN FOODS (frozen items)
- LIQUOR,WINE,BEER (alcoholic beverages)
- PETS (pet supplies)
- LAWN AND GARDEN (gardening)
- HARDWARE (tools, hardware)
- HOME AND KITCHEN I (kitchen items)
- HOME AND KITCHEN II (home items)
- BOOKS (books, magazines)
- SCHOOL AND OFFICE SUPPLIES (stationery)
- PLAYERS AND ELECTRONICS (electronics)
- CELEBRATION (party supplies)
- LADIESWEAR (women's clothing)
- LINGERIE (underwear)
- MAGAZINES (magazines)
- HOME APPLIANCES (appliances)
- IRRIGATION (irrigation equipment)
- BABY CARE (additional baby items)
- HOME CARE (home cleaning)

Dataset Size: 125M+ records
Stores: 54 real stores across Ecuador
Products: 1,700+ real product families
Time Period: 2013-2017 (4+ years)
"""

# =============================================================================
# COMPREHENSIVE FIELD COMPARISON - ALL DATASETS
# =============================================================================
"""
FIELD COUNT SUMMARY:
1. Walmart Sales: 8 core fields + additional files
2. Instacart: 6 files with 2-4 fields each (total 19 unique fields)
3. Online Retail: 8 fields
4. Rossmann: 2 files with 9 + 10 fields (total 19 unique fields)
5. Favorita: 7 files with 3-7 fields each (total 25+ unique fields)

PRODUCT GRANULARITY:
1. Walmart: Department-level (not individual products)
2. Instacart: Individual product level (50k+ products)
3. Online Retail: Individual product level (4k+ products)
4. Rossmann: Store-level sales (not product-specific)
5. Favorita: Product family level (33 families)

DEMAND PREDICTION CAPABILITY:
1. Walmart: ⭐⭐⭐⭐ (Great for store-level demand)
2. Instacart: ⭐⭐⭐⭐⭐ (Perfect for product-level demand)
3. Online Retail: ⭐⭐⭐⭐ (Good for product-level demand)
4. Rossmann: ⭐⭐⭐ (Good for store-level demand)
5. Favorita: ⭐⭐⭐⭐ (Great for category-level demand)

EXTERNAL FACTORS:
1. Walmart: Temperature, Fuel Price, CPI, Unemployment
2. Instacart: Time patterns (day, hour)
3. Online Retail: Geographic (country)
4. Rossmann: Competition, Promotions, Holidays
5. Favorita: Oil prices, Holidays, Promotions

BEST FOR WALMART SPARKATHON:
Ranking: 1. Walmart > 2. Instacart > 3. Favorita > 4. Online Retail > 5. Rossmann
"""

# =============================================================================
# DETAILED FIELD TYPES AND RANGES
# =============================================================================
"""
WALMART DATASET FIELD DETAILS:
- Store: Integer (1-45)
- Date: Date (2010-02-05 to 2012-10-26)
- Weekly_Sales: Float ($209.99 to $3,818,686.45)
- Holiday_Flag: Binary (0, 1)
- Temperature: Float (-2.06°F to 100.14°F)
- Fuel_Price: Float ($2.472 to $4.468 per gallon)
- CPI: Float (126.064 to 227.233)
- Unemployment: Float (3.879% to 14.313%)

INSTACART DATASET FIELD DETAILS:
- order_id: Integer (1 to 3,421,083)
- user_id: Integer (1 to 206,209)
- product_id: Integer (1 to 49,688)
- product_name: String (Real product names)
- order_dow: Integer (0-6, Sunday=0)
- order_hour_of_day: Integer (0-23)
- days_since_prior_order: Float (0.0 to 30.0)
- reordered: Binary (0, 1)

ONLINE RETAIL FIELD DETAILS:
- InvoiceNo: String (6 digits or 'C' prefix)
- StockCode: String (5 digits or special codes)
- Description: String (Real product descriptions)
- Quantity: Integer (-80995 to 80995)
- InvoiceDate: DateTime (2010-12-01 to 2011-12-09)
- UnitPrice: Float (0.0 to 38970.0 GBP)
- CustomerID: Float (12346.0 to 18287.0)
- Country: String (38 countries)

ROSSMANN FIELD DETAILS:
- Store: Integer (1-1115)
- Sales: Float (0.0 to 41551.0 EUR)
- Customers: Integer (0 to 7388)
- DayOfWeek: Integer (1-7)
- Date: Date (2013-01-01 to 2015-07-31)
- Open: Binary (0, 1)
- Promo: Binary (0, 1)
- StateHoliday: String ('0', 'a', 'b', 'c')
- SchoolHoliday: Binary (0, 1)

FAVORITA FIELD DETAILS:
- store_nbr: Integer (1-54)
- family: String (33 product families)
- sales: Float (0.0 to 264834.0)
- onpromotion: Integer (0 to 741)
- date: Date (2013-01-01 to 2017-08-31)
- transactions: Integer (5 to 8359)
- dcoilwtico: Float (Oil price, 26.19 to 110.62)
"""

# =============================================================================
# SAMPLE CODE TO LOAD REAL DATASET
# =============================================================================
"""
import pandas as pd

# Load real Walmart sales data
df = pd.read_csv('walmart_sales_data.csv')

# Display real product demand information
print("Real Product Categories and Demand:")
print(df.groupby('Store')['Weekly_Sales'].agg(['count', 'mean', 'sum']).head())

# Real seasonal demand patterns
print("\nReal Seasonal Demand Patterns:")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
monthly_demand = df.groupby('Month')['Weekly_Sales'].mean()
print(monthly_demand)

# Real holiday impact on demand
print("\nReal Holiday Impact on Demand:")
holiday_impact = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()
print(holiday_impact)
"""