# 🚀 RE.AI - Returns & Exchange AI Platform

> **🎯 PRODUCTION READY** - Django web application with advanced ML demand prediction capabilities!

Complete platform featuring Django web application for returns/exchange management and ultra-advanced machine learning models for demand prediction.

## 🧠 **NEW: Advanced ML Models**

Our platform now includes state-of-the-art demand prediction models:

- **🏆 Best Model**: Stacking Ensemble (R² = 0.3655)
- **📊 Features**: 60 ultra-engineered features
- **🚀 Techniques**: GPU acceleration, hyperparameter optimization, advanced ensembles
- **💾 Model Size**: 1.4 GB optimized weights

```python
# 🔥 INSTANT PREDICTION - Try this now!
from ml_models import DemandPredictor
predictor = DemandPredictor()
predictions = predictor.predict(your_data)
```

## 📡 **QUICK REMOTE CONNECTION**

```bash
# 🔥 INSTANT TEST - Run this now!
python -c "from supabase_api_client import SupabaseClient; SupabaseClient().test_connections()"
```

**VS Code PostgreSQL Extension Setup:**
```
Host: aws-0-ap-south-1.pooler.supabase.com
User: postgres.yujmrarqlotmdvkhygmh  
Password: admin@2027
Database: postgres | Port: 5432 | SSL: require
```

---

## 🌟 Remote Supabase Connection (HIGHLIGHTED)

### 🔗 Connect to Your Remote Supabase Database

This client provides **multiple ways** to connect to your remote Supabase database:

#### 🎯 **Method 1: VS Code PostgreSQL Extension (RECOMMENDED)**
1. **Install Extension**: Search "PostgreSQL" in VS Code Extensions
2. **Add Connection** with these details:
   ```
   🔧 Connection Details:
   ├── Host: aws-0-ap-south-1.pooler.supabase.com
   ├── Port: 5432
   ├── Database: postgres
   ├── Username: postgres.yujmrarqlotmdvkhygmh
   ├── Password: admin@2027
   └── SSL: require
   ```
3. **✅ Result**: Full SQL access, query editor, table browser

#### 🎯 **Method 2: Python REST API (ALWAYS WORKS)**
```python
from supabase_api_client import SupabaseClient
client = SupabaseClient()
data = client.query_table('returns_customer')  # Get all customers
```

#### 🎯 **Method 3: Direct SQL Queries**
```python
client = SupabaseClient()
df = client.query_sql('SELECT * FROM returns_product LIMIT 10')
```

#### 🎯 **Method 4: pgAdmin 4**
Use the same connection details as VS Code extension above.

### 🔍 **Why Connection Pooler Works**
- **Direct hostname**: `db.yujmrarqlotmdvkhygmh.supabase.co` → IPv6 only (fails on most networks)
- **Connection pooler**: `aws-0-ap-south-1.pooler.supabase.com` → IPv4 compatible (works everywhere)

---

## 📋 Features

- ✅ **REST API Access** - Always reliable, works everywhere
- ✅ **PostgreSQL Connection Pooler** - High-performance IPv4 connection
- ✅ **Direct PostgreSQL with IPv6** - Native database access
- ✅ **Auto-fallback** - Automatically tries best available method
- ✅ **Pandas Integration** - Query results as DataFrames
- ✅ **Full CRUD Operations** - Create, Read, Update, Delete via REST API

## 🔧 Installation

```bash
# Install from requirements.txt (minimal, production-ready)
pip install -r requirements.txt

# Or install individually
pip install Django psycopg2-binary python-dotenv dj-database-url requests pandas

# For virtual environment
source .venv/bin/activate
pip install -r requirements.txt
```

**📦 Minimal Dependencies**: This project uses only essential libraries - no bloat!

## ⚙️ Configuration

Your `.env` file should contain:

```env
# Supabase Configuration
supabase_url = 'https://yujmrarqlotmdvkhygmh.supabase.co'
supabase_anon_key = 'your_anon_key'
supabase_service_key = 'your_service_key'

# Connection Pooler Variables (Working for VS Code & pgAdmin)
pooler_host=aws-0-ap-south-1.pooler.supabase.com
pooler_user=postgres.yujmrarqlotmdvkhygmh
pooler_password=admin@2027
pooler_port=5432
pooler_dbname=postgres
```

## ⚡ **INSTANT CONNECTION TEST**

Test your remote connection immediately:

```python
# Test all connection methods
from supabase_api_client import SupabaseClient
client = SupabaseClient()
client.test_connections()
```

**Expected Output:**
```
🧪 Testing Supabase Connections
==================================================
🌐 REST API: ✅ Working
🔗 Direct IPv6: ❌ Failed (No route to host)
🔗 Direct Hostname: ❌ Failed (DNS resolution)  
🔗 Connection Pooler: ✅ Working
```

**✅ Success**: If you see "REST API: ✅ Working" and "Connection Pooler: ✅ Working", you're connected!

## 🚀 Quick Start

```python
from supabase_api_client import SupabaseClient

# Initialize client
client = SupabaseClient()

# Test all connection methods
client.test_connections()

# Get all customers using REST API
customers = client.query_table('returns_customer')
print(customers)

# Execute SQL query using PostgreSQL
df = client.query_sql('SELECT * FROM returns_product LIMIT 10')
print(df)

# Insert new record
new_customer = client.insert_record('returns_customer', {
    'name': 'Jane Smith',
    'email': 'jane@example.com',
    'phone': '+1-555-0199',
    'address': '456 Oak Street, NY'
})
```

## 🌐 **Remote Connection Architecture**

```
Your Computer               Supabase Cloud
─────────────────          ─────────────────────────
                          
🖥️  VS Code               ☁️  Connection Pooler
🖥️  pgAdmin          ──→  📡  aws-0-ap-south-1.pooler.supabase.com
🖥️  Python Client         │   ├── IPv4: 3.111.105.85 ✅
                          │   ├── Port: 5432
                          │   ├── User: postgres.yujmrarqlotmdvkhygmh  
                          │   └── SSL: Required
                          │
                          ├── 🗄️  PostgreSQL Database
                          │   └── 21 Tables (Django + Returns)
                          │
                          └── 🌐  REST API
                              ├── https://yujmrarqlotmdvkhygmh.supabase.co
                              └── Full CRUD operations
                              
❌ Direct Connection (IPv6 only - often fails)
📡 db.yujmrarqlotmdvkhygmh.supabase.co
└── IPv6: 2406:da1a:6b0:f602:f8ce:c474:821a:215d
```

## 📊 Connection Methods

### 🌐 REST API (Recommended)
- **Always works** - No network configuration needed
- **Full CRUD operations** - Insert, update, delete, query
- **Automatic JSON handling** - Returns Python objects/DataFrames
- **Rate limited** - Suitable for most applications

```python
# REST API methods
tables = client.get_tables()
data = client.query_table('returns_customer', limit=50)
result = client.insert_record('returns_customer', {...})
client.update_record('returns_customer', record_id=1, data={...})
client.delete_record('returns_customer', record_id=1)
```

### 🔗 PostgreSQL Connection Pooler (High Performance)
- **IPv4 connection** - `aws-0-ap-south-1.pooler.supabase.com`
- **Optimized for performance** - Connection pooling
- **Works with pgAdmin, VS Code** - Standard PostgreSQL tools
- **Full SQL support** - Complex queries, transactions

```python
# SQL methods
df = client.query_sql('SELECT * FROM returns_customer WHERE name LIKE %s', ['%John%'])
conn = client.get_postgresql_connection('pooler')
```

### 🔗 Direct PostgreSQL (IPv6)
- **Direct database access** - `2406:da1a:6b0:f602:f8ce:c474:821a:215d`
- **IPv6 only** - Requires IPv6 network support
- **Low latency** - Direct connection to database
- **May not work** - Depends on network configuration

## 🎯 Available Tables

### 🔐 Django System Tables (10)
- `auth_group`, `auth_group_permissions`, `auth_permission`
- `auth_user`, `auth_user_groups`, `auth_user_user_permissions`
- `django_admin_log`, `django_content_type`, `django_migrations`, `django_session`

### 📦 Returns Management Tables (11)
- `returns_attachment` - File attachments for returns
- `returns_category` - Product categories
- `returns_customer` - Customer information
- `returns_product` - Product catalog
- `returns_productinstance` - Individual product instances
- `returns_returnaction` - Available return actions (refund, replace, etc.)
- `returns_returnimage` - Images associated with returns
- `returns_returnreason` - Predefined return reasons
- `returns_returnrequest` - Main return requests
- `returns_returnstatushistory` - Status change history
- `returns_store` - Store locations

## 📝 Example Queries

### Basic Data Retrieval
```python
# Get all customers
customers = client.query_table('returns_customer')

# Get recent return requests
recent_returns = client.query_table('returns_returnrequest', 
                                   limit=20, 
                                   filters={'order': 'created_at.desc'})

# Get products by brand
apple_products = client.query_table('returns_product', 
                                   filters={'brand': 'eq.Apple'})
```

### Advanced SQL Queries
```python
# Complex join query
query = """
SELECT 
    rr.id,
    c.name as customer_name,
    p.name as product_name,
    rr.status,
    rr.created_at
FROM returns_returnrequest rr
JOIN returns_customer c ON rr.customer_id = c.id
JOIN returns_product p ON rr.product_id = p.id
WHERE rr.created_at >= '2025-01-01'
ORDER BY rr.created_at DESC
LIMIT 10
"""
df = client.query_sql(query)
```

### Data Analysis
```python
# Return status distribution
status_query = """
SELECT status, COUNT(*) as count 
FROM returns_returnrequest 
GROUP BY status 
ORDER BY count DESC
"""
status_df = client.query_sql(status_query)

# Monthly return trends
monthly_query = """
SELECT 
    DATE_TRUNC('month', created_at) as month,
    COUNT(*) as return_count,
    AVG(confidence_score) as avg_confidence
FROM returns_returnrequest 
WHERE created_at >= '2024-01-01'
GROUP BY month 
ORDER BY month
"""
trends_df = client.query_sql(monthly_query)
```

## 🔍 Troubleshooting Remote Connections

### 🚨 **Quick Fix Guide**

| Problem | Solution | Status |
|---------|----------|--------|
| **❌ "No route to host" (IPv6)** | Use connection pooler instead | ✅ **SOLVED** |
| **❌ "nodename not known" (DNS)** | Use connection pooler instead | ✅ **SOLVED** |
| **❌ VS Code extension won't connect** | Use pooler hostname `aws-0-ap-south-1.pooler.supabase.com` | ✅ **WORKS** |
| **❌ REST API fails** | Check `supabase_url` and API keys in `.env` | 🔧 **CONFIG** |
| **❌ All connections fail** | Check internet connectivity and Supabase project status | 🌐 **NETWORK** |

### 🎯 **Connection Priority Order**

The client automatically tries connections in this order:
1. **Connection Pooler** (IPv4) → Most reliable ✅
2. **Direct IPv6** → Requires IPv6 support ⚠️  
3. **Direct Hostname** → Usually fails due to DNS ❌
4. **REST API Fallback** → Always works as backup ✅

### 💡 **Pro Tips for Remote Connection**

```bash
# Test your connection instantly
python -c "from supabase_api_client import SupabaseClient; SupabaseClient().test_connections()"

# Check if Supabase project is active  
curl -s "https://yujmrarqlotmdvkhygmh.supabase.co/rest/v1/" | head -1

# Verify DNS resolution
nslookup aws-0-ap-south-1.pooler.supabase.com
```

### Connection Issues

| Issue | Solution |
|-------|----------|
| **REST API fails** | Check `supabase_url` and API keys in `.env` |
| **PostgreSQL pooler fails** | Verify `pooler_host` and credentials |
| **IPv6 direct fails** | Normal - requires IPv6 network support |
| **All connections fail** | Check internet connectivity and Supabase project status |

### Common Errors

```python
# Handle connection errors gracefully
try:
    client = SupabaseClient()
    results = client.test_connections()
    
    if results['rest_api']:
        print("✅ REST API available")
        data = client.query_table('returns_customer')
    elif results['pooler']:
        print("✅ PostgreSQL pooler available")
        data = client.query_sql('SELECT * FROM returns_customer')
    else:
        print("❌ No connections available")
        
except Exception as e:
    print(f"Error: {e}")
```

### Performance Tips

1. **Use REST API for small queries** - Simpler and more reliable
2. **Use PostgreSQL for complex queries** - Better performance for joins
3. **Limit result sets** - Add `LIMIT` clauses to large queries
4. **Use connection pooler** - Better performance than direct connection
5. **Cache frequently used data** - Store results locally when possible

## 🛠️ Remote Database Connection Setup

### VS Code PostgreSQL Extension

1. **Install Extension**: Search for "PostgreSQL" in VS Code Extensions (`Ctrl+Shift+X`)
   - Install: **"PostgreSQL"** by Microsoft (`ms-ossdata.vscode-pgsql`)

2. **Add Connection** (`Ctrl+Shift+P` → "PostgreSQL: Add Connection"):
   - **Server Name**: `aws-0-ap-south-1.pooler.supabase.com`
   - **Port**: `5432`
   - **Database**: `postgres`
   - **Username**: `postgres.yujmrarqlotmdvkhygmh`
   - **Password**: `admin@2027`
   - **Connection Name**: `Supabase RE.AI` (or any name)
   - **SSL Mode**: `require`

### pgAdmin 4 Connection

Use the same connection details as above:
```
Host: aws-0-ap-south-1.pooler.supabase.com
Port: 5432
Database: postgres
Username: postgres.yujmrarqlotmdvkhygmh
Password: admin@2027
SSL Mode: Require
```

### Direct Django Connection

Update your Django `settings.py` to use Supabase PostgreSQL:

```python
# In your .env file
DATABASE_URL=postgresql://postgres.yujmrarqlotmdvkhygmh:admin@2027@aws-0-ap-south-1.pooler.supabase.com:5432/postgres

# In settings.py
DATABASES = {
    'default': dj_database_url.config(
        default=os.environ.get('DATABASE_URL'),
        conn_max_age=600,
        conn_health_checks=True,
        ssl_require=True,
    )
}
```

### Connection Troubleshooting

#### Why Connection Pooler Works but Direct Doesn't

- **Direct Hostname**: `db.yujmrarqlotmdvkhygmh.supabase.co` → IPv6 only (`2406:da1a:6b0:f602:f8ce:c474:821a:215d`)
- **Connection Pooler**: `aws-0-ap-south-1.pooler.supabase.com` → IPv4 (`3.111.105.85`)

Most networks don't support IPv6, so the connection pooler (IPv4) is the reliable solution.

#### Common Issues

| Problem | Solution |
|---------|----------|
| Connection timeout | Use connection pooler instead of direct hostname |
| DNS resolution fails | Use IPv4 pooler endpoint |
| Authentication error | Ensure username includes project ID: `postgres.yujmrarqlotmdvkhygmh` |
| SSL errors | Always enable SSL/TLS (required by Supabase) |

## 📚 API Reference

### SupabaseClient Methods

#### Connection Testing
- `test_connections()` - Test all available connection methods
- `get_postgresql_connection(method='auto')` - Get PostgreSQL connection

#### REST API Methods
- `get_tables()` - List all available tables
- `query_table(table_name, limit=100, filters=None)` - Query table via REST
- `insert_record(table_name, data)` - Insert new record
- `update_record(table_name, record_id, data)` - Update existing record
- `delete_record(table_name, record_id)` - Delete record

#### SQL Methods
- `query_sql(query, method='auto')` - Execute SQL query and return DataFrame

## 🔐 Security Notes

- **Service Key**: Used for full database access - keep secure
- **Environment Variables**: Never commit `.env` file to version control
- **SSL Required**: All connections use SSL/TLS encryption
- **IP Restrictions**: Consider restricting database access by IP if needed

## 🎯 Next Steps

1. **Explore your data** - Run the client to see available tables
2. **Build queries** - Use REST API or SQL based on your needs
3. **Integrate with apps** - Import client into your Django/Flask apps
4. **Monitor usage** - Check Supabase dashboard for API usage
5. **Scale up** - Upgrade Supabase plan for higher limits

---

**File**: `supabase_api_client.py`  
**Author**: RE.AI Project  
**Updated**: July 3, 2025  
**Python**: 3.9+  
**Dependencies**: requests, psycopg2-binary, pandas, python-dotenv
