#!/usr/bin/env python3
"""
Supabase Database Client
Complete solution for connecting to Supabase database with multiple methods:
1. REST API (Always works)
2. PostgreSQL Direct with IPv6 support
3. PostgreSQL via Connection Pooler (IPv4)

Author: RE.AI Project
Date: July 3, 2025
"""

import requests
import json
import os
import socket
import psycopg2
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import pandas as pd

# Load environment variables
load_dotenv()

class SupabaseClient:
    """Complete Supabase client with multiple connection methods"""
    
    def __init__(self):
        self.base_url = os.getenv('supabase_url')
        self.anon_key = os.getenv('supabase_anon_key')
        self.service_key = os.getenv('supabase_service_key')
        
        # REST API headers (use service key for full access)
        self.headers = {
            'apikey': self.service_key,
            'Authorization': f'Bearer {self.service_key}',
            'Content-Type': 'application/json'
        }
        
        # PostgreSQL connection details
        self.pg_direct = {
            'host': 'db.yujmrarqlotmdvkhygmh.supabase.co',
            'ipv6': '2406:da1a:6b0:f602:f8ce:c474:821a:215d',
            'port': 5432,
            'database': 'postgres',
            'user': 'postgres',
            'password': 'admin@2027'
        }
        
        self.pg_pooler = {
            'host': 'aws-0-ap-south-1.pooler.supabase.com',
            'ipv4': '3.111.105.85',
            'port': 5432,
            'database': 'postgres',
            'user': 'postgres.yujmrarqlotmdvkhygmh',
            'password': 'admin@2027'
        }
    
    # REST API Methods
    def test_connections(self):
        """Test all available connection methods"""
        results = {}
        
        print("🧪 Testing Supabase Connections")
        print("=" * 50)
        
        # Test REST API
        try:
            response = requests.get(f"{self.base_url}/rest/v1/", headers=self.headers, timeout=10)
            results['rest_api'] = response.status_code == 200
            print(f"🌐 REST API: {'✅ Working' if results['rest_api'] else '❌ Failed'}")
        except Exception as e:
            results['rest_api'] = False
            print(f"🌐 REST API: ❌ Failed - {e}")
        
        # Test PostgreSQL Direct (IPv6)
        try:
            conn = psycopg2.connect(
                host=self.pg_direct['ipv6'],
                port=self.pg_direct['port'],
                database=self.pg_direct['database'],
                user=self.pg_direct['user'],
                password=self.pg_direct['password'],
                sslmode='require',
                connect_timeout=10
            )
            conn.close()
            results['direct_ipv6'] = True
            print(f"🔗 Direct IPv6: ✅ Working")
        except Exception as e:
            results['direct_ipv6'] = False
            print(f"🔗 Direct IPv6: ❌ Failed - {e}")
        
        # Test PostgreSQL Direct (Hostname)
        try:
            conn = psycopg2.connect(
                host=self.pg_direct['host'],
                port=self.pg_direct['port'],
                database=self.pg_direct['database'],
                user=self.pg_direct['user'],
                password=self.pg_direct['password'],
                sslmode='require',
                connect_timeout=10
            )
            conn.close()
            results['direct_hostname'] = True
            print(f"🔗 Direct Hostname: ✅ Working")
        except Exception as e:
            results['direct_hostname'] = False
            print(f"🔗 Direct Hostname: ❌ Failed - {e}")
        
        # Test Connection Pooler
        try:
            conn = psycopg2.connect(
                host=self.pg_pooler['host'],
                port=self.pg_pooler['port'],
                database=self.pg_pooler['database'],
                user=self.pg_pooler['user'],
                password=self.pg_pooler['password'],
                sslmode='require',
                connect_timeout=10
            )
            conn.close()
            results['pooler'] = True
            print(f"🔗 Connection Pooler: ✅ Working")
        except Exception as e:
            results['pooler'] = False
            print(f"🔗 Connection Pooler: ❌ Failed - {e}")
        
        return results
    
    def get_postgresql_connection(self, method='auto'):
        """Get PostgreSQL connection using specified method"""
        if method == 'auto':
            # Try methods in order of preference
            for method_name in ['pooler', 'direct_ipv6', 'direct_hostname']:
                try:
                    return self.get_postgresql_connection(method_name)
                except:
                    continue
            raise Exception("All PostgreSQL connection methods failed")
        
        if method == 'pooler':
            return psycopg2.connect(
                host=self.pg_pooler['host'],
                port=self.pg_pooler['port'],
                database=self.pg_pooler['database'],
                user=self.pg_pooler['user'],
                password=self.pg_pooler['password'],
                sslmode='require'
            )
        elif method == 'direct_ipv6':
            return psycopg2.connect(
                host=self.pg_direct['ipv6'],
                port=self.pg_direct['port'],
                database=self.pg_direct['database'],
                user=self.pg_direct['user'],
                password=self.pg_direct['password'],
                sslmode='require'
            )
        elif method == 'direct_hostname':
            return psycopg2.connect(
                host=self.pg_direct['host'],
                port=self.pg_direct['port'],
                database=self.pg_direct['database'],
                user=self.pg_direct['user'],
                password=self.pg_direct['password'],
                sslmode='require'
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def query_sql(self, query: str, method='auto') -> pd.DataFrame:
        """Execute SQL query using PostgreSQL connection"""
        try:
            conn = self.get_postgresql_connection(method)
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"SQL query failed: {e}")
            return pd.DataFrame()
    
    def get_tables(self) -> List[str]:
        """Get list of all tables via REST API"""
        try:
            response = requests.get(f"{self.base_url}/rest/v1/", headers=self.headers)
            if response.status_code == 200:
                swagger = response.json()
                tables = list(swagger.get('paths', {}).keys())
                return [table.strip('/') for table in tables if table != '/']
            else:
                print(f"Error fetching tables: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def query_table(self, table_name: str, limit: int = 100, 
                   filters: Optional[Dict] = None) -> pd.DataFrame:
        """Query a table via REST API and return as DataFrame"""
        try:
            url = f"{self.base_url}/rest/v1/{table_name}"
            params = {'limit': limit}
            
            if filters:
                params.update(filters)
            
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data) if data else pd.DataFrame()
            else:
                print(f"Error querying {table_name}: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
    
    def insert_record(self, table_name: str, data: Dict) -> Dict:
        """Insert a new record into a table via REST API"""
        try:
            url = f"{self.base_url}/rest/v1/{table_name}"
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 201:
                return response.json()
            else:
                print(f"Error inserting into {table_name}: {response.status_code}")
                print(response.text)
                return {}
        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    def update_record(self, table_name: str, record_id: int, data: Dict) -> Dict:
        """Update a record in a table via REST API"""
        try:
            url = f"{self.base_url}/rest/v1/{table_name}?id=eq.{record_id}"
            response = requests.patch(url, headers=self.headers, json=data)
            if response.status_code == 204:
                return {"success": True, "message": "Record updated"}
            else:
                print(f"Error updating {table_name}: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    def delete_record(self, table_name: str, record_id: int) -> Dict:
        """Delete a record from a table via REST API"""
        try:
            url = f"{self.base_url}/rest/v1/{table_name}?id=eq.{record_id}"
            response = requests.delete(url, headers=self.headers)
            if response.status_code == 204:
                return {"success": True, "message": "Record deleted"}
            else:
                print(f"Error deleting from {table_name}: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    def execute_custom_query(self, query_params: str) -> List[Dict]:
        """Execute a custom query using PostgREST syntax"""
        try:
            url = f"{self.base_url}/rest/v1/rpc/your_function_name"
            # This would be for custom SQL functions
            # For now, we'll use table queries with advanced filters
            print("Custom queries require RPC functions. Use query_table with filters instead.")
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []

def main():
    """Interactive demo of Supabase client capabilities"""
    client = SupabaseClient()
    
    print("🚀 Supabase Multi-Connection Client")
    print("=" * 50)
    
    # Test all connections
    results = client.test_connections()
    
    # Get table list via REST API
    print("\n📊 Available Tables:")
    tables = client.get_tables()
    if tables:
        django_tables = [t for t in tables if t.startswith(('auth_', 'django_'))]
        app_tables = [t for t in tables if t.startswith('returns_')]
        
        print(f"\n� Django Tables ({len(django_tables)}):")
        for table in sorted(django_tables):
            print(f"  • {table}")
        
        print(f"\n📦 Application Tables ({len(app_tables)}):")
        for table in sorted(app_tables):
            print(f"  • {table}")
    
    # Demo data access
    if results.get('rest_api'):
        print("\n📝 Sample Customer Data (REST API):")
        customers = client.query_table('returns_customer', limit=3)
        if not customers.empty:
            print(customers.to_string(index=False))
        else:
            print("  No customers found")
    
    # Demo SQL access if PostgreSQL works
    working_methods = [k for k, v in results.items() if v and k != 'rest_api']
    if working_methods:
        print(f"\n🔍 Sample SQL Query (using {working_methods[0]}):")
        try:
            df = client.query_sql("SELECT COUNT(*) as total_tables FROM information_schema.tables WHERE table_schema = 'public'")
            if not df.empty:
                print(f"  Total public tables: {df.iloc[0]['total_tables']}")
        except Exception as e:
            print(f"  SQL query failed: {e}")
    
    print("\n💡 Usage Examples:")
    print("  # REST API")
    print("  customers = client.query_table('returns_customer')")
    print("  client.insert_record('returns_customer', {...})")
    print()
    print("  # SQL Queries")
    print("  df = client.query_sql('SELECT * FROM returns_product')")
    print("  conn = client.get_postgresql_connection('pooler')")

if __name__ == "__main__":
    main()
