import pandas as pd
import numpy as np
import requests
import time
import json
from itertools import combinations
import os

def get_coordinates_from_city(city_state):
    """Get latitude and longitude from city_state string using OpenStreetMap Nominatim API"""
    # Convert format from "City_ST" to "City, ST"
    city, state = city_state.replace('_', ' ').rsplit(' ', 1)
    location_query = f"{city}, {state}, USA"
    
    # Use Nominatim API (free, no API key required)
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': location_query,
        'format': 'json',
        'limit': 1
    }
    
    headers = {
        'User-Agent': 'DemandPredictionApp/1.0'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
        else:
            print(f"No coordinates found for {location_query}")
            return None, None
    except Exception as e:
        print(f"Error getting coordinates for {location_query}: {e}")
        return None, None

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in miles
    r = 3956
    
    return c * r

def create_store_distances():
    """Create a comprehensive distance matrix for all store combinations"""
    print("Creating store distance matrix...")
    
    # Load data to get unique store locations
    df = pd.read_csv('data/train.csv')
    store_locations = df['store_location'].unique()
    distribution_centers = df['distribution_center'].unique()
    
    # Combine all unique locations
    all_locations = list(set(list(store_locations) + list(distribution_centers)))
    
    print(f"Found {len(all_locations)} unique locations")
    print("Locations:", all_locations)
    
    # Get coordinates for all locations
    location_coords = {}
    
    for location in all_locations:
        print(f"Getting coordinates for {location}...")
        lat, lon = get_coordinates_from_city(location)
        
        if lat is not None and lon is not None:
            location_coords[location] = (lat, lon)
            print(f"  {location}: ({lat:.4f}, {lon:.4f})")
        else:
            # Use default coordinates if API fails
            print(f"  Using default coordinates for {location}")
            location_coords[location] = (39.8283, -98.5795)  # Geographic center of US
        
        # Be respectful to the API
        time.sleep(1)
    
    # Create distance matrix
    distances_data = []
    
    print("\nCalculating distances between all location pairs...")
    
    # Calculate distances for all combinations
    for loc1 in all_locations:
        for loc2 in all_locations:
            if loc1 != loc2:
                lat1, lon1 = location_coords[loc1]
                lat2, lon2 = location_coords[loc2]
                
                distance = calculate_distance(lat1, lon1, lat2, lon2)
                
                distances_data.append({
                    'origin': loc1,
                    'destination': loc2,
                    'distance_miles': round(distance, 2),
                    'logistics_cost_per_mile': 0.01,  # $0.01 per mile
                    'total_logistics_cost': round(distance * 0.01, 2)
                })
    
    # Create DataFrame
    distances_df = pd.DataFrame(distances_data)
    
    # Save to CSV
    distances_df.to_csv('data/store_distances.csv', index=False)
    
    print(f"\nDistance matrix created with {len(distances_df)} combinations")
    print(f"Average distance: {distances_df['distance_miles'].mean():.2f} miles")
    print(f"Max distance: {distances_df['distance_miles'].max():.2f} miles")
    print(f"Min distance: {distances_df['distance_miles'].min():.2f} miles")
    
    # Display sample of the data
    print("\nSample of distance data:")
    print(distances_df.head(10))
    
    return distances_df

def create_simplified_distances():
    """Create a simplified distance matrix using realistic estimates"""
    print("Creating simplified store distance matrix...")
    
    # Load data to get unique store locations
    df = pd.read_csv('data/train.csv')
    store_locations = df['store_location'].unique()
    distribution_centers = df['distribution_center'].unique()
    
    # Combine all unique locations
    all_locations = list(set(list(store_locations) + list(distribution_centers)))
    
    # Create simplified distance matrix with realistic estimates
    distances_data = []
    
    # Major cities and their approximate positions (simplified grouping)
    city_regions = {
        'Denver_CO': 'West',
        'Phoenix_AZ': 'West', 
        'Tucson_AZ': 'West',
        'Las_Vegas_NV': 'West',
        'Fresno_CA': 'West',
        'Sacramento_CA': 'West',
        'Seattle_WA': 'Northwest',
        'Albuquerque_NM': 'Southwest',
        'Chicago_IL': 'Midwest',
        'Milwaukee_WI': 'Midwest',
        'Minneapolis_MN': 'Midwest',
        'Indianapolis_IN': 'Midwest',
        'Detroit_MI': 'Midwest',
        'Cincinnati_OH': 'Midwest',
        'Columbus_OH': 'Midwest',
        'St_Louis_MO': 'Midwest',
        'Kansas_City_MO': 'Midwest',
        'Oklahoma_City_OK': 'South',
        'Dallas_TX': 'South',
        'Houston_TX': 'South',
        'San_Antonio_TX': 'South',
        'Memphis_TN': 'South',
        'Nashville_TN': 'South',
        'Atlanta_GA': 'Southeast',
        'Charlotte_NC': 'Southeast',
        'Louisville_KY': 'Southeast',
        'Miami_FL': 'Southeast',
        'Orlando_FL': 'Southeast',
        'Tampa_FL': 'Southeast',
        'Boston_MA': 'Northeast',
        'Pittsburgh_PA': 'Northeast'
    }
    
    # Distance estimates between regions
    region_distances = {
        ('West', 'West'): np.random.normal(300, 100),
        ('West', 'Northwest'): np.random.normal(800, 200),
        ('West', 'Southwest'): np.random.normal(400, 150),
        ('West', 'Midwest'): np.random.normal(1000, 200),
        ('West', 'South'): np.random.normal(1200, 300),
        ('West', 'Southeast'): np.random.normal(1500, 300),
        ('West', 'Northeast'): np.random.normal(1800, 400),
        ('Northwest', 'Midwest'): np.random.normal(1200, 200),
        ('Northwest', 'South'): np.random.normal(1500, 300),
        ('Northwest', 'Southeast'): np.random.normal(1800, 400),
        ('Northwest', 'Northeast'): np.random.normal(2000, 400),
        ('Southwest', 'Midwest'): np.random.normal(800, 200),
        ('Southwest', 'South'): np.random.normal(600, 150),
        ('Southwest', 'Southeast'): np.random.normal(1000, 200),
        ('Southwest', 'Northeast'): np.random.normal(1500, 300),
        ('Midwest', 'Midwest'): np.random.normal(400, 100),
        ('Midwest', 'South'): np.random.normal(600, 150),
        ('Midwest', 'Southeast'): np.random.normal(800, 200),
        ('Midwest', 'Northeast'): np.random.normal(800, 200),
        ('South', 'South'): np.random.normal(400, 100),
        ('South', 'Southeast'): np.random.normal(600, 150),
        ('South', 'Northeast'): np.random.normal(1000, 200),
        ('Southeast', 'Southeast'): np.random.normal(300, 100),
        ('Southeast', 'Northeast'): np.random.normal(700, 150),
        ('Northeast', 'Northeast'): np.random.normal(300, 100)
    }
    
    for loc1 in all_locations:
        for loc2 in all_locations:
            if loc1 == loc2:
                distance = 0
            else:
                region1 = city_regions.get(loc1, 'Unknown')
                region2 = city_regions.get(loc2, 'Unknown')
                
                # Get distance estimate
                key = tuple(sorted([region1, region2]))
                if key in region_distances:
                    distance = max(50, region_distances[key])  # Minimum 50 miles
                else:
                    distance = np.random.normal(800, 200)  # Default distance
                
                distance = max(50, distance)  # Ensure minimum distance
            
            distances_data.append({
                'origin': loc1,
                'destination': loc2,
                'distance_miles': round(distance, 2),
                'logistics_cost_per_mile': 0.01,
                'total_logistics_cost': round(distance * 0.01, 2)
            })
    
    # Create DataFrame
    distances_df = pd.DataFrame(distances_data)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    distances_df.to_csv('data/store_distances.csv', index=False)
    
    print(f"Distance matrix created with {len(distances_df)} combinations")
    print(f"Average distance: {distances_df['distance_miles'].mean():.2f} miles")
    print(f"Max distance: {distances_df['distance_miles'].max():.2f} miles")
    print(f"Min distance: {distances_df['distance_miles'].min():.2f} miles")
    
    # Display sample of the data
    print("\nSample of distance data:")
    print(distances_df.head(10))
    
    return distances_df

if __name__ == "__main__":
    # Try to create realistic distances using API, fall back to simplified version
    try:
        distances_df = create_store_distances()
    except Exception as e:
        print(f"API method failed: {e}")
        print("Using simplified distance estimation...")
        distances_df = create_simplified_distances()
    
    print("\nStore distances file created successfully!")
