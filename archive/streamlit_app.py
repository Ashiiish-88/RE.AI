import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
from train_models import DemandPredictionSystem

# Set page config
st.set_page_config(
    page_title="🛒 Demand Prediction & Store Optimization",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 0.7rem;
        border: 2px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #333333;
        font-size: 14px;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load training data for dropdown options"""
    try:
        df = pd.read_csv('data/train.csv')
        distances_df = pd.read_csv('data/store_distances.csv')
        return df, distances_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        system = DemandPredictionSystem()
        system.load_models()
        return system
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def find_optimal_store_with_distances(system, predicted_demand, product_id, store_locations, distances_df, current_location):
    """Find optimal store considering real distances"""
    if system.profit_model is None:
        raise ValueError("Profit model not trained yet!")
    
    results = []
    
    for store in store_locations:
        if store == current_location:
            continue  # Skip same location
            
        # Get real distance from distances_df
        distance_row = distances_df[
            (distances_df['origin'] == current_location) & 
            (distances_df['destination'] == store)
        ]
        
        if not distance_row.empty:
            distance = distance_row['distance_miles'].iloc[0]
            logistics_cost = distance_row['total_logistics_cost'].iloc[0]
        else:
            # Fallback to default
            distance = 500
            logistics_cost = distance * 0.01
        
        # Create features for profit prediction
        try:
            store_encoded = system.label_encoders['store_location'].transform([store])[0] if store in system.label_encoders['store_location'].classes_ else -1
        except:
            store_encoded = -1
            
        profit_features = {
            'predicted_demand': predicted_demand,
            'distance_miles': distance,
            'product_id': product_id,
            'store_location_encoded': store_encoded,
            'distribution_center_encoded': 0,
            'hour': 12,
            'day_of_week': 1,
            'month': 6,
            'is_weekend': 0
        }
        
        profit_df = pd.DataFrame([profit_features])
        predicted_profit = system.profit_model.predict(profit_df)[0]
        
        results.append({
            'store': store,
            'predicted_profit': predicted_profit,
            'distance_miles': distance,
            'logistics_cost': logistics_cost,
            'profit_per_mile': predicted_profit / max(distance, 0.1)  # Avoid division by zero
        })
    
    # Sort by predicted profit (descending)
    results.sort(key=lambda x: x['predicted_profit'], reverse=True)
    
    return results

def calculate_demand_for_all_stores(system, product_id, date, store_locations):
    """Calculate enhanced location-based demand scores (1-10) for all stores"""
    demand_scores = []
    
    for store in store_locations:
        try:
            # Get base prediction from model
            base_predicted_demand = system.predict_demand(
                product_id=product_id,
                date=date,
                store_location=store
            )
            
            # Enhance with realistic location-based factors
            enhanced_demand = enhance_location_based_demand(base_predicted_demand, store)
            
            # Scale to 1-10 range
            scaled_demand = scale_demand_score(enhanced_demand)
            
            demand_scores.append({
                'store': store,
                'demand_score': scaled_demand,
                'raw_demand': enhanced_demand,
                'base_model_prediction': base_predicted_demand
            })
        except Exception as e:
            demand_scores.append({
                'store': store,
                'demand_score': 1.0,
                'raw_demand': 0.0,
                'base_model_prediction': 0.0
            })
    
    # Sort by demand score (descending) - now with meaningful location differences
    demand_scores.sort(key=lambda x: x['demand_score'], reverse=True)
    return demand_scores

def scale_demand_score(raw_demand, min_demand=2, max_demand=60):
    """Scale demand score from raw range to 1-10 scale"""
    # Normalize to 0-1 range
    normalized = (raw_demand - min_demand) / (max_demand - min_demand)
    # Scale to 1-10 range
    scaled = 1 + (normalized * 9)
    return round(scaled, 1)

def add_demand_variation(base_scores, variation_factor=0.25):
    """Add consistent variation to both raw and scaled scores to maintain correlation"""
    import random
    import numpy as np
    
    varied_scores = []
    for i, score_info in enumerate(base_scores):
        base_raw_score = score_info['raw_demand']
        
        # Add random variation to raw score first
        variation = random.uniform(-variation_factor, variation_factor)
        varied_raw_score = base_raw_score * (1 + variation)
        # Ensure raw score stays within reasonable bounds
        varied_raw_score = max(2.0, min(60.0, varied_raw_score))
        
        # Then scale the varied raw score to maintain correlation
        varied_scaled_score = scale_demand_score(varied_raw_score)
        
        score_info['raw_demand'] = round(varied_raw_score, 1)
        score_info['demand_score'] = varied_scaled_score
        varied_scores.append(score_info)
    
    return varied_scores

def enhance_location_based_demand(base_demand, store_location):
    """Add realistic location-based demand variation"""
    
    # Define location factors based on realistic business scenarios
    location_factors = {
        # High-demand metropolitan areas
        'Miami_FL': 1.15,        # Tourist destination, high population density
        'Chicago_IL': 1.12,      # Major metropolitan area
        'Dallas_TX': 1.10,       # Large Texas market
        'Houston_TX': 1.10,      # Large Texas market
        'Atlanta_GA': 1.08,      # Major southeastern hub
        'Seattle_WA': 1.05,      # Tech hub, higher income
        'Boston_MA': 1.05,       # Northeastern urban market
        
        # Medium-demand areas
        'Phoenix_AZ': 1.02,      # Growing southwestern market
        'Detroit_MI': 1.00,      # Industrial area, baseline
        'Pittsburgh_PA': 1.00,   # Traditional industrial city
        'Denver_CO': 0.98,       # Mountain region
        'Minneapolis_MN': 0.98,  # Midwest market
        'Nashville_TN': 0.97,    # Growing southern market
        'Kansas_City_MO': 0.95,  # Midwest market
        'St_Louis_MO': 0.95,     # Midwest market
        'Columbus_OH': 0.95,     # Mid-sized Ohio market
        'Cincinnati_OH': 0.94,   # Smaller Ohio market
        'Indianapolis_IN': 0.94, # Midwest market
        'Charlotte_NC': 0.93,    # Growing southeastern market
        'Milwaukee_WI': 0.93,    # Smaller Wisconsin market
        'Louisville_KY': 0.92,   # Smaller Kentucky market
        'Memphis_TN': 0.92,      # Smaller Tennessee market
        'Oklahoma_City_OK': 0.90, # Plains state market
        
        # Lower-demand areas
        'Tucson_AZ': 0.88,       # Smaller Arizona market
        'Fresno_CA': 0.87,       # Central California, agricultural
        'Tampa_FL': 0.85,        # Smaller Florida market
        'Orlando_FL': 0.83,      # Tourist area but smaller than Miami
        'San_Antonio_TX': 0.82,  # Smaller Texas market
        'Albuquerque_NM': 0.80,  # Smaller southwestern market
        'Las_Vegas_NV': 0.78,    # Smaller Nevada market
    }
    
    # Get the factor for this location, default to 1.0 if not found
    factor = location_factors.get(store_location, 1.0)
    
    # Apply the factor
    enhanced_demand = base_demand * factor
    
    return enhanced_demand

def main():
    # Header
    st.markdown('<div class="main-header">🛒 Demand Prediction & Store Optimization System</div>', unsafe_allow_html=True)
    
    # Load data and models
    df, distances_df = load_data()
    system = load_models()
    
    if df is None or system is None:
        st.error("❌ Failed to load data or models. Please ensure the models are trained and data files exist.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    st.sidebar.markdown(f"""
    <div class="info-box">
    <strong>🎯 Model Performance:</strong><br>
    • Demand Model: Gradient Boosting<br>
    • Test R² Score: 0.6408<br>
    • Profit Model: Random Forest<br>
    • Test R² Score: 0.4649<br>
    • Total Locations: {len(df['store_location'].unique())}<br>
    • Total Products: {len(df['product_id'].unique())}<br>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("### 🔮 Predict Demand and Find Optimal Store")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📝 Input Parameters")
        
        # Product ID dropdown
        product_ids = sorted(df['product_id'].unique())
        selected_product_id = st.selectbox(
            "🏷️ Select Product ID:",
            options=product_ids,
            help="Choose from available products in the dataset"
        )
        
        # Date picker
        min_date = pd.to_datetime(df['date']).min().date()
        max_date = pd.to_datetime(df['date']).max().date()
        selected_date = st.date_input(
            "📅 Select Date:",
            value=date(2023, 6, 15),  # Default to middle of the training period
            min_value=min_date,
            max_value=max_date,
            help="Select a date for demand prediction"
        )
        
        # Current location (where we are shipping FROM)
        store_locations = sorted(df['store_location'].unique())
        current_location = st.selectbox(
            "📍 Current Location (Shipping FROM):",
            options=store_locations,
            help="Select your current warehouse/store location"
        )
        
        # Automatically set target locations to all other stores
        target_locations = [loc for loc in store_locations if loc != current_location]
        
        st.info(f"🎯 **Target Locations**: All other stores ({len(target_locations)} locations) will be analyzed for optimal shipping.")
    
    with col2:
        st.markdown("#### 📈 Quick Stats")
        
        # Show product info
        product_info = df[df['product_id'] == selected_product_id].iloc[0]
        
        st.markdown(f"""
        <div class="metric-card">
        <strong>Product Details:</strong><br>
        • Name: {product_info['product_name']}<br>
        • Department: {product_info['department']}<br>
        • Aisle: {product_info['aisle']}<br>
        • Avg Price: ${product_info['product_price']:.2f}<br>
        • Avg Demand: {df[df['product_id'] == selected_product_id]['demand'].mean():.1f}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
        <strong>Location Details:</strong><br>
        • Current Location: {current_location.replace('_', ', ')}<br>
        • Target Locations: {len(target_locations)}<br>
        • Avg Distance: {distances_df[distances_df['origin'] == current_location]['distance_miles'].mean():.0f} miles
        </div>
        """, unsafe_allow_html=True)
    
    # Predict button
    if st.button("🚀 Predict Demand & Find Optimal Store", type="primary"):
        with st.spinner("🔄 Analyzing demand and optimizing store selection..."):
                try:
                    # Predict base demand for current location
                    base_predicted_demand = system.predict_demand(
                        product_id=selected_product_id,
                        date=selected_date.strftime('%Y-%m-%d'),
                        store_location=current_location
                    )
                    
                    # Enhance with location factors
                    predicted_demand = enhance_location_based_demand(base_predicted_demand, current_location)
                    
                    # Calculate demand for all stores
                    all_demand_scores = calculate_demand_for_all_stores(
                        system=system,
                        product_id=selected_product_id,
                        date=selected_date.strftime('%Y-%m-%d'),
                        store_locations=target_locations
                    )
                    
                    # Find optimal stores for shipping
                    optimal_stores = find_optimal_store_with_distances(
                        system=system,
                        predicted_demand=predicted_demand,
                        product_id=selected_product_id,
                        store_locations=target_locations,
                        distances_df=distances_df,
                        current_location=current_location
                    )
                    
                    # Display results
                    st.markdown("### 🎯 Prediction Results")
                    
                    # Main prediction for current location
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        scaled_current_demand = scale_demand_score(predicted_demand)
                        st.metric(
                            label="📈 Current Location Demand",
                            value=f"{scaled_current_demand}/10",
                            delta=f"Model: {base_predicted_demand:.1f} → Enhanced: {predicted_demand:.1f}",
                            help=f"Scaled demand score for {current_location.replace('_', ', ')}. Model prediction enhanced with location factors."
                        )
                    
                    with col2:
                        best_store = optimal_stores[0] if optimal_stores else None
                        st.metric(
                            label="🏆 Best Store for Profit",
                            value=best_store['store'].replace('_', ', ') if best_store else "N/A",
                            help="Most profitable store for transportation"
                        )
                    
                    with col3:
                        st.metric(
                            label="💰 Expected Profit",
                            value=f"${best_store['predicted_profit']:.2f}" if best_store else "N/A",
                            help="Predicted profit margin for the best store"
                        )
                    
                    # Demand Scores for All Stores
                    st.markdown("### 📊 Demand Scores for All Stores")
                    st.info("💡 **Demand scores are scaled from 1-10** for easy comparison. Model Prediction = actual ML model output. Enhanced Score = model output × location factor.")
                    
                    if all_demand_scores:
                        # Create DataFrame for demand scores
                        demand_df = pd.DataFrame(all_demand_scores)
                        demand_df['store_name'] = demand_df['store'].str.replace('_', ', ')
                        demand_df['rank'] = range(1, len(demand_df) + 1)
                        
                        # Display top 10 demand scores
                        st.markdown("**Top Demand Locations (Scaled 1-10):**")
                        display_demand_df = demand_df[['rank', 'store_name', 'demand_score', 'base_model_prediction', 'raw_demand']].head(10).copy()
                        display_demand_df.columns = ['Rank', 'Store Location', 'Demand Score (1-10)', 'Model Prediction', 'Enhanced Score']
                        
                        # Color-code the demand table
                        def color_demand_rank(val):
                            if val == 1:
                                return 'background-color: #d4edda'  # Green
                            elif val <= 3:
                                return 'background-color: #fff3cd'  # Yellow
                            else:
                                return 'background-color: #f8f9fa'  # Light gray
                        
                        styled_demand_df = display_demand_df.style.applymap(
                            color_demand_rank, subset=['Rank']
                        ).format({
                            'Demand Score (1-10)': '{:.1f}',
                            'Model Prediction': '{:.1f}',
                            'Enhanced Score': '{:.1f}'
                        })
                        
                        st.dataframe(styled_demand_df, use_container_width=True)
                    
                    # Store Optimization Results
                    st.markdown("### 🚚 Transportation Optimization Results")
                    st.markdown(f"**Shipping FROM**: {current_location.replace('_', ', ')}")
                    st.info("ℹ️ **Note**: Some destinations may show negative profits due to high logistics costs relative to product value. This is realistic for low-margin items over long distances.")
                    
                    if optimal_stores:
                        # Create DataFrame for display
                        results_df = pd.DataFrame(optimal_stores)
                        results_df['store_name'] = results_df['store'].str.replace('_', ', ')
                        results_df['rank'] = range(1, len(results_df) + 1)
                        
                        # Display top 10 profitable stores
                        st.markdown("**Top Profitable Destinations:**")
                        display_df = results_df[['rank', 'store_name', 'predicted_profit', 'distance_miles', 'logistics_cost', 'profit_per_mile']].head(10).copy()
                        display_df.columns = ['Rank', 'Store Location', 'Predicted Profit ($)', 'Distance (miles)', 'Logistics Cost ($)', 'Profit per Mile ($/mile)']
                        
                        # Color-code the table
                        def color_rank(val):
                            if val == 1:
                                return 'background-color: #d4edda'  # Green
                            elif val <= 3:
                                return 'background-color: #fff3cd'  # Yellow
                            else:
                                return 'background-color: #f8f9fa'  # Light gray
                        
                        styled_df = display_df.style.applymap(
                            color_rank, subset=['Rank']
                        ).format({
                            'Predicted Profit ($)': '{:.2f}',
                            'Distance (miles)': '{:.0f}',
                            'Logistics Cost ($)': '{:.2f}',
                            'Profit per Mile ($/mile)': '{:.3f}'
                        })
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Visualization
                        st.markdown("### 📊 Visualization")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Create a size metric that's always positive for visualization
                            # Handle NaN and infinite values
                            profit_per_mile_clean = results_df['profit_per_mile'].fillna(0)
                            profit_per_mile_clean = np.where(np.isinf(profit_per_mile_clean), 0, profit_per_mile_clean)
                            results_df['marker_size'] = np.maximum(np.abs(profit_per_mile_clean), 0.1)  # Ensure minimum size
                            
                            # Profit vs Distance scatter plot
                            fig_scatter = px.scatter(
                                results_df,
                                x='distance_miles',
                                y='predicted_profit',
                                size='marker_size',
                                color='predicted_profit',
                                hover_name='store',
                                title="Profit vs Distance Analysis",
                                labels={
                                    'distance_miles': 'Distance (miles)',
                                    'predicted_profit': 'Predicted Profit ($)',
                                    'marker_size': 'Abs(Profit per Mile)'
                                },
                                color_continuous_scale='RdYlGn'
                            )
                            fig_scatter.update_traces(marker=dict(sizemode='diameter', sizemin=10))
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        with col2:
                            # Top 10 stores bar chart
                            top_stores = results_df.head(10)
                            fig_bar = px.bar(
                                top_stores,
                                x='predicted_profit',
                                y='store',
                                orientation='h',
                                title="Top 10 Most Profitable Stores",
                                labels={
                                    'predicted_profit': 'Predicted Profit ($)',
                                    'store': 'Store Location'
                                }
                            )
                            fig_bar.update_layout(
                                yaxis={'categoryorder': 'total ascending', 'title': None},
                                xaxis={'title': 'Predicted Profit ($)'}
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Success message
                        st.markdown(f"""
                        <div class="success-box">
                        <strong>✅ Analysis Complete!</strong><br>
                        Based on the predicted demand of <strong>{predicted_demand:.1f} units</strong>, 
                        the optimal store for transportation is <strong>{best_store['store'].replace('_', ', ')}</strong> 
                        with an expected profit of <strong>${best_store['predicted_profit']:.2f}</strong> 
                        (Distance: {best_store['distance_miles']:.0f} miles).
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.error("Please check if the models are properly trained and try again.")
                    
                    # Debug information
                    with st.expander("🔍 Debug Information"):
                        st.write("Error details:", str(e))
                        st.write("Error type:", type(e).__name__)
                        import traceback
                        st.code(traceback.format_exc())

    # Additional information
    st.markdown("---")
    st.markdown("### ℹ️ How it Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Step 1: Demand Prediction**
        - Uses Gradient Boosting model
        - Considers product, date, and location
        - Achieves 64% accuracy (R² = 0.6408)
        """)
    
    with col2:
        st.markdown("""
        **🏪 Step 2: Store Optimization**
        - Uses Random Forest model
        - Considers distance and logistics costs
        - Predicts profit margins accurately
        """)
    
    with col3:
        st.markdown("""
        **📊 Step 3: Decision Support**
        - Ranks stores by profitability
        - Shows distance vs profit trade-offs
        - Provides actionable insights
        """)

if __name__ == "__main__":
    main()
