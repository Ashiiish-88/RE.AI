"""
Walmart Demand Prediction Streamlit App
=======================================

Professional web interface for the Walmart demand prediction and logistics optimization system.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.demand_system import DemandPredictionSystem
from src.utils.data_processor import DataProcessor
from src.utils.visualization import VisualizationEngine


class WalmartDemandApp:
    """
    Professional Streamlit application for demand prediction and store optimization.
    
    Features:
    - Interactive product selection
    - Real-time demand prediction
    - Store optimization recommendations
    - Professional visualizations
    - Business intelligence dashboard
    """
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.setup_page_config()
        self.load_data_and_models()
        self.viz_engine = VisualizationEngine()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="🛒 Walmart Demand Prediction System",
            page_icon="🛒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for professional styling
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                color: #0066cc;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: bold;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
                border: 1px solid rgba(255,255,255,0.2);
            }
            .success-box {
                padding: 1.5rem;
                border-radius: 10px;
                background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                border: none;
                color: #2c3e50;
                font-weight: 500;
            }
            .info-box {
                padding: 1.5rem;
                border-radius: 10px;
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                border: none;
                color: #2c3e50;
                font-weight: 500;
            }
            .stSelectbox > div > div > select {
                border-radius: 10px;
                border: 2px solid #0066cc;
            }
            .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 0.75rem 2rem;
                font-weight: bold;
                font-size: 1.1rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
        </style>
        """, unsafe_allow_html=True)
    
    def load_data_and_models(self):
        """Load data and trained models."""
        with st.spinner("🔄 Loading data and models..."):
            try:
                # Load data
                self.processor = DataProcessor()
                self.df = self.processor.load_train_data()
                self.distances_df = self.processor.load_store_distances()
                
                # Load models
                self.system = DemandPredictionSystem()
                self.system.load_models()
                
                # Cache frequently used data
                self.unique_products = self.processor.get_unique_products(self.df)
                self.store_locations = self.processor.get_unique_locations(self.df)
                
                st.success("✅ Data and models loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading data/models: {e}")
                st.stop()
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">🛒 Walmart Demand Prediction & Logistics Optimization</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 System Overview</h3>
        This advanced system predicts product demand across Walmart stores and recommends optimal logistics 
        strategies for maximum profitability. Using machine learning models trained on historical data, 
        we analyze demand patterns, calculate logistics costs, and identify the most profitable store destinations.
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with input controls."""
        st.sidebar.markdown("## 🎛️ Control Panel")
        
        # Product selection
        st.sidebar.markdown("### 📦 Product Selection")
        product_options = dict(zip(
            self.unique_products['product_name'] + " ($" + self.unique_products['product_price'].round(2).astype(str) + ")",
            self.unique_products['product_id']
        ))
        
        selected_product_display = st.sidebar.selectbox(
            "Choose Product:",
            options=list(product_options.keys()),
            help="Select a product to analyze demand and optimize logistics"
        )
        
        selected_product_id = product_options[selected_product_display]
        
        # Date selection
        st.sidebar.markdown("### 📅 Date Selection")
        selected_date = st.sidebar.date_input(
            "Analysis Date:",
            value=date.today(),
            help="Select the date for demand prediction"
        )
        
        # Current location
        st.sidebar.markdown("### 📍 Current Location")
        current_location = st.sidebar.selectbox(
            "Current Store Location:",
            options=self.store_locations,
            help="Select your current store location"
        )
        
        # Analysis options
        st.sidebar.markdown("### ⚙️ Analysis Options")
        
        analysis_mode = st.sidebar.radio(
            "Analysis Mode:",
            ["Quick Analysis (Top 10)", "Comprehensive Analysis (All Stores)"],
            help="Choose analysis depth"
        )
        
        show_advanced_metrics = st.sidebar.checkbox(
            "Show Advanced Metrics",
            value=True,
            help="Display detailed analytics and model insights"
        )
        
        return {
            'product_id': selected_product_id,
            'date': selected_date,
            'current_location': current_location,
            'analysis_mode': analysis_mode,
            'show_advanced': show_advanced_metrics
        }
    
    def predict_store_optimization(self, params: Dict) -> Dict:
        """
        Perform store optimization analysis.
        
        Args:
            params (Dict): Analysis parameters
            
        Returns:
            Dict: Optimization results
        """
        # Get product details
        product_info = self.unique_products[
            self.unique_products['product_id'] == params['product_id']
        ].iloc[0]
        
        # Get target locations (exclude current location)
        target_locations = [loc for loc in self.store_locations 
                           if loc != params['current_location']]
        
        results = []
        
        for store in target_locations:
            # Calculate distance
            distance = self.processor.calculate_distance(
                params['current_location'], store, self.distances_df
            )
            
            # Predict demand
            predicted_demand = self.system.predict_demand(
                product_id=params['product_id'],
                date=params['date'].strftime('%Y-%m-%d'),
                store_location=store,
                product_name=product_info['product_name'],
                aisle=product_info['aisle'],
                department=product_info['department'],
                product_price=product_info['product_price']
            )
            
            # Calculate logistics cost
            logistics_cost = distance * 0.01  # $0.01 per mile
            
            # Predict profit
            try:
                store_encoded = self.system.label_encoders['store_location'].transform([store])[0] \
                    if store in self.system.label_encoders['store_location'].classes_ else -1
            except:
                store_encoded = -1
            
            profit_features = {
                'predicted_demand': predicted_demand,
                'distance_miles': distance,
                'product_id': params['product_id'],
                'store_location_encoded': store_encoded,
                'distribution_center_encoded': 0,
                'hour': 12,
                'day_of_week': 1,
                'month': 6,
                'is_weekend': 0
            }
            
            profit_df = pd.DataFrame([profit_features])
            predicted_profit = self.system.profit_model.predict(profit_df)[0]
            
            # Enhanced demand score (1-10 scale)
            base_score = min(predicted_demand / 10, 8)  # Cap at 8 for realism
            location_factors = {
                'California': 1.3, 'Texas': 1.2, 'Florida': 1.1, 'New_York': 1.25,
                'Illinois': 1.0, 'Pennsylvania': 0.95, 'Ohio': 0.9, 'Georgia': 1.05,
                'North_Carolina': 0.95, 'Michigan': 0.85
            }
            
            state = store.split('_')[0] if '_' in store else 'Unknown'
            location_factor = location_factors.get(state, 1.0)
            enhanced_score = min(base_score * location_factor, 10)
            
            results.append({
                'store': store,
                'store_name': store.replace('_', ', '),
                'distance_miles': distance,
                'predicted_demand': predicted_demand,
                'predicted_profit': predicted_profit,
                'logistics_cost': logistics_cost,
                'profit_per_mile': predicted_profit / max(distance, 0.1),
                'demand_score_raw': base_score,
                'demand_score_enhanced': enhanced_score,
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by profit and assign ranks
        results.sort(key=lambda x: x['predicted_profit'], reverse=True)
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        return {
            'results': results,
            'product_info': product_info,
            'total_stores': len(results),
            'best_profit': results[0]['predicted_profit'] if results else 0,
            'avg_distance': np.mean([r['distance_miles'] for r in results]) if results else 0
        }
    
    def render_results(self, optimization_results: Dict, params: Dict):
        """Render optimization results."""
        results = optimization_results['results']
        product_info = optimization_results['product_info']
        
        # Limit results based on analysis mode
        if params['analysis_mode'] == "Quick Analysis (Top 10)":
            display_results = results[:10]
        else:
            display_results = results
        
        # Create results dataframe
        results_df = pd.DataFrame(display_results)
        
        # Header with key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🏆 Best Profit", 
                f"${optimization_results['best_profit']:.2f}",
                delta=f"Rank #1"
            )
        
        with col2:
            st.metric(
                "📏 Avg Distance", 
                f"{optimization_results['avg_distance']:.0f} miles",
                delta=f"{optimization_results['total_stores']} stores"
            )
        
        with col3:
            profitable_stores = len([r for r in results if r['predicted_profit'] > 0])
            success_rate = (profitable_stores / len(results)) * 100 if results else 0
            st.metric(
                "✅ Success Rate", 
                f"{success_rate:.1f}%",
                delta=f"{profitable_stores}/{len(results)} profitable"
            )
        
        with col4:
            avg_demand = np.mean([r['demand_score_enhanced'] for r in display_results])
            st.metric(
                "📈 Avg Demand Score", 
                f"{avg_demand:.1f}/10",
                delta="Enhanced"
            )
        
        # Main results section
        st.markdown("## 📊 Optimization Results")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Detailed Results", "📈 Visualizations", "🎯 Insights"])
        
        with tab1:
            self.render_detailed_results(results_df, product_info)
        
        with tab2:
            self.render_visualizations(results_df)
        
        with tab3:
            self.render_insights(results_df, optimization_results, params)
    
    def render_detailed_results(self, results_df: pd.DataFrame, product_info: pd.Series):
        """Render detailed results table."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🏪 Store Rankings")
            
            # Format display dataframe
            display_df = results_df[['rank', 'store_name', 'predicted_profit', 'distance_miles', 
                                   'logistics_cost', 'demand_score_enhanced']].head(10).copy()
            
            display_df.columns = ['Rank', 'Store Location', 'Predicted Profit ($)', 
                                'Distance (miles)', 'Logistics Cost ($)', 'Demand Score']
            
            # Format values
            display_df['Predicted Profit ($)'] = display_df['Predicted Profit ($)'].apply(lambda x: f"${x:.2f}")
            display_df['Distance (miles)'] = display_df['Distance (miles)'].apply(lambda x: f"{x:.1f}")
            display_df['Logistics Cost ($)'] = display_df['Logistics Cost ($)'].apply(lambda x: f"${x:.2f}")
            display_df['Demand Score'] = display_df['Demand Score'].apply(lambda x: f"{x:.1f}/10")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.markdown("### 📦 Product Information")
            
            st.markdown(f"""
            <div class="metric-card">
            <h4>🛍️ {product_info['product_name']}</h4>
            <p><strong>Department:</strong> {product_info['department']}</p>
            <p><strong>Aisle:</strong> {product_info['aisle']}</p>
            <p><strong>Price:</strong> ${product_info['product_price']:.2f}</p>
            <p><strong>Avg Historical Demand:</strong> {product_info['demand']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_visualizations(self, results_df: pd.DataFrame):
        """Render interactive visualizations."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit vs Distance scatter plot
            fig_scatter = self.viz_engine.create_profit_analysis_plot(results_df)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Top stores bar chart
            fig_bar = self.viz_engine.create_top_stores_chart(results_df)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Demand scores comparison
        demand_data = dict(zip(results_df['store_name'], results_df['demand_score_enhanced']))
        fig_demand = self.viz_engine.create_demand_comparison_chart(demand_data)
        st.plotly_chart(fig_demand, use_container_width=True)
    
    def render_insights(self, results_df: pd.DataFrame, optimization_results: Dict, params: Dict):
        """Render business insights and recommendations."""
        st.markdown("### 💡 Key Insights & Recommendations")
        
        # Analysis insights
        best_store = results_df.iloc[0]
        worst_profitable = results_df[results_df['predicted_profit'] > 0].iloc[-1] if any(results_df['predicted_profit'] > 0) else None
        
        insights = []
        
        # Best recommendation
        insights.append(f"🏆 **Top Recommendation**: {best_store['store_name']} offers the highest profit potential at ${best_store['predicted_profit']:.2f}")
        
        # Distance efficiency
        if best_store['profit_per_mile'] > 0:
            insights.append(f"📏 **Distance Efficiency**: Best store provides ${best_store['profit_per_mile']:.3f} profit per mile")
        
        # Demand insights
        high_demand_stores = results_df[results_df['demand_score_enhanced'] >= 7]
        if len(high_demand_stores) > 0:
            insights.append(f"📈 **High Demand Markets**: {len(high_demand_stores)} stores show high demand potential (7+ score)")
        
        # Risk assessment
        negative_profit_stores = len(results_df[results_df['predicted_profit'] < 0])
        if negative_profit_stores > 0:
            insights.append(f"⚠️ **Risk Assessment**: {negative_profit_stores} stores show negative profit margins - avoid these destinations")
        
        # Display insights
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Strategic recommendations
        st.markdown("### 🎯 Strategic Recommendations")
        
        if best_store['predicted_profit'] > 0:
            st.markdown(f"""
            <div class="success-box">
            <h4>✅ Proceed with Shipment</h4>
            Ship to <strong>{best_store['store_name']}</strong> for optimal returns. 
            Expected profit: <strong>${best_store['predicted_profit']:.2f}</strong> with 
            <strong>{best_store['demand_score_enhanced']:.1f}/10</strong> demand score.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <h4>⚠️ Consider Alternative Strategy</h4>
            Current analysis shows limited profit potential. Consider adjusting pricing, 
            exploring different products, or waiting for better market conditions.
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit application."""
        self.render_header()
        
        # Sidebar controls
        params = self.render_sidebar()
        
        # Analysis button
        if st.button("🚀 Analyze Demand & Optimize Logistics", type="primary"):
            with st.spinner("🔄 Analyzing demand patterns and optimizing store selection..."):
                try:
                    optimization_results = self.predict_store_optimization(params)
                    self.render_results(optimization_results, params)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    st.exception(e)


def main():
    """Main application entry point."""
    app = WalmartDemandApp()
    app.run()


if __name__ == "__main__":
    main()
