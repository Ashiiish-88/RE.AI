"""
Visualization Engine
===================

Professional visualization utilities for the Walmart demand prediction system.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


class VisualizationEngine:
    """
    Professional visualization engine for demand prediction analytics.
    
    Features:
    - Interactive Plotly charts
    - Statistical matplotlib plots
    - Business intelligence dashboards
    - Model performance visualization
    """
    
    def __init__(self):
        """Initialize visualization settings."""
        # Set style preferences
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Plotly color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'info': '#17a2b8'
        }
    
    def create_demand_distribution_plot(self, df: pd.DataFrame) -> go.Figure:
        """
        Create demand distribution visualization.
        
        Args:
            df (pd.DataFrame): Dataset with demand column
            
        Returns:
            go.Figure: Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Demand Distribution', 'Demand by Department', 
                          'Demand by Store', 'Demand Over Time'),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Demand distribution histogram
        fig.add_trace(
            go.Histogram(x=df['demand'], name='Demand Distribution', 
                        marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # Demand by department
        dept_demand = df.groupby('department')['demand'].mean().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=dept_demand.values, y=dept_demand.index, 
                  orientation='h', name='Avg Demand by Department',
                  marker_color=self.colors['secondary']),
            row=1, col=2
        )
        
        # Demand by store (top 10)
        store_demand = df.groupby('store_location')['demand'].mean().nlargest(10)
        fig.add_trace(
            go.Bar(x=store_demand.index, y=store_demand.values,
                  name='Top 10 Stores by Demand',
                  marker_color=self.colors['success']),
            row=2, col=1
        )
        
        # Demand over time
        df_time = df.copy()
        df_time['date'] = pd.to_datetime(df_time['date'])
        daily_demand = df_time.groupby(df_time['date'].dt.date)['demand'].mean()
        
        fig.add_trace(
            go.Scatter(x=daily_demand.index, y=daily_demand.values,
                      mode='lines', name='Daily Average Demand',
                      line=dict(color=self.colors['info'])),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="📊 Demand Analysis Dashboard",
            showlegend=False
        )
        
        return fig
    
    def create_profit_analysis_plot(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Create profit vs distance analysis visualization.
        
        Args:
            results_df (pd.DataFrame): Results with profit and distance data
            
        Returns:
            go.Figure: Plotly scatter plot
        """
        # Ensure marker sizes are valid
        profit_per_mile_clean = results_df['profit_per_mile'].fillna(0)
        profit_per_mile_clean = np.where(np.isinf(profit_per_mile_clean), 0, profit_per_mile_clean)
        marker_sizes = np.maximum(np.abs(profit_per_mile_clean), 0.1)
        
        fig = px.scatter(
            results_df,
            x='distance_miles',
            y='predicted_profit',
            size=marker_sizes,
            color='predicted_profit',
            hover_name='store',
            hover_data={
                'predicted_profit': ':$.2f',
                'distance_miles': ':.1f',
                'logistics_cost': ':$.2f'
            },
            title="💰 Profit vs Distance Analysis",
            labels={
                'distance_miles': 'Distance (miles)',
                'predicted_profit': 'Predicted Profit ($)',
                'logistics_cost': 'Logistics Cost ($)'
            },
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_traces(marker=dict(sizemode='diameter', sizemin=10))
        
        fig.update_layout(
            height=500,
            title_font_size=16
        )
        
        return fig
    
    def create_top_stores_chart(self, results_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """
        Create top profitable stores bar chart.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            top_n (int): Number of top stores to show
            
        Returns:
            go.Figure: Plotly bar chart
        """
        top_stores = results_df.head(top_n)
        
        fig = px.bar(
            top_stores,
            x='predicted_profit',
            y='store_name',
            orientation='h',
            color='predicted_profit',
            title=f"🏆 Top {top_n} Most Profitable Stores",
            labels={
                'predicted_profit': 'Predicted Profit ($)',
                'store_name': 'Store Location'
            },
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_demand_comparison_chart(self, demand_data: Dict) -> go.Figure:
        """
        Create demand score comparison chart.
        
        Args:
            demand_data (Dict): Dictionary with store locations and demand scores
            
        Returns:
            go.Figure: Plotly bar chart
        """
        stores = list(demand_data.keys())
        scores = list(demand_data.values())
        
        # Color code based on demand level
        colors = ['#d62728' if score < 4 else '#ff7f0e' if score < 7 else '#2ca02c' 
                 for score in scores]
        
        fig = go.Figure(data=[
            go.Bar(
                x=stores,
                y=scores,
                marker_color=colors,
                text=[f"{score:.1f}" for score in scores],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Demand Score: %{y:.1f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="📈 Demand Scores by Store Location",
            xaxis_title="Store Location",
            yaxis_title="Demand Score (1-10)",
            height=400,
            xaxis_tickangle=-45
        )
        
        # Add reference lines
        fig.add_hline(y=5, line_dash="dash", line_color="orange", 
                     annotation_text="Average")
        fig.add_hline(y=7, line_dash="dash", line_color="green", 
                     annotation_text="Good")
        
        return fig
    
    def create_model_performance_plot(self, results: Dict) -> go.Figure:
        """
        Create model performance comparison visualization.
        
        Args:
            results (Dict): Model training results
            
        Returns:
            go.Figure: Plotly comparison chart
        """
        models = list(results.keys())
        train_r2 = [results[model]['train_r2'] for model in models]
        test_r2 = [results[model]['test_r2'] for model in models]
        overfitting = [results[model]['overfitting'] for model in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Scores', 'Overfitting Analysis'),
            specs=[[{"type": "xy"}, {"type": "xy"}]]
        )
        
        # R² comparison
        fig.add_trace(
            go.Bar(name='Train R²', x=models, y=train_r2, 
                  marker_color=self.colors['primary']),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Test R²', x=models, y=test_r2,
                  marker_color=self.colors['secondary']),
            row=1, col=1
        )
        
        # Overfitting analysis
        colors = ['green' if of < 0.1 else 'orange' if of < 0.2 else 'red' 
                 for of in overfitting]
        fig.add_trace(
            go.Bar(name='Overfitting', x=models, y=overfitting,
                  marker_color=colors),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            title_text="🎯 Model Performance Analysis"
        )
        
        return fig
    
    def create_summary_metrics_plot(self, metrics: Dict) -> go.Figure:
        """
        Create summary metrics visualization.
        
        Args:
            metrics (Dict): Summary metrics
            
        Returns:
            go.Figure: Plotly metrics dashboard
        """
        # Create gauge charts for key metrics
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=('Best Profit', 'Average Distance', 'Total Stores', 'Success Rate')
        )
        
        # Best profit gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = metrics.get('best_profit', 0),
            title = {'text': "Best Profit ($)"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray"}),
            row=1, col=1)
        
        # Average distance
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = metrics.get('avg_distance', 0),
            title = {'text': "Avg Distance (miles)"},
            gauge = {'axis': {'range': [0, 500]},
                    'bar': {'color': "darkblue"}}),
            row=1, col=2)
        
        # Total stores
        fig.add_trace(go.Indicator(
            mode = "number",
            value = metrics.get('total_stores', 0),
            title = {'text': "Total Stores Analyzed"}),
            row=2, col=1)
        
        # Success rate
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = metrics.get('success_rate', 0),
            title = {'text': "Success Rate (%)"},
            gauge = {'axis': {'range': [0, 100]},
                    'bar': {'color': "green"},
                    'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 80], 'color': "gray"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}),
            row=2, col=2)
        
        fig.update_layout(height=600, title_text="📊 Summary Dashboard")
        
        return fig
    
    @staticmethod
    def save_plot(fig, filename: str, format: str = 'png'):
        """
        Save a plotly figure to file.
        
        Args:
            fig: Plotly figure
            filename (str): Output filename
            format (str): File format (png, html, pdf)
        """
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'pdf':
            fig.write_image(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ Plot saved as {filename}")


# Example usage
if __name__ == "__main__":
    viz = VisualizationEngine()
    print("🎨 Visualization engine initialized!")
    print("Available methods:")
    methods = [method for method in dir(viz) if not method.startswith('_')]
    for method in methods:
        print(f"  - {method}")
