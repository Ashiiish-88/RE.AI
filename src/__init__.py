"""
Walmart Demand Prediction & Logistics Optimization System
=========================================================

A comprehensive system for predicting product demand across Walmart stores
and optimizing logistics for maximum profitability.

Main Components:
- DemandPredictionSystem: Core ML models for demand and profit prediction
- StreamlitApp: Interactive web interface
- DataProcessor: Data cleaning and preprocessing utilities
- VisualizationEngine: Charts and analytics visualization

Author: AI Assistant
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .models.demand_system import DemandPredictionSystem
from .app.streamlit_app import WalmartDemandApp
from .utils.data_processor import DataProcessor
from .utils.visualization import VisualizationEngine

__all__ = [
    "DemandPredictionSystem",
    "WalmartDemandApp", 
    "DataProcessor",
    "VisualizationEngine"
]
