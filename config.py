"""
Configuration settings for E-Commerce Analytics Capstone Project
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data sources
DATA_SOURCES = {
    "ecommerce_transactions": "https://archive.ics.uci.edu/ml/datasets/online+retail",
    "economic_indicators": "https://fred.stlouisfed.org/",
    "product_catalog": "sample_product_data.csv",
}

# Analysis parameters
ANALYSIS_CONFIG = {
    "date_column": "invoice_date",
    "customer_id_column": "customer_id",
    "revenue_column": "total_amount",
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
}

# Customer segmentation parameters
SEGMENTATION_CONFIG = {
    "rfm_quantiles": 5,
    "clustering_algorithms": ["kmeans", "hierarchical"],
    "n_clusters_range": range(2, 11),
    "features_for_clustering": ["recency", "frequency", "monetary"],
}

# Machine learning parameters
ML_CONFIG = {
    "classification_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "regression_metrics": ["mse", "rmse", "mae", "r2"],
    "hyperparameter_tuning": {
        "cv_folds": 3,
        "n_iter": 50,
        "scoring": "roc_auc",
    },
}

# Visualization settings
VIZ_CONFIG = {
    "color_palette": "viridis",
    "figure_size": (12, 8),
    "style": "whitegrid",
    "font_scale": 1.2,
    "dpi": 300,
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "port": 8501,
    "host": "localhost",
    "title": "E-Commerce Analytics Dashboard",
    "layout": "wide",
    "sidebar_width": 300,
}

# File paths for outputs
OUTPUT_PATHS = {
    "customer_segments": PROCESSED_DATA_DIR / "customer_segments.csv",
    "churn_predictions": PROCESSED_DATA_DIR / "churn_predictions.csv",
    "revenue_forecast": PROCESSED_DATA_DIR / "revenue_forecast.csv",
    "executive_report": REPORTS_DIR / "executive_summary.pdf",
    "technical_report": REPORTS_DIR / "technical_report.pdf",
}

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)