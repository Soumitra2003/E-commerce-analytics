"""
Utility functions for E-Commerce Analytics
Helper functions for data processing, analysis, and reporting

Author: Soumitra Upadhyay
Date: September 2025
Project: E-Commerce Analytics Capstone
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_level (str): Logging level
        log_file (str): Optional log file path
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger('ecommerce_analytics')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def validate_data_schema(data: pd.DataFrame, required_columns: List[str],
                        column_types: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Validate data schema and return validation report
    
    Args:
        data (pd.DataFrame): Data to validate
        required_columns (List[str]): Required column names
        column_types (Dict[str, str]): Expected column data types
        
    Returns:
        Dict[str, Any]: Validation report
    """
    report = {
        'is_valid': True,
        'missing_columns': [],
        'extra_columns': [],
        'type_mismatches': [],
        'data_quality_issues': []
    }
    
    # Check required columns
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        report['missing_columns'] = list(missing_cols)
        report['is_valid'] = False
    
    # Check for extra columns
    extra_cols = set(data.columns) - set(required_columns)
    if extra_cols:
        report['extra_columns'] = list(extra_cols)
    
    # Check data types
    if column_types:
        for col, expected_type in column_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if not actual_type.startswith(expected_type):
                    report['type_mismatches'].append({
                        'column': col,
                        'expected': expected_type,
                        'actual': actual_type
                    })
    
    # Check for data quality issues
    for col in data.columns:
        null_count = data[col].isnull().sum()
        if null_count > 0:
            report['data_quality_issues'].append({
                'column': col,
                'issue': 'missing_values',
                'count': int(null_count),
                'percentage': float(null_count / len(data) * 100)
            })
    
    return report

def generate_data_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data summary
    
    Args:
        data (pd.DataFrame): Input data
        
    Returns:
        Dict[str, Any]: Data summary
    """
    summary = {
        'basic_info': {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
            'column_count': len(data.columns),
            'row_count': len(data)
        },
        'data_types': data.dtypes.value_counts().to_dict(),
        'missing_data': {
            'total_missing': data.isnull().sum().sum(),
            'columns_with_missing': data.columns[data.isnull().any()].tolist(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict()
        },
        'duplicates': {
            'duplicate_rows': data.duplicated().sum(),
            'duplicate_percentage': data.duplicated().sum() / len(data) * 100
        }
    }
    
    # Numeric columns summary
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = data[numeric_cols].describe().to_dict()
    
    # Categorical columns summary
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        summary['categorical_summary'] = {}
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_count': data[col].nunique(),
                'most_frequent': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                'top_values': data[col].value_counts().head().to_dict()
            }
    
    return summary

def calculate_business_metrics(data: pd.DataFrame, customer_col: str,
                             amount_col: str, date_col: str) -> Dict[str, float]:
    """
    Calculate key business metrics
    
    Args:
        data (pd.DataFrame): Transaction data
        customer_col (str): Customer ID column
        amount_col (str): Amount column
        date_col (str): Date column
        
    Returns:
        Dict[str, float]: Business metrics
    """
    # Basic metrics
    total_revenue = data[amount_col].sum()
    total_transactions = len(data)
    unique_customers = data[customer_col].nunique()
    avg_order_value = data[amount_col].mean()
    
    # Customer metrics
    customer_data = data.groupby(customer_col).agg({
        amount_col: ['sum', 'count', 'mean'],
        date_col: ['min', 'max']
    }).reset_index()
    
    customer_data.columns = [customer_col, 'total_spent', 'total_orders', 
                           'avg_order_value', 'first_purchase', 'last_purchase']
    
    avg_customer_value = customer_data['total_spent'].mean()
    avg_customer_orders = customer_data['total_orders'].mean()
    
    # Time-based metrics
    data_with_dates = data.copy()
    data_with_dates[date_col] = pd.to_datetime(data_with_dates[date_col])
    
    date_range = (data_with_dates[date_col].max() - data_with_dates[date_col].min()).days
    avg_daily_revenue = total_revenue / max(date_range, 1)
    avg_daily_transactions = total_transactions / max(date_range, 1)
    
    # Customer lifetime metrics
    customer_data['lifetime_days'] = (customer_data['last_purchase'] - 
                                    customer_data['first_purchase']).dt.days
    avg_customer_lifetime = customer_data['lifetime_days'].mean()
    
    # Purchase frequency
    avg_days_between_purchases = customer_data['lifetime_days'] / customer_data['total_orders']
    avg_purchase_frequency = avg_days_between_purchases.mean()
    
    return {
        'total_revenue': float(total_revenue),
        'total_transactions': int(total_transactions),
        'unique_customers': int(unique_customers),
        'avg_order_value': float(avg_order_value),
        'avg_customer_value': float(avg_customer_value),
        'avg_customer_orders': float(avg_customer_orders),
        'avg_daily_revenue': float(avg_daily_revenue),
        'avg_daily_transactions': float(avg_daily_transactions),
        'avg_customer_lifetime_days': float(avg_customer_lifetime),
        'avg_purchase_frequency_days': float(avg_purchase_frequency),
        'customer_retention_rate': float(len(customer_data[customer_data['total_orders'] > 1]) / len(customer_data) * 100)
    }

def detect_seasonality(data: pd.DataFrame, date_col: str, 
                      amount_col: str, period: str = 'M') -> Dict[str, Any]:
    """
    Detect seasonal patterns in sales data
    
    Args:
        data (pd.DataFrame): Sales data
        date_col (str): Date column
        amount_col (str): Amount column
        period (str): Aggregation period ('D', 'W', 'M', 'Q')
        
    Returns:
        Dict[str, Any]: Seasonality analysis
    """
    # Prepare data
    data_seasonal = data.copy()
    data_seasonal[date_col] = pd.to_datetime(data_seasonal[date_col])
    
    # Aggregate by period
    seasonal_data = data_seasonal.groupby(pd.Grouper(key=date_col, freq=period))[amount_col].sum()
    
    # Calculate seasonal indices
    if period == 'M':
        # Monthly seasonality
        monthly_avg = seasonal_data.groupby(seasonal_data.index.month).mean()
        overall_avg = seasonal_data.mean()
        seasonal_indices = (monthly_avg / overall_avg).to_dict()
        
    elif period == 'D':
        # Daily seasonality (day of week)
        daily_avg = seasonal_data.groupby(seasonal_data.index.dayofweek).mean()
        overall_avg = seasonal_data.mean()
        seasonal_indices = (daily_avg / overall_avg).to_dict()
        
    else:
        seasonal_indices = {}
    
    # Identify peak and low periods
    if seasonal_indices:
        peak_period = max(seasonal_indices, key=seasonal_indices.get)
        low_period = min(seasonal_indices, key=seasonal_indices.get)
        seasonality_strength = max(seasonal_indices.values()) - min(seasonal_indices.values())
    else:
        peak_period = None
        low_period = None
        seasonality_strength = 0
    
    return {
        'seasonal_indices': seasonal_indices,
        'peak_period': peak_period,
        'low_period': low_period,
        'seasonality_strength': float(seasonality_strength),
        'period': period,
        'data_points': len(seasonal_data)
    }

def create_cohort_table(data: pd.DataFrame, customer_col: str,
                       date_col: str, period_type: str = 'M') -> pd.DataFrame:
    """
    Create cohort analysis table
    
    Args:
        data (pd.DataFrame): Transaction data
        customer_col (str): Customer ID column
        date_col (str): Date column
        period_type (str): Period type ('M' for monthly, 'W' for weekly)
        
    Returns:
        pd.DataFrame: Cohort table with retention rates
    """
    # Get customer's first purchase date
    data_cohort = data.copy()
    data_cohort[date_col] = pd.to_datetime(data_cohort[date_col])
    
    # Determine cohort of each customer
    customer_cohorts = data_cohort.groupby(customer_col)[date_col].min().reset_index()
    customer_cohorts.columns = [customer_col, 'cohort_date']
    customer_cohorts['cohort_period'] = customer_cohorts['cohort_date'].dt.to_period(period_type)
    
    # Merge back to original data
    data_cohort = data_cohort.merge(customer_cohorts, on=customer_col)
    data_cohort['transaction_period'] = data_cohort[date_col].dt.to_period(period_type)
    
    # Calculate period number
    data_cohort['period_number'] = (
        data_cohort['transaction_period'] - data_cohort['cohort_period']
    ).apply(lambda x: x.n)
    
    # Create cohort table
    cohort_data = data_cohort.groupby(['cohort_period', 'period_number'])[customer_col].nunique().reset_index()
    cohort_sizes = customer_cohorts.groupby('cohort_period')[customer_col].nunique()
    
    cohort_table = cohort_data.pivot(index='cohort_period', 
                                   columns='period_number', 
                                   values=customer_col)
    
    # Calculate retention rates
    for i in cohort_table.columns:
        cohort_table[i] = cohort_table[i] / cohort_sizes
    
    return cohort_table

def calculate_statistical_significance(data1: pd.Series, data2: pd.Series,
                                     test_type: str = 'ttest') -> Dict[str, float]:
    """
    Calculate statistical significance between two groups
    
    Args:
        data1 (pd.Series): First group data
        data2 (pd.Series): Second group data
        test_type (str): Type of test ('ttest', 'mannwhitney', 'chi2')
        
    Returns:
        Dict[str, float]: Test results
    """
    from scipy import stats
    
    if test_type == 'ttest':
        statistic, p_value = stats.ttest_ind(data1.dropna(), data2.dropna())
        test_name = 'Independent t-test'
        
    elif test_type == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(data1.dropna(), data2.dropna())
        test_name = 'Mann-Whitney U test'
        
    else:
        raise ValueError(f"Unsupported test type: {test_type}")
    
    # Effect size (Cohen's d for t-test)
    if test_type == 'ttest':
        pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + 
                             (len(data2) - 1) * data2.var()) / 
                            (len(data1) + len(data2) - 2))
        effect_size = (data1.mean() - data2.mean()) / pooled_std
    else:
        effect_size = None
    
    return {
        'test_name': test_name,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_significant': p_value < 0.05,
        'effect_size': float(effect_size) if effect_size is not None else None
    }

def export_results_to_json(results: Dict, filepath: str) -> None:
    """
    Export analysis results to JSON file
    
    Args:
        results (Dict): Results dictionary
        filepath (str): Output file path
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    print(f"✅ Results exported to {filepath}")

def create_project_report(analysis_results: Dict, 
                         output_path: str = "project_report.md") -> None:
    """
    Create a comprehensive project report in Markdown format
    
    Args:
        analysis_results (Dict): Combined analysis results
        output_path (str): Output file path
    """
    report_content = f"""# E-Commerce Analytics Project Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the findings from a comprehensive analysis of e-commerce transaction data, 
focusing on customer behavior, sales trends, and business performance metrics.

## Key Findings

"""
    
    # Add business metrics if available
    if 'business_metrics' in analysis_results:
        metrics = analysis_results['business_metrics']
        report_content += f"""
### Business Performance Metrics

- **Total Revenue**: ${metrics.get('total_revenue', 0):,.2f}
- **Total Transactions**: {metrics.get('total_transactions', 0):,}
- **Unique Customers**: {metrics.get('unique_customers', 0):,}
- **Average Order Value**: ${metrics.get('avg_order_value', 0):.2f}
- **Average Customer Value**: ${metrics.get('avg_customer_value', 0):.2f}
- **Customer Retention Rate**: {metrics.get('customer_retention_rate', 0):.1f}%

"""
    
    # Add customer segmentation results if available
    if 'customer_segments' in analysis_results:
        report_content += """
### Customer Segmentation

The analysis identified distinct customer segments based on RFM (Recency, Frequency, Monetary) analysis:

"""
        # Add segment details here
    
    # Add seasonality findings if available
    if 'seasonality' in analysis_results:
        seasonality = analysis_results['seasonality']
        report_content += f"""
### Seasonal Patterns

- **Peak Period**: {seasonality.get('peak_period', 'N/A')}
- **Low Period**: {seasonality.get('low_period', 'N/A')}
- **Seasonality Strength**: {seasonality.get('seasonality_strength', 0):.2f}

"""
    
    report_content += """
## Methodology

This analysis employed the following techniques:

1. **Data Preprocessing**: Data cleaning, missing value imputation, and outlier detection
2. **Exploratory Data Analysis**: Statistical analysis and data visualization
3. **Customer Segmentation**: RFM analysis and clustering techniques
4. **Predictive Modeling**: Machine learning models for churn prediction and CLV
5. **Time Series Analysis**: Trend analysis and seasonality detection

## Recommendations

Based on the analysis, the following recommendations are proposed:

1. **Customer Retention**: Focus on high-value customers at risk of churning
2. **Marketing Strategy**: Tailor campaigns to different customer segments
3. **Inventory Management**: Adjust stock levels based on seasonal patterns
4. **Pricing Strategy**: Optimize pricing based on customer value segments

## Technical Implementation

The analysis was implemented using:
- **Python**: Data processing and analysis
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Machine learning models
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **Jupyter Notebooks**: Interactive analysis

## Conclusion

This comprehensive analysis provides actionable insights for business decision-making
and demonstrates the value of data-driven approaches in e-commerce analytics.

---

*This report was generated automatically as part of the E-Commerce Analytics Capstone Project.*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Project report created: {output_path}")

def benchmark_model_performance(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a benchmark comparison of different models
    
    Args:
        results (Dict[str, Dict]): Model results dictionary
        
    Returns:
        pd.DataFrame: Benchmark comparison table
    """
    benchmark_data = []
    
    for model_name, model_results in results.items():
        if 'metrics' in model_results:
            metrics = model_results['metrics']
            row = {'model': model_name}
            row.update(metrics)
            benchmark_data.append(row)
    
    if benchmark_data:
        benchmark_df = pd.DataFrame(benchmark_data)
        benchmark_df = benchmark_df.round(4)
        return benchmark_df
    else:
        return pd.DataFrame()

def create_data_dictionary(data: pd.DataFrame, 
                          descriptions: Dict[str, str] = None) -> pd.DataFrame:
    """
    Create a data dictionary for the dataset
    
    Args:
        data (pd.DataFrame): Input dataset
        descriptions (Dict[str, str]): Column descriptions
        
    Returns:
        pd.DataFrame: Data dictionary
    """
    data_dict = []
    
    for col in data.columns:
        col_info = {
            'column_name': col,
            'data_type': str(data[col].dtype),
            'non_null_count': data[col].count(),
            'null_count': data[col].isnull().sum(),
            'unique_values': data[col].nunique(),
            'description': descriptions.get(col, '') if descriptions else ''
        }
        
        # Add sample values for categorical columns
        if data[col].dtype == 'object' or data[col].dtype.name == 'category':
            sample_values = data[col].dropna().unique()[:5]
            col_info['sample_values'] = ', '.join(map(str, sample_values))
        
        # Add statistics for numeric columns
        elif np.issubdtype(data[col].dtype, np.number):
            col_info['min_value'] = data[col].min()
            col_info['max_value'] = data[col].max()
            col_info['mean_value'] = data[col].mean()
            col_info['std_value'] = data[col].std()
        
        data_dict.append(col_info)
    
    return pd.DataFrame(data_dict)