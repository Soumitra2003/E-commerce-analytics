"""
Data Processing Module for E-Commerce Analytics
Handles data loading, cleaning, and preprocessing operations

Author: Soumitra Upadhyay
Date: September 2025
Project: E-Commerce Analytics Capstone
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Tuple, Dict, List, Optional
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class DataProcessor:
    """
    A comprehensive data processing class for e-commerce analytics
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the DataProcessor with configuration parameters
        
        Args:
            config (Dict): Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.data = None
        self.processed_data = None
        
    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            filepath (str): Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath, **kwargs)
            elif filepath.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(filepath, **kwargs)
            elif filepath.endswith('.json'):
                data = pd.read_json(filepath, **kwargs)
            elif filepath.endswith('.parquet'):
                data = pd.read_parquet(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
            self.data = data
            print(f"✅ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def data_quality_report(self, data: pd.DataFrame = None) -> Dict:
        """
        Generate a comprehensive data quality report
        
        Args:
            data (pd.DataFrame): Data to analyze (uses self.data if None)
            
        Returns:
            Dict: Data quality metrics
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available. Please load data first.")
        
        report = {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # MB
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict(),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object']).columns),
            'datetime_columns': list(data.select_dtypes(include=['datetime64']).columns)
        }
        
        # Summary statistics for numeric columns
        if report['numeric_columns']:
            report['numeric_summary'] = data[report['numeric_columns']].describe().to_dict()
        
        return report
    
    def visualize_missing_data(self, data: pd.DataFrame = None, figsize: Tuple = (12, 8)):
        """
        Visualize missing data patterns
        
        Args:
            data (pd.DataFrame): Data to visualize
            figsize (Tuple): Figure size for plots
        """
        if data is None:
            data = self.data
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')
        
        # Missing data matrix
        msno.matrix(data, ax=axes[0,0])
        axes[0,0].set_title('Missing Data Matrix')
        
        # Missing data bar chart
        msno.bar(data, ax=axes[0,1])
        axes[0,1].set_title('Missing Data Count')
        
        # Missing data heatmap
        msno.heatmap(data, ax=axes[1,0])
        axes[1,0].set_title('Missing Data Correlation')
        
        # Missing data percentage
        missing_pct = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        
        if len(missing_pct) > 0:
            axes[1,1].barh(range(len(missing_pct)), missing_pct.values)
            axes[1,1].set_yticks(range(len(missing_pct)))
            axes[1,1].set_yticklabels(missing_pct.index)
            axes[1,1].set_xlabel('Missing Percentage (%)')
            axes[1,1].set_title('Missing Data Percentage by Column')
        else:
            axes[1,1].text(0.5, 0.5, 'No Missing Data', 
                          horizontalalignment='center', verticalalignment='center',
                          fontsize=14, transform=axes[1,1].transAxes)
            axes[1,1].set_title('Missing Data Status')
        
        plt.tight_layout()
        plt.show()
    
    def handle_missing_values(self, data: pd.DataFrame, strategies: Dict = None) -> pd.DataFrame:
        """
        Handle missing values using various strategies
        
        Args:
            data (pd.DataFrame): Input data
            strategies (Dict): Column-specific strategies for handling missing values
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        data_clean = data.copy()
        
        if strategies is None:
            # Default strategies
            strategies = {
                'numeric': 'median',
                'categorical': 'mode',
                'datetime': 'forward_fill'
            }
        
        # Handle numeric columns
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in strategies:
                strategy = strategies[col]
            else:
                strategy = strategies.get('numeric', 'median')
            
            if strategy == 'mean':
                data_clean[col].fillna(data_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                data_clean[col].fillna(data_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                data_clean[col].fillna(data_clean[col].mode().iloc[0], inplace=True)
            elif strategy == 'zero':
                data_clean[col].fillna(0, inplace=True)
        
        # Handle categorical columns
        categorical_cols = data_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in strategies:
                strategy = strategies[col]
            else:
                strategy = strategies.get('categorical', 'mode')
            
            if strategy == 'mode':
                if not data_clean[col].mode().empty:
                    data_clean[col].fillna(data_clean[col].mode().iloc[0], inplace=True)
            elif strategy == 'unknown':
                data_clean[col].fillna('Unknown', inplace=True)
        
        # Handle datetime columns
        datetime_cols = data_clean.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            if col in strategies:
                strategy = strategies[col]
            else:
                strategy = strategies.get('datetime', 'forward_fill')
            
            if strategy == 'forward_fill':
                data_clean[col].fillna(method='ffill', inplace=True)
            elif strategy == 'backward_fill':
                data_clean[col].fillna(method='bfill', inplace=True)
        
        print(f"✅ Missing values handled. Remaining missing values: {data_clean.isnull().sum().sum()}")
        return data_clean
    
    def detect_outliers(self, data: pd.DataFrame, columns: List[str] = None, 
                       method: str = 'iqr', threshold: float = 1.5) -> Dict:
        """
        Detect outliers in numeric columns
        
        Args:
            data (pd.DataFrame): Input data
            columns (List[str]): Columns to check for outliers
            method (str): Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
            threshold (float): Threshold for outlier detection
            
        Returns:
            Dict: Outlier information for each column
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        outliers_info = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(data) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_indices': outliers.index.tolist()
                }
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outliers = data[z_scores > threshold]
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(data) * 100,
                    'threshold': threshold,
                    'outlier_indices': outliers.index.tolist()
                }
        
        return outliers_info
    
    def remove_duplicates(self, data: pd.DataFrame, subset: List[str] = None, 
                         keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset
        
        Args:
            data (pd.DataFrame): Input data
            subset (List[str]): Columns to consider for identifying duplicates
            keep (str): Which duplicates to keep ('first', 'last', False)
            
        Returns:
            pd.DataFrame: Data with duplicates removed
        """
        initial_count = len(data)
        data_clean = data.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(data_clean)
        
        print(f"✅ Removed {removed_count} duplicate rows ({removed_count/initial_count*100:.2f}%)")
        return data_clean
    
    def convert_data_types(self, data: pd.DataFrame, type_mapping: Dict) -> pd.DataFrame:
        """
        Convert data types for specified columns
        
        Args:
            data (pd.DataFrame): Input data
            type_mapping (Dict): Mapping of column names to target data types
            
        Returns:
            pd.DataFrame: Data with converted types
        """
        data_converted = data.copy()
        
        for col, target_type in type_mapping.items():
            if col in data_converted.columns:
                try:
                    if target_type == 'datetime':
                        data_converted[col] = pd.to_datetime(data_converted[col])
                    elif target_type == 'category':
                        data_converted[col] = data_converted[col].astype('category')
                    else:
                        data_converted[col] = data_converted[col].astype(target_type)
                    print(f"✅ Converted {col} to {target_type}")
                except Exception as e:
                    print(f"❌ Failed to convert {col} to {target_type}: {str(e)}")
        
        return data_converted
    
    def create_date_features(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create additional date-based features from a datetime column
        
        Args:
            data (pd.DataFrame): Input data
            date_column (str): Name of the datetime column
            
        Returns:
            pd.DataFrame: Data with additional date features
        """
        data_with_dates = data.copy()
        
        if date_column not in data_with_dates.columns:
            raise ValueError(f"Column {date_column} not found in data")
        
        # Ensure the column is datetime
        data_with_dates[date_column] = pd.to_datetime(data_with_dates[date_column])
        
        # Extract date components
        data_with_dates[f'{date_column}_year'] = data_with_dates[date_column].dt.year
        data_with_dates[f'{date_column}_month'] = data_with_dates[date_column].dt.month
        data_with_dates[f'{date_column}_day'] = data_with_dates[date_column].dt.day
        data_with_dates[f'{date_column}_dayofweek'] = data_with_dates[date_column].dt.dayofweek
        data_with_dates[f'{date_column}_quarter'] = data_with_dates[date_column].dt.quarter
        data_with_dates[f'{date_column}_weekofyear'] = data_with_dates[date_column].dt.isocalendar().week
        
        # Additional features
        data_with_dates[f'{date_column}_is_weekend'] = data_with_dates[f'{date_column}_dayofweek'].isin([5, 6])
        data_with_dates[f'{date_column}_month_name'] = data_with_dates[date_column].dt.month_name()
        data_with_dates[f'{date_column}_day_name'] = data_with_dates[date_column].dt.day_name()
        
        print(f"✅ Created {8} date features from {date_column}")
        return data_with_dates
    
    def full_preprocessing_pipeline(self, data: pd.DataFrame = None, 
                                  missing_strategies: Dict = None,
                                  outlier_method: str = 'iqr',
                                  remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Execute the complete data preprocessing pipeline
        
        Args:
            data (pd.DataFrame): Input data
            missing_strategies (Dict): Strategies for handling missing values
            outlier_method (str): Method for outlier detection
            remove_duplicates (bool): Whether to remove duplicate rows
            
        Returns:
            pd.DataFrame: Fully processed data
        """
        if data is None:
            data = self.data
            
        if data is None:
            raise ValueError("No data available. Please load data first.")
        
        print("🚀 Starting full preprocessing pipeline...")
        
        # Step 1: Data quality report
        print("\n📊 Generating data quality report...")
        quality_report = self.data_quality_report(data)
        print(f"Initial data shape: {quality_report['shape']}")
        print(f"Missing values: {sum(quality_report['missing_values'].values())}")
        print(f"Duplicate rows: {quality_report['duplicate_rows']}")
        
        # Step 2: Handle missing values
        print("\n🔧 Handling missing values...")
        data_processed = self.handle_missing_values(data, missing_strategies)
        
        # Step 3: Remove duplicates
        if remove_duplicates:
            print("\n🗑️ Removing duplicates...")
            data_processed = self.remove_duplicates(data_processed)
        
        # Step 4: Detect outliers
        print("\n🎯 Detecting outliers...")
        outliers_info = self.detect_outliers(data_processed, method=outlier_method)
        total_outliers = sum([info['count'] for info in outliers_info.values()])
        print(f"Total outliers detected: {total_outliers}")
        
        self.processed_data = data_processed
        print(f"\n✅ Preprocessing complete! Final shape: {data_processed.shape}")
        
        return data_processed