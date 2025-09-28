"""
Feature Engineering Module for E-Commerce Analytics
Advanced feature creation and transformation functions

Author: Soumitra Upadhyay
Date: September 2025
Project: E-Commerce Analytics Capstone
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from operator import attrgetter
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Comprehensive feature engineering class for e-commerce analytics
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_rfm_features(self, data: pd.DataFrame, customer_col: str, 
                          date_col: str, amount_col: str, 
                          reference_date: datetime = None) -> pd.DataFrame:
        """
        Create RFM (Recency, Frequency, Monetary) features for customer segmentation
        
        Args:
            data (pd.DataFrame): Transaction data
            customer_col (str): Customer ID column name
            date_col (str): Transaction date column name
            amount_col (str): Transaction amount column name
            reference_date (datetime): Reference date for recency calculation
            
        Returns:
            pd.DataFrame: RFM features by customer
        """
        if reference_date is None:
            reference_date = data[date_col].max()
        
        # Calculate RFM metrics
        rfm = data.groupby(customer_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,  # Recency
            customer_col: 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = [customer_col, 'recency', 'frequency', 'monetary']
        
        # Create RFM scores (1-5 scale)
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Combine RFM scores
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        rfm['rfm_score_numeric'] = rfm['r_score'].astype(int) + rfm['f_score'].astype(int) + rfm['m_score'].astype(int)
        
        # Customer segments based on RFM
        def segment_customers(row):
            if row['rfm_score_numeric'] >= 9 and row['r_score'] >= 3:
                return 'Champions'
            elif row['rfm_score_numeric'] >= 6 and row['r_score'] >= 2:
                return 'Loyal Customers'
            elif row['rfm_score_numeric'] >= 6 and row['r_score'] >= 3:
                return 'Potential Loyalists'
            elif row['rfm_score_numeric'] >= 6 and row['r_score'] <= 2:
                return 'At Risk'
            elif row['rfm_score_numeric'] <= 6 and row['r_score'] <= 2:
                return "Can't Lose Them"
            else:
                return 'Others'
        
        rfm['customer_segment'] = rfm.apply(segment_customers, axis=1)
        
        print(f"✅ Created RFM features for {len(rfm)} customers")
        return rfm
    
    def create_time_based_features(self, data: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Create comprehensive time-based features
        
        Args:
            data (pd.DataFrame): Input data with datetime column
            date_col (str): Name of the datetime column
            
        Returns:
            pd.DataFrame: Data with time-based features
        """
        data_time = data.copy()
        data_time[date_col] = pd.to_datetime(data_time[date_col])
        
        # Basic time features
        data_time['year'] = data_time[date_col].dt.year
        data_time['month'] = data_time[date_col].dt.month
        data_time['day'] = data_time[date_col].dt.day
        data_time['dayofweek'] = data_time[date_col].dt.dayofweek
        data_time['hour'] = data_time[date_col].dt.hour
        data_time['quarter'] = data_time[date_col].dt.quarter
        
        # Advanced time features
        data_time['is_weekend'] = data_time['dayofweek'].isin([5, 6])
        data_time['is_month_start'] = data_time[date_col].dt.is_month_start
        data_time['is_month_end'] = data_time[date_col].dt.is_month_end
        data_time['is_quarter_start'] = data_time[date_col].dt.is_quarter_start
        data_time['is_quarter_end'] = data_time[date_col].dt.is_quarter_end
        
        # Seasonal features
        data_time['season'] = data_time['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Business time features (assuming business hours 9-17)
        data_time['is_business_hours'] = data_time['hour'].between(9, 17)
        
        # Holiday features (basic implementation)
        holidays = ['2023-01-01', '2023-07-04', '2023-12-25']  # Add more holidays
        data_time['is_holiday'] = data_time[date_col].dt.date.astype(str).isin(holidays)
        
        print(f"✅ Created {13} time-based features")
        return data_time
    
    def create_customer_behavior_features(self, data: pd.DataFrame, 
                                        customer_col: str, amount_col: str,
                                        date_col: str) -> pd.DataFrame:
        """
        Create customer behavior and purchasing pattern features
        
        Args:
            data (pd.DataFrame): Transaction data
            customer_col (str): Customer ID column
            amount_col (str): Transaction amount column
            date_col (str): Transaction date column
            
        Returns:
            pd.DataFrame: Customer behavior features
        """
        # Aggregate customer metrics
        customer_features = data.groupby(customer_col).agg({
            amount_col: ['sum', 'mean', 'std', 'count', 'min', 'max'],
            date_col: ['min', 'max', 'count']
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = [
            customer_col, 'total_spent', 'avg_order_value', 'spending_std',
            'total_orders', 'min_order_value', 'max_order_value',
            'first_purchase', 'last_purchase', 'order_frequency'
        ]
        
        # Calculate additional features
        customer_features['customer_lifetime_days'] = (
            customer_features['last_purchase'] - customer_features['first_purchase']
        ).dt.days
        
        customer_features['avg_days_between_orders'] = (
            customer_features['customer_lifetime_days'] / customer_features['total_orders']
        )
        
        customer_features['spending_consistency'] = (
            customer_features['spending_std'] / customer_features['avg_order_value']
        )
        
        # Categorize customers by behavior
        customer_features['spending_tier'] = pd.qcut(
            customer_features['total_spent'], 
            4, 
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        customer_features['order_frequency_tier'] = pd.qcut(
            customer_features['total_orders'], 
            4, 
            labels=['Infrequent', 'Occasional', 'Regular', 'Frequent']
        )
        
        print(f"✅ Created customer behavior features for {len(customer_features)} customers")
        return customer_features
    
    def create_product_features(self, data: pd.DataFrame, product_col: str, 
                              amount_col: str, quantity_col: str = None) -> pd.DataFrame:
        """
        Create product-level features and metrics
        
        Args:
            data (pd.DataFrame): Transaction data
            product_col (str): Product ID column
            amount_col (str): Transaction amount column
            quantity_col (str): Quantity column (optional)
            
        Returns:
            pd.DataFrame: Product features
        """
        # Basic product metrics
        product_features = data.groupby(product_col).agg({
            amount_col: ['sum', 'mean', 'count'],
        }).reset_index()
        
        product_features.columns = [product_col, 'total_revenue', 'avg_price', 'total_sales']
        
        if quantity_col and quantity_col in data.columns:
            quantity_features = data.groupby(product_col)[quantity_col].agg(['sum', 'mean']).reset_index()
            quantity_features.columns = [product_col, 'total_quantity_sold', 'avg_quantity_per_order']
            product_features = product_features.merge(quantity_features, on=product_col)
        
        # Product performance tiers
        product_features['revenue_tier'] = pd.qcut(
            product_features['total_revenue'], 
            5, 
            labels=['Bottom', 'Low', 'Medium', 'High', 'Top']
        )
        
        product_features['sales_tier'] = pd.qcut(
            product_features['total_sales'], 
            5, 
            labels=['Bottom', 'Low', 'Medium', 'High', 'Top']
        )
        
        print(f"✅ Created product features for {len(product_features)} products")
        return product_features
    
    def create_cohort_features(self, data: pd.DataFrame, customer_col: str, 
                             date_col: str, amount_col: str) -> pd.DataFrame:
        """
        Create cohort analysis features
        
        Args:
            data (pd.DataFrame): Transaction data
            customer_col (str): Customer ID column
            date_col (str): Transaction date column
            amount_col (str): Transaction amount column
            
        Returns:
            pd.DataFrame: Cohort features
        """
        # Get customer's first purchase date
        first_purchase = data.groupby(customer_col)[date_col].min().reset_index()
        first_purchase.columns = [customer_col, 'first_purchase_date']
        first_purchase['cohort_month'] = first_purchase['first_purchase_date'].dt.to_period('M')
        
        # Merge with transaction data
        data_cohort = data.merge(first_purchase, on=customer_col)
        data_cohort['transaction_period'] = data_cohort[date_col].dt.to_period('M')
        
        # Calculate period number (months since first purchase)
        data_cohort['period_number'] = (
            data_cohort['transaction_period'] - data_cohort['cohort_month']
        ).apply(attrgetter('n'))
        
        # Create cohort table
        cohort_data = data_cohort.groupby(['cohort_month', 'period_number'])[customer_col].nunique().reset_index()
        cohort_sizes = first_purchase.groupby('cohort_month')[customer_col].nunique()
        
        cohort_table = cohort_data.pivot(index='cohort_month', 
                                       columns='period_number', 
                                       values=customer_col)
        
        # Calculate retention rates
        for i in cohort_table.columns:
            cohort_table[i] = cohort_table[i] / cohort_sizes
        
        print("✅ Created cohort analysis features")
        return cohort_table, data_cohort
    
    def create_statistical_features(self, data: pd.DataFrame, 
                                  numeric_columns: List[str] = None) -> pd.DataFrame:
        """
        Create statistical features from numeric columns
        
        Args:
            data (pd.DataFrame): Input data
            numeric_columns (List[str]): Columns to create features from
            
        Returns:
            pd.DataFrame: Data with statistical features
        """
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        data_stats = data.copy()
        
        for col in numeric_columns:
            # Rolling statistics (if data is time-ordered)
            if len(data) > 10:
                data_stats[f'{col}_rolling_mean_7'] = data[col].rolling(window=7, min_periods=1).mean()
                data_stats[f'{col}_rolling_std_7'] = data[col].rolling(window=7, min_periods=1).std()
                data_stats[f'{col}_rolling_max_7'] = data[col].rolling(window=7, min_periods=1).max()
                data_stats[f'{col}_rolling_min_7'] = data[col].rolling(window=7, min_periods=1).min()
            
            # Lag features
            data_stats[f'{col}_lag_1'] = data[col].shift(1)
            data_stats[f'{col}_lag_7'] = data[col].shift(7)
            
            # Difference features
            data_stats[f'{col}_diff_1'] = data[col].diff(1)
            data_stats[f'{col}_pct_change_1'] = data[col].pct_change(1)
        
        # Interaction features
        if len(numeric_columns) >= 2:
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    data_stats[f'{col1}_{col2}_ratio'] = data[col1] / (data[col2] + 1e-8)
                    data_stats[f'{col1}_{col2}_product'] = data[col1] * data[col2]
        
        print(f"✅ Created statistical features for {len(numeric_columns)} columns")
        return data_stats
    
    def encode_categorical_features(self, data: pd.DataFrame, 
                                  categorical_columns: List[str] = None,
                                  encoding_method: str = 'onehot',
                                  max_categories: int = 10) -> pd.DataFrame:
        """
        Encode categorical features using various methods
        
        Args:
            data (pd.DataFrame): Input data
            categorical_columns (List[str]): Columns to encode
            encoding_method (str): Encoding method ('onehot', 'label', 'target')
            max_categories (int): Maximum categories for one-hot encoding
            
        Returns:
            pd.DataFrame: Data with encoded categorical features
        """
        if categorical_columns is None:
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        data_encoded = data.copy()
        
        for col in categorical_columns:
            unique_values = data_encoded[col].nunique()
            
            if encoding_method == 'onehot' and unique_values <= max_categories:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(data_encoded[col], prefix=col, drop_first=True)
                data_encoded = pd.concat([data_encoded, dummies], axis=1)
                data_encoded.drop(col, axis=1, inplace=True)
                
            elif encoding_method == 'label' or unique_values > max_categories:
                # Label encoding for high cardinality
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    data_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(data_encoded[col].astype(str))
                else:
                    data_encoded[f'{col}_encoded'] = self.encoders[col].transform(data_encoded[col].astype(str))
        
        print(f"✅ Encoded {len(categorical_columns)} categorical features using {encoding_method} method")
        return data_encoded
    
    def scale_features(self, data: pd.DataFrame, numeric_columns: List[str] = None,
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            data (pd.DataFrame): Input data
            numeric_columns (List[str]): Columns to scale
            method (str): Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        data_scaled = data.copy()
        
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        
        for col in numeric_columns:
            if col not in self.scalers:
                self.scalers[col] = scaler
                data_scaled[f'{col}_scaled'] = self.scalers[col].fit_transform(data_scaled[[col]])
            else:
                data_scaled[f'{col}_scaled'] = self.scalers[col].transform(data_scaled[[col]])
        
        print(f"✅ Scaled {len(numeric_columns)} features using {method} scaling")
        return data_scaled
    
    def create_customer_lifetime_value(self, data: pd.DataFrame, customer_col: str,
                                     amount_col: str, date_col: str,
                                     prediction_days: int = 365) -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value (CLV)
        
        Args:
            data (pd.DataFrame): Transaction data
            customer_col (str): Customer ID column
            amount_col (str): Transaction amount column
            date_col (str): Transaction date column
            prediction_days (int): Days to predict CLV for
            
        Returns:
            pd.DataFrame: Customer CLV features
        """
        # Calculate basic metrics per customer
        customer_metrics = data.groupby(customer_col).agg({
            amount_col: ['sum', 'mean', 'count'],
            date_col: ['min', 'max']
        }).reset_index()
        
        customer_metrics.columns = [
            customer_col, 'total_revenue', 'avg_order_value', 'frequency',
            'first_purchase', 'last_purchase'
        ]
        
        # Calculate customer lifespan
        customer_metrics['lifespan_days'] = (
            customer_metrics['last_purchase'] - customer_metrics['first_purchase']
        ).dt.days
        
        # Avoid division by zero
        customer_metrics['lifespan_days'] = customer_metrics['lifespan_days'].replace(0, 1)
        
        # Calculate purchase frequency (purchases per day)
        customer_metrics['purchase_frequency'] = customer_metrics['frequency'] / customer_metrics['lifespan_days']
        
        # Calculate CLV using simple formula: AOV × Frequency × Lifespan
        customer_metrics['clv_historical'] = (
            customer_metrics['avg_order_value'] * customer_metrics['purchase_frequency'] * customer_metrics['lifespan_days']
        )
        
        # Predict future CLV
        customer_metrics['clv_predicted'] = (
            customer_metrics['avg_order_value'] * customer_metrics['purchase_frequency'] * prediction_days
        )
        
        # CLV tiers
        customer_metrics['clv_tier'] = pd.qcut(
            customer_metrics['clv_predicted'], 
            5, 
            labels=['Bottom', 'Low', 'Medium', 'High', 'Top']
        )
        
        print(f"✅ Calculated CLV for {len(customer_metrics)} customers")
        return customer_metrics