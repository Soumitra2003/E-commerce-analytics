"""
Power BI Data Preparation Script
Exports processed data in Power BI-friendly formats

Author: Soumitra Upadhyay
Date: September 2025
Project: E-Commerce Analytics Capstone
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import calculate_business_metrics, generate_data_summary

def prepare_powerbi_data():
    """
    Prepare and export data for Power BI dashboard
    """
    print("🔄 Preparing data for Power BI Dashboard...")
    
    # Create Power BI data directory
    powerbi_dir = '../data/powerbi'
    os.makedirs(powerbi_dir, exist_ok=True)
    
    try:
        # Load processed transaction data
        transactions_df = pd.read_csv('../data/processed/transactions_clean.csv')
        customers_df = pd.read_csv('../data/processed/customers.csv')
        
        # Convert date columns
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
        customers_df['registration_date'] = pd.to_datetime(customers_df['registration_date'])
        
        print(f"✅ Loaded {len(transactions_df):,} transactions and {len(customers_df):,} customers")
        
    except FileNotFoundError:
        print("❌ Data files not found. Please run the data collection notebook first.")
        return
    
    # 1. Main Transactions Table (Fact Table)
    transactions_powerbi = transactions_df.copy()
    
    # Add date dimensions for Power BI
    transactions_powerbi['Year'] = transactions_powerbi['transaction_date'].dt.year
    transactions_powerbi['Month'] = transactions_powerbi['transaction_date'].dt.month
    transactions_powerbi['Quarter'] = transactions_powerbi['transaction_date'].dt.quarter
    transactions_powerbi['Day_of_Week'] = transactions_powerbi['transaction_date'].dt.dayofweek
    transactions_powerbi['Month_Name'] = transactions_powerbi['transaction_date'].dt.month_name()
    transactions_powerbi['Day_Name'] = transactions_powerbi['transaction_date'].dt.day_name()
    transactions_powerbi['Is_Weekend'] = transactions_powerbi['Day_of_Week'].isin([5, 6])
    
    # Add season
    transactions_powerbi['Season'] = transactions_powerbi['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Export main transactions table
    transactions_powerbi.to_csv(f'{powerbi_dir}/Transactions.csv', index=False)
    print("✅ Exported Transactions.csv")
    
    # 2. Customer Dimension Table
    customers_powerbi = customers_df.copy()
    
    # Add customer metrics
    customer_metrics = transactions_df.groupby('customer_id').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'transaction_date': ['min', 'max']
    }).reset_index()
    
    customer_metrics.columns = ['customer_id', 'total_spent', 'avg_order_value', 
                               'total_orders', 'first_purchase', 'last_purchase']
    
    # Calculate customer lifetime and recency
    customer_metrics['customer_lifetime_days'] = (
        customer_metrics['last_purchase'] - customer_metrics['first_purchase']
    ).dt.days
    
    reference_date = transactions_df['transaction_date'].max()
    customer_metrics['recency_days'] = (
        reference_date - customer_metrics['last_purchase']
    ).dt.days
    
    # Create customer segments based on spending
    customer_metrics['spending_tier'] = pd.qcut(
        customer_metrics['total_spent'], 
        4, 
        labels=['Low', 'Medium', 'High', 'Premium']
    )
    
    customer_metrics['order_frequency_tier'] = pd.qcut(
        customer_metrics['total_orders'], 
        4, 
        labels=['Infrequent', 'Occasional', 'Regular', 'Frequent']
    )
    
    # Merge with customer info
    customers_powerbi = customers_powerbi.merge(customer_metrics, on='customer_id', how='left')
    
    # Export customer dimension
    customers_powerbi.to_csv(f'{powerbi_dir}/Customers.csv', index=False)
    print("✅ Exported Customers.csv")
    
    # 3. Product Category Dimension
    category_metrics = transactions_df.groupby('product_category').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'quantity': 'sum'
    }).reset_index()
    
    category_metrics.columns = ['product_category', 'total_revenue', 'avg_price', 
                               'total_transactions', 'total_quantity']
    
    category_metrics['revenue_share'] = (
        category_metrics['total_revenue'] / category_metrics['total_revenue'].sum() * 100
    )
    
    category_metrics.to_csv(f'{powerbi_dir}/Product_Categories.csv', index=False)
    print("✅ Exported Product_Categories.csv")
    
    # 4. Date Dimension Table
    date_range = pd.date_range(
        start=transactions_df['transaction_date'].min(),
        end=transactions_df['transaction_date'].max(),
        freq='D'
    )
    
    date_dim = pd.DataFrame({
        'Date': date_range,
        'Year': date_range.year,
        'Month': date_range.month,
        'Quarter': date_range.quarter,
        'Day': date_range.day,
        'Day_of_Week': date_range.dayofweek,
        'Month_Name': date_range.month_name(),
        'Day_Name': date_range.day_name(),
        'Is_Weekend': date_range.dayofweek.isin([5, 6]),
        'Is_Month_Start': date_range.is_month_start,
        'Is_Month_End': date_range.is_month_end,
        'Week_of_Year': date_range.isocalendar().week
    })
    
    date_dim['Season'] = date_dim['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    date_dim.to_csv(f'{powerbi_dir}/Date_Dimension.csv', index=False)
    print("✅ Exported Date_Dimension.csv")
    
    # 5. Monthly Summary for Trends
    monthly_summary = transactions_df.groupby(
        transactions_df['transaction_date'].dt.to_period('M')
    ).agg({
        'total_amount': 'sum',
        'transaction_id': 'count',
        'customer_id': 'nunique'
    }).reset_index()
    
    monthly_summary.columns = ['Month', 'Total_Revenue', 'Total_Transactions', 'Unique_Customers']
    monthly_summary['Month'] = monthly_summary['Month'].astype(str)
    monthly_summary['Avg_Order_Value'] = monthly_summary['Total_Revenue'] / monthly_summary['Total_Transactions']
    monthly_summary['Revenue_Growth'] = monthly_summary['Total_Revenue'].pct_change() * 100
    
    monthly_summary.to_csv(f'{powerbi_dir}/Monthly_Summary.csv', index=False)
    print("✅ Exported Monthly_Summary.csv")
    
    # 6. Key Metrics Table
    business_metrics = calculate_business_metrics(
        transactions_df,
        customer_col='customer_id',
        amount_col='total_amount',
        date_col='transaction_date'
    )
    
    # Convert to DataFrame for Power BI
    metrics_df = pd.DataFrame([
        {'Metric': 'Total Revenue', 'Value': business_metrics['total_revenue'], 'Format': 'Currency'},
        {'Metric': 'Total Transactions', 'Value': business_metrics['total_transactions'], 'Format': 'Number'},
        {'Metric': 'Unique Customers', 'Value': business_metrics['unique_customers'], 'Format': 'Number'},
        {'Metric': 'Average Order Value', 'Value': business_metrics['avg_order_value'], 'Format': 'Currency'},
        {'Metric': 'Average Customer Value', 'Value': business_metrics['avg_customer_value'], 'Format': 'Currency'},
        {'Metric': 'Customer Retention Rate', 'Value': business_metrics['customer_retention_rate'], 'Format': 'Percentage'},
        {'Metric': 'Average Daily Revenue', 'Value': business_metrics['avg_daily_revenue'], 'Format': 'Currency'},
        {'Metric': 'Average Customer Lifetime', 'Value': business_metrics['avg_customer_lifetime_days'], 'Format': 'Days'}
    ])
    
    metrics_df.to_csv(f'{powerbi_dir}/Key_Metrics.csv', index=False)
    print("✅ Exported Key_Metrics.csv")
    
    # 7. Customer Cohort Analysis
    # Get customer's first purchase month
    first_purchase = transactions_df.groupby('customer_id')['transaction_date'].min().reset_index()
    first_purchase['cohort_month'] = first_purchase['transaction_date'].dt.to_period('M')
    
    # Merge back to transactions
    transactions_cohort = transactions_df.merge(first_purchase[['customer_id', 'cohort_month']], on='customer_id')
    transactions_cohort['transaction_month'] = transactions_cohort['transaction_date'].dt.to_period('M')
    
    # Calculate period number
    transactions_cohort['period_number'] = (
        transactions_cohort['transaction_month'] - transactions_cohort['cohort_month']
    ).apply(lambda x: x.n)
    
    # Create cohort analysis table
    cohort_data = transactions_cohort.groupby(['cohort_month', 'period_number']).agg({
        'customer_id': 'nunique',
        'total_amount': 'sum'
    }).reset_index()
    
    cohort_data.columns = ['Cohort_Month', 'Period_Number', 'Active_Customers', 'Revenue']
    cohort_data['Cohort_Month'] = cohort_data['Cohort_Month'].astype(str)
    
    cohort_data.to_csv(f'{powerbi_dir}/Cohort_Analysis.csv', index=False)
    print("✅ Exported Cohort_Analysis.csv")
    
    # 8. Create Power BI Import Instructions
    instructions = """
# Power BI Data Import Instructions

## Files Created for Power BI:

1. **Transactions.csv** - Main fact table with all transaction details
2. **Customers.csv** - Customer dimension with segmentation and metrics
3. **Product_Categories.csv** - Product category performance metrics
4. **Date_Dimension.csv** - Complete date dimension for time analysis
5. **Monthly_Summary.csv** - Pre-aggregated monthly trends
6. **Key_Metrics.csv** - Business KPIs and metrics
7. **Cohort_Analysis.csv** - Customer retention cohort data

## Power BI Setup Steps:

### 1. Import Data
- Open Power BI Desktop
- Get Data > Text/CSV
- Import all CSV files from the powerbi folder
- Set appropriate data types for each column

### 2. Create Relationships
- Transactions[customer_id] → Customers[customer_id]
- Transactions[transaction_date] → Date_Dimension[Date]
- Transactions[product_category] → Product_Categories[product_category]

### 3. Key Measures to Create:
```
Total Revenue = SUM(Transactions[total_amount])
Total Transactions = COUNT(Transactions[transaction_id])
Unique Customers = DISTINCTCOUNT(Transactions[customer_id])
Average Order Value = DIVIDE([Total Revenue], [Total Transactions])
Revenue Growth = 
    VAR CurrentMonth = [Total Revenue]
    VAR PreviousMonth = CALCULATE([Total Revenue], PREVIOUSMONTH(Date_Dimension[Date]))
    RETURN DIVIDE(CurrentMonth - PreviousMonth, PreviousMonth)
```

### 4. Suggested Visualizations:
- Revenue trend line chart
- Customer segment pie chart
- Product category bar chart
- Geographic map (if location data available)
- Cohort retention heatmap
- KPI cards for key metrics

### 5. Dashboard Pages:
- **Executive Summary** - High-level KPIs and trends
- **Customer Analysis** - Segmentation and behavior
- **Product Performance** - Category and product insights
- **Time Analysis** - Seasonal patterns and trends
"""
    
    with open(f'{powerbi_dir}/PowerBI_Setup_Instructions.md', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("✅ Created Power BI setup instructions")
    
    # Summary
    print("\n📊 POWER BI DATA PREPARATION COMPLETE!")
    print("=" * 50)
    print(f"📁 Files created in: {powerbi_dir}/")
    print("📋 Next steps:")
    print("   1. Download Power BI Desktop (free)")
    print("   2. Import the CSV files")
    print("   3. Follow the setup instructions")
    print("   4. Create your professional dashboard!")
    
    return True

if __name__ == "__main__":
    prepare_powerbi_data()