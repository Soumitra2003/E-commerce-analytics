"""
Interactive E-Commerce Analytics Dashboard
Built with Streamlit for executive-level business insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization import EcommerceVisualizer
from utils import calculate_business_metrics, generate_data_summary

# Page configuration
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.25rem;
    }
    .insights-box {
        background-color: #e8f4fd;  
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the data"""
    try:
        # Try to load processed data
        transactions = pd.read_csv('../data/processed/transactions_clean.csv')
        customers = pd.read_csv('../data/processed/customers.csv')
        
        # Convert date column
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        customers['registration_date'] = pd.to_datetime(customers['registration_date'])
        
        return transactions, customers
    except FileNotFoundError:
        st.error("Data files not found. Please run the data collection notebook first.")
        return None, None

@st.cache_data  
def calculate_kpis(transactions_df):
    """Calculate key performance indicators"""
    if transactions_df is None:
        return {}
    
    return calculate_business_metrics(
        transactions_df,
        customer_col='customer_id',
        amount_col='total_amount',
        date_col='transaction_date'
    )

def create_kpi_cards(kpis):
    """Create KPI cards layout"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">${kpis.get('total_revenue', 0):,.0f}</div>
                <div class="metric-label">Total Revenue</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{kpis.get('total_transactions', 0):,}</div>
                <div class="metric-label">Total Transactions</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{kpis.get('unique_customers', 0):,}</div>
                <div class="metric-label">Unique Customers</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">${kpis.get('avg_order_value', 0):.0f}</div>
                <div class="metric-label">Average Order Value</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

def create_sales_trend_chart(transactions_df):
    """Create sales trend visualization"""
    # Aggregate daily sales
    daily_sales = transactions_df.groupby(transactions_df['transaction_date'].dt.date).agg({
        'total_amount': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    
    daily_sales.columns = ['date', 'revenue', 'transactions']
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Revenue', 'Daily Transaction Count'),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    # Revenue trend
    fig.add_trace(
        go.Scatter(
            x=daily_sales['date'],
            y=daily_sales['revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Transaction count trend  
    fig.add_trace(
        go.Scatter(
            x=daily_sales['date'],
            y=daily_sales['transactions'],
            mode='lines',
            name='Transactions',
            line=dict(color='#ff7f0e', width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Transactions:</b> %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Sales Performance Over Time',
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    return fig

def create_category_analysis(transactions_df):
    """Create product category analysis"""
    category_metrics = transactions_df.groupby('product_category').agg({
        'total_amount': ['sum', 'count', 'mean']
    }).reset_index()
    
    category_metrics.columns = ['category', 'total_revenue', 'transaction_count', 'avg_order_value']
    category_metrics = category_metrics.sort_values('total_revenue', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by category
        fig1 = px.bar(
            category_metrics,
            x='category',
            y='total_revenue',
            title='Revenue by Product Category',
            color='total_revenue',
            color_continuous_scale='Blues'
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Transaction count by category
        fig2 = px.pie(
            category_metrics,
            values='transaction_count',
            names='category',
            title='Transaction Distribution by Category'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    return category_metrics

def create_customer_analysis(transactions_df):
    """Create customer behavior analysis"""
    # Customer metrics
    customer_metrics = transactions_df.groupby('customer_id').agg({
        'total_amount': ['sum', 'count', 'mean'],
        'transaction_date': 'max'
    }).reset_index()
    
    customer_metrics.columns = ['customer_id', 'total_spent', 'order_count', 'avg_order_value', 'last_purchase']
    
    # Customer value distribution
    fig = px.histogram(
        customer_metrics,
        x='total_spent',
        nbins=30,
        title='Customer Value Distribution',
        labels={'total_spent': 'Total Spent ($)', 'count': 'Number of Customers'}
    )
    fig.update_layout(showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer segments based on spending
    customer_metrics['spending_tier'] = pd.qcut(
        customer_metrics['total_spent'], 
        4, 
        labels=['Low', 'Medium', 'High', 'Premium']
    )
    
    segment_summary = customer_metrics.groupby('spending_tier').agg({
        'customer_id': 'count',
        'total_spent': 'sum',
        'avg_order_value': 'mean'
    }).reset_index()
    
    segment_summary.columns = ['segment', 'customer_count', 'total_revenue', 'avg_order_value']
    
    return customer_metrics, segment_summary

def create_insights_section(kpis, transactions_df):
    """Create business insights section"""
    st.markdown("### 🎯 Business Insights & Recommendations")
    
    # Calculate additional metrics for insights
    revenue_per_customer = kpis['total_revenue'] / kpis['unique_customers']
    retention_rate = kpis['customer_retention_rate']
    
    # Seasonal analysis
    transactions_df['month'] = transactions_df['transaction_date'].dt.month
    monthly_revenue = transactions_df.groupby('month')['total_amount'].sum()
    peak_month = monthly_revenue.idxmax()
    low_month = monthly_revenue.idxmin()
    
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                   7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    
    insights = [
        f"💰 **Revenue Performance**: Average revenue per customer is ${revenue_per_customer:.2f}, indicating {'strong' if revenue_per_customer > 500 else 'moderate'} customer value.",
        
        f"🔄 **Customer Retention**: {retention_rate:.1f}% of customers made repeat purchases, {'above' if retention_rate > 60 else 'below'} industry average.",
        
        f"📈 **Seasonality**: Peak sales month is {month_names[peak_month]}, while {month_names[low_month]} shows lowest performance.",
        
        f"🎯 **Order Frequency**: Customers average {kpis['avg_customer_orders']:.1f} orders with {kpis['avg_purchase_frequency_days']:.0f} days between purchases."
    ]
    
    recommendations = [
        "🚀 **Focus on High-Value Customers**: Implement loyalty programs for premium customers to increase retention.",
        
        "📊 **Improve Low-Performing Periods**: Create targeted campaigns during low-sales months to boost revenue.",
        
        "🎯 **Customer Acquisition**: With strong AOV, invest in customer acquisition to scale revenue growth.",
        
        "📱 **Personalization**: Use customer segments to create personalized marketing campaigns and product recommendations."
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Key Insights")
        for insight in insights:
            st.markdown(f"<div class='insights-box'>{insight}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 💡 Recommendations")
        for rec in recommendations:
            st.markdown(f"<div class='insights-box'>{rec}</div>", unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.title("🛒 E-Commerce Analytics Dashboard")
    st.markdown("**Comprehensive Business Intelligence for Data-Driven Decisions**")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        transactions_df, customers_df = load_data()
    
    if transactions_df is None:
        st.error("Unable to load data. Please ensure data files exist in ../data/processed/")
        return
    
    # Sidebar filters
    st.sidebar.header("📊 Dashboard Filters")
    
    # Date range filter
    min_date = transactions_df['transaction_date'].min().date()
    max_date = transactions_df['transaction_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Category filter
    categories = ['All'] + list(transactions_df['product_category'].unique())
    selected_category = st.sidebar.selectbox("Product Category", categories)
    
    # Filter data based on selections
    filtered_df = transactions_df[
        (transactions_df['transaction_date'].dt.date >= date_range[0]) &
        (transactions_df['transaction_date'].dt.date <= date_range[1])
    ]
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['product_category'] == selected_category]
    
    # Calculate KPIs for filtered data
    kpis = calculate_kpis(filtered_df)
    
    # Display KPIs
    st.markdown("### 📈 Key Performance Indicators")
    create_kpi_cards(kpis)
    st.markdown("---")
    
    # Sales trends
    st.markdown("### 📊 Sales Performance Trends")
    sales_chart = create_sales_trend_chart(filtered_df)
    st.plotly_chart(sales_chart, use_container_width=True)
    st.markdown("---")
    
    # Category analysis
    st.markdown("### 🏷️ Product Category Analysis")
    category_metrics = create_category_analysis(filtered_df)
    
    # Show category metrics table
    st.markdown("#### Category Performance Summary")
    st.dataframe(
        category_metrics.style.format({
            'total_revenue': '${:,.2f}',
            'avg_order_value': '${:.2f}'
        }),
        use_container_width=True
    )
    st.markdown("---")
    
    # Customer analysis
    st.markdown("### 👥 Customer Behavior Analysis")
    customer_metrics, segment_summary = create_customer_analysis(filtered_df)
    
    # Customer segments summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Customer Segments")
        st.dataframe(
            segment_summary.style.format({
                'total_revenue': '${:,.2f}',
                'avg_order_value': '${:.2f}'
            }),
            use_container_width=True
        )
    
    with col2:
        # Top customers
        st.markdown("#### Top 10 Customers")
        top_customers = customer_metrics.nlargest(10, 'total_spent')[['customer_id', 'total_spent', 'order_count']]
        st.dataframe(
            top_customers.style.format({'total_spent': '${:,.2f}'}),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Business insights
    create_insights_section(kpis, filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        📊 E-Commerce Analytics Dashboard | Built with Streamlit & Python<br>
        🎯 Capstone Project: Data Analytics for Business Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()