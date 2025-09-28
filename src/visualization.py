"""
Visualization Module for E-Commerce Analytics
Custom plotting functions and dashboard components

Author: Soumitra Upadhyay
Date: September 2025
Project: E-Commerce Analytics Capstone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style defaults
plt.style.use('default')
sns.set_palette("husl")

class EcommerceVisualizer:
    """
    Comprehensive visualization class for e-commerce analytics
    """
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'husl', 
                 figsize: Tuple = (12, 8), dpi: int = 300):
        """
        Initialize the visualizer with style preferences
        
        Args:
            style (str): Seaborn style
            palette (str): Color palette
            figsize (Tuple): Default figure size
            dpi (int): Figure DPI
        """
        sns.set_style(style)
        sns.set_palette(palette)
        self.figsize = figsize
        self.dpi = dpi
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def plot_customer_segments(self, data: pd.DataFrame, 
                             segment_col: str = 'customer_segment',
                             metrics: List[str] = ['recency', 'frequency', 'monetary']) -> None:
        """
        Visualize customer segments with multiple metrics
        
        Args:
            data (pd.DataFrame): Customer data with segments
            segment_col (str): Column containing segment labels
            metrics (List[str]): Metrics to visualize by segment
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # Segment distribution
        segment_counts = data[segment_col].value_counts()
        axes[0,0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Customer Segment Distribution')
        
        # Average metrics by segment
        segment_metrics = data.groupby(segment_col)[metrics].mean()
        segment_metrics.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Average Metrics by Segment')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Heatmap of segment characteristics
        segment_heatmap = segment_metrics.T
        sns.heatmap(segment_heatmap, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1,0])
        axes[1,0].set_title('Segment Characteristics Heatmap')
        
        # Box plot for key metric
        if len(metrics) > 0:
            sns.boxplot(data=data, x=segment_col, y=metrics[0], ax=axes[1,1])
            axes[1,1].set_title(f'{metrics[0].title()} Distribution by Segment')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_rfm_analysis(self, rfm_data: pd.DataFrame) -> None:
        """
        Create comprehensive RFM analysis visualizations
        
        Args:
            rfm_data (pd.DataFrame): RFM data with recency, frequency, monetary columns
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RFM Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Recency distribution
        axes[0,0].hist(rfm_data['recency'], bins=30, alpha=0.7, color=self.colors['primary'])
        axes[0,0].set_title('Recency Distribution')
        axes[0,0].set_xlabel('Days Since Last Purchase')
        axes[0,0].set_ylabel('Number of Customers')
        
        # Frequency distribution
        axes[0,1].hist(rfm_data['frequency'], bins=30, alpha=0.7, color=self.colors['secondary'])
        axes[0,1].set_title('Frequency Distribution')
        axes[0,1].set_xlabel('Number of Purchases')
        axes[0,1].set_ylabel('Number of Customers')
        
        # Monetary distribution
        axes[0,2].hist(rfm_data['monetary'], bins=30, alpha=0.7, color=self.colors['success'])
        axes[0,2].set_title('Monetary Distribution')
        axes[0,2].set_xlabel('Total Spend')
        axes[0,2].set_ylabel('Number of Customers')
        
        # RFM Score distribution
        if 'rfm_score_numeric' in rfm_data.columns:
            rfm_score_dist = rfm_data['rfm_score_numeric'].value_counts().sort_index()
            axes[1,0].bar(rfm_score_dist.index, rfm_score_dist.values, color=self.colors['info'])
            axes[1,0].set_title('RFM Score Distribution')
            axes[1,0].set_xlabel('RFM Score')
            axes[1,0].set_ylabel('Number of Customers')
        
        # Frequency vs Monetary scatter
        axes[1,1].scatter(rfm_data['frequency'], rfm_data['monetary'], 
                         alpha=0.6, color=self.colors['warning'])
        axes[1,1].set_title('Frequency vs Monetary Value')
        axes[1,1].set_xlabel('Frequency')
        axes[1,1].set_ylabel('Monetary Value')
        
        # Customer segments pie chart
        if 'customer_segment' in rfm_data.columns:
            segment_counts = rfm_data['customer_segment'].value_counts()
            axes[1,2].pie(segment_counts.values, labels=segment_counts.index, 
                         autopct='%1.1f%%', startangle=90)
            axes[1,2].set_title('Customer Segments')
        
        plt.tight_layout()
        plt.show()
    
    def plot_sales_trends(self, data: pd.DataFrame, date_col: str, 
                         amount_col: str, period: str = 'M') -> None:
        """
        Plot sales trends over time
        
        Args:
            data (pd.DataFrame): Sales data
            date_col (str): Date column name
            amount_col (str): Sales amount column name
            period (str): Aggregation period ('D', 'W', 'M', 'Q')
        """
        # Prepare data
        data_trend = data.copy()
        data_trend[date_col] = pd.to_datetime(data_trend[date_col])
        
        # Aggregate by period
        sales_trend = data_trend.groupby(pd.Grouper(key=date_col, freq=period))[amount_col].agg(['sum', 'count']).reset_index()
        sales_trend.columns = [date_col, 'total_sales', 'transaction_count']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Sales Trends Analysis', fontsize=16, fontweight='bold')
        
        # Total sales trend
        axes[0,0].plot(sales_trend[date_col], sales_trend['total_sales'], 
                      marker='o', linewidth=2, color=self.colors['primary'])
        axes[0,0].set_title('Total Sales Over Time')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Total Sales')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True)
        
        # Transaction count trend
        axes[0,1].plot(sales_trend[date_col], sales_trend['transaction_count'], 
                      marker='s', linewidth=2, color=self.colors['secondary'])
        axes[0,1].set_title('Transaction Count Over Time')
        axes[0,1].set_xlabel('Date')
        axes[0,1].set_ylabel('Number of Transactions')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True)
        
        # Average order value trend
        sales_trend['avg_order_value'] = sales_trend['total_sales'] / sales_trend['transaction_count']
        axes[1,0].plot(sales_trend[date_col], sales_trend['avg_order_value'], 
                      marker='^', linewidth=2, color=self.colors['success'])
        axes[1,0].set_title('Average Order Value Over Time')
        axes[1,0].set_xlabel('Date')
        axes[1,0].set_ylabel('Average Order Value')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True)
        
        # Sales growth rate
        sales_trend['growth_rate'] = sales_trend['total_sales'].pct_change() * 100
        axes[1,1].bar(sales_trend[date_col], sales_trend['growth_rate'], 
                     color=self.colors['info'], alpha=0.7)
        axes[1,1].axhline(y=0, color='red', linestyle='--')
        axes[1,1].set_title('Sales Growth Rate (%)')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('Growth Rate (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cohort_analysis(self, cohort_table: pd.DataFrame) -> None:
        """
        Create cohort retention heatmap
        
        Args:
            cohort_table (pd.DataFrame): Cohort analysis table
        """
        plt.figure(figsize=(15, 8))
        
        # Create heatmap
        sns.heatmap(cohort_table, annot=True, fmt='.1%', cmap='YlOrRd', 
                   linewidths=0.5, linecolor='white')
        
        plt.title('Customer Cohort Retention Rates', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Period Number (Months)', fontsize=12)
        plt.ylabel('Cohort Month', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_product_analysis(self, data: pd.DataFrame, product_col: str,
                            amount_col: str, top_n: int = 20) -> None:
        """
        Analyze and visualize product performance
        
        Args:
            data (pd.DataFrame): Product sales data
            product_col (str): Product column name
            amount_col (str): Sales amount column
            top_n (int): Number of top products to show
        """
        # Calculate product metrics
        product_metrics = data.groupby(product_col).agg({
            amount_col: ['sum', 'count', 'mean']
        }).reset_index()
        
        product_metrics.columns = [product_col, 'total_revenue', 'total_sales', 'avg_price']
        
        # Get top products
        top_products_revenue = product_metrics.nlargest(top_n, 'total_revenue')
        top_products_sales = product_metrics.nlargest(top_n, 'total_sales')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Product Performance Analysis', fontsize=16, fontweight='bold')
        
        # Top products by revenue
        axes[0,0].barh(range(len(top_products_revenue)), top_products_revenue['total_revenue'])
        axes[0,0].set_yticks(range(len(top_products_revenue)))
        axes[0,0].set_yticklabels(top_products_revenue[product_col])
        axes[0,0].set_title(f'Top {top_n} Products by Revenue')
        axes[0,0].set_xlabel('Total Revenue')
        
        # Top products by sales count
        axes[0,1].barh(range(len(top_products_sales)), top_products_sales['total_sales'])
        axes[0,1].set_yticks(range(len(top_products_sales)))
        axes[0,1].set_yticklabels(top_products_sales[product_col])
        axes[0,1].set_title(f'Top {top_n} Products by Sales Count')
        axes[0,1].set_xlabel('Total Sales')
        
        # Revenue distribution
        axes[1,0].hist(product_metrics['total_revenue'], bins=30, alpha=0.7, 
                      color=self.colors['primary'])
        axes[1,0].set_title('Product Revenue Distribution')
        axes[1,0].set_xlabel('Total Revenue')
        axes[1,0].set_ylabel('Number of Products')
        
        # Price vs Sales scatter
        axes[1,1].scatter(product_metrics['avg_price'], product_metrics['total_sales'], 
                         alpha=0.6, color=self.colors['secondary'])
        axes[1,1].set_title('Average Price vs Total Sales')
        axes[1,1].set_xlabel('Average Price')
        axes[1,1].set_ylabel('Total Sales')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard_data(self, data: pd.DataFrame,
                                        date_col: str, amount_col: str,
                                        customer_col: str) -> Dict:
        """
        Prepare data for interactive dashboard
        
        Args:
            data (pd.DataFrame): Transaction data
            date_col (str): Date column name
            amount_col (str): Amount column name
            customer_col (str): Customer column name
            
        Returns:
            Dict: Processed data for dashboard components
        """
        # Daily sales trend
        daily_sales = data.groupby(data[date_col].dt.date)[amount_col].agg(['sum', 'count']).reset_index()
        daily_sales.columns = ['date', 'total_sales', 'transaction_count']
        
        # Monthly trends
        monthly_sales = data.groupby(data[date_col].dt.to_period('M'))[amount_col].agg(['sum', 'count']).reset_index()
        monthly_sales['date'] = monthly_sales[date_col].astype(str)
        monthly_sales.columns = ['period', 'date', 'total_sales', 'transaction_count']
        
        # Customer metrics
        customer_metrics = data.groupby(customer_col).agg({
            amount_col: ['sum', 'count', 'mean'],
            date_col: ['min', 'max']
        }).reset_index()
        
        customer_metrics.columns = [customer_col, 'total_spent', 'total_orders', 
                                  'avg_order_value', 'first_purchase', 'last_purchase']
        
        # Key performance indicators
        kpis = {
            'total_revenue': data[amount_col].sum(),
            'total_transactions': len(data),
            'unique_customers': data[customer_col].nunique(),
            'avg_order_value': data[amount_col].mean(),
            'avg_customer_value': customer_metrics['total_spent'].mean()
        }
        
        return {
            'daily_sales': daily_sales,
            'monthly_sales': monthly_sales,
            'customer_metrics': customer_metrics,
            'kpis': kpis
        }
    
    def plot_interactive_sales_trend(self, daily_sales: pd.DataFrame) -> go.Figure:
        """
        Create interactive sales trend plot using Plotly
        
        Args:
            daily_sales (pd.DataFrame): Daily sales data
            
        Returns:
            go.Figure: Interactive Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Sales Revenue', 'Daily Transaction Count'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Sales revenue
        fig.add_trace(
            go.Scatter(
                x=daily_sales['date'],
                y=daily_sales['total_sales'],
                mode='lines+markers',
                name='Sales Revenue',
                line=dict(color=self.colors['primary'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Transaction count
        fig.add_trace(
            go.Scatter(
                x=daily_sales['date'],
                y=daily_sales['transaction_count'],
                mode='lines+markers',
                name='Transaction Count',
                line=dict(color=self.colors['secondary'], width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Transactions:</b> %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Interactive Sales Trends',
            xaxis_title='Date',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        
        return fig
    
    def plot_customer_value_distribution(self, customer_metrics: pd.DataFrame) -> go.Figure:
        """
        Create interactive customer value distribution
        
        Args:
            customer_metrics (pd.DataFrame): Customer metrics data
            
        Returns:
            go.Figure: Interactive Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Customer Value Distribution', 'Order Frequency vs Value'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram of customer values
        fig.add_trace(
            go.Histogram(
                x=customer_metrics['total_spent'],
                nbinsx=30,
                name='Customer Value',
                marker_color=self.colors['primary'],
                opacity=0.7,
                hovertemplate='<b>Value Range:</b> $%{x}<br><b>Customers:</b> %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Scatter plot: orders vs value
        fig.add_trace(
            go.Scatter(
                x=customer_metrics['total_orders'],
                y=customer_metrics['total_spent'],
                mode='markers',
                name='Customer Scatter',
                marker=dict(
                    size=8,
                    color=customer_metrics['avg_order_value'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Order Value")
                ),
                hovertemplate='<b>Orders:</b> %{x}<br><b>Total Spent:</b> $%{y:,.2f}<br><b>Avg Order:</b> $%{marker.color:,.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Customer Analysis Dashboard',
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Total Spent ($)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Customers", row=1, col=1)
        fig.update_xaxes(title_text="Total Orders", row=1, col=2)
        fig.update_yaxes(title_text="Total Spent ($)", row=1, col=2)
        
        return fig
    
    def create_kpi_cards(self, kpis: Dict) -> str:
        """
        Generate HTML for KPI cards
        
        Args:
            kpis (Dict): Key performance indicators
            
        Returns:
            str: HTML string for KPI cards
        """
        kpi_html = """
        <div style="display: flex; justify-content: space-around; margin: 20px 0;">
        """
        
        kpi_items = [
            ("Total Revenue", f"${kpis['total_revenue']:,.2f}", self.colors['primary']),
            ("Total Transactions", f"{kpis['total_transactions']:,}", self.colors['secondary']),
            ("Unique Customers", f"{kpis['unique_customers']:,}", self.colors['success']),
            ("Avg Order Value", f"${kpis['avg_order_value']:,.2f}", self.colors['info']),
            ("Avg Customer Value", f"${kpis['avg_customer_value']:,.2f}", self.colors['warning'])
        ]
        
        for title, value, color in kpi_items:
            kpi_html += f"""
            <div style="
                background-color: white;
                border-left: 4px solid {color};
                padding: 20px;
                margin: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
                min-width: 150px;
            ">
                <h3 style="margin: 0; color: {color}; font-size: 24px;">{value}</h3>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{title}</p>
            </div>
            """
        
        kpi_html += "</div>"
        return kpi_html
    
    def save_plots_to_files(self, plots: Dict, output_dir: str = "plots/") -> None:
        """
        Save matplotlib plots to files
        
        Args:
            plots (Dict): Dictionary of plot names and figure objects
            output_dir (str): Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for plot_name, fig in plots.items():
            filepath = os.path.join(output_dir, f"{plot_name}.png")
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Plot saved: {filepath}")
        
        plt.close('all')  # Close all figures to free memory