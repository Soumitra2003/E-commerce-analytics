
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
