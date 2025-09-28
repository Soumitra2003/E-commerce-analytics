# Power BI Dashboard Templates and Setup

## 📊 Files in This Directory

### 1. `powerbi_data_prep.py`
Python script that prepares all data for Power BI import:
- Exports clean CSV files optimized for Power BI
- Creates dimension tables and fact tables
- Generates business metrics and KPIs
- Provides setup instructions

### 2. Power BI Template (Coming Soon)
- Pre-built dashboard template
- Professional visualizations
- Executive-ready reports
- Interactive filters and slicers

## 🚀 Quick Start Guide

### Step 1: Prepare the Data
```bash
# Run from the dashboard directory
cd dashboard
python powerbi_data_prep.py
```

This creates a `../data/powerbi/` folder with all the CSV files needed.

### Step 2: Import Data to Power BI
1. Open Power BI Desktop
2. **Get Data** → **Text/CSV**
3. Import all CSV files from `../data/powerbi/` folder
4. Set proper data types (dates, numbers, text)

### Step 3: Create Data Model
Set up relationships between tables:
- `Transactions[customer_id]` → `Customers[customer_id]`
- `Transactions[transaction_date]` → `Date_Dimension[Date]`
- `Transactions[product_category]` → `Product_Categories[product_category]`



```dax
// Revenue Measures
Total Revenue = SUM(Transactions[total_amount])

Average Order Value = DIVIDE([Total Revenue], COUNT(Transactions[transaction_id]))

Revenue Growth % = 
VAR CurrentPeriod = [Total Revenue]
VAR PreviousPeriod = CALCULATE([Total Revenue], PREVIOUSMONTH(Date_Dimension[Date]))
RETURN DIVIDE(CurrentPeriod - PreviousPeriod, PreviousPeriod)

// Customer Measures
Unique Customers = DISTINCTCOUNT(Transactions[customer_id])

Customer Retention Rate = 
DIVIDE(
    CALCULATE(DISTINCTCOUNT(Transactions[customer_id]), Transactions[total_orders] > 1),
    [Unique Customers]
)

Average Customer Value = DIVIDE([Total Revenue], [Unique Customers])

// Product Measures
Categories Count = DISTINCTCOUNT(Transactions[product_category])

Top Category Revenue = 
CALCULATE([Total Revenue], 
    TOPN(1, VALUES(Transactions[product_category]), [Total Revenue])
)
```

