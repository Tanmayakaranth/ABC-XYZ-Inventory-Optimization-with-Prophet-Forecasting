# ABC-XYZ Inventory Optimization with Prophet Forecasting

A **production-ready inventory optimization system** that combines **ABC-XYZ analysis** with **AI-powered demand forecasting (Prophet)** to improve stock management, reduce risk, and optimize ordering strategies.

---

## Overview

This project implements a **data-driven inventory classification and forecasting pipeline** that:

- Classifies items using **ABC analysis** (value-based prioritization)
- Classifies demand variability using **XYZ analysis**
- Combines both into a **9-category decision matrix**
- Uses **Facebook Prophet** to forecast demand for priority items
- Computes **inventory optimization metrics** like:
  - Safety Stock
  - Reorder Point
  - EOQ (Economic Order Quantity)

---

## Key Features

### 1. ABC Classification
- Based on **annual consumption value**
- Uses Pareto principle:
  - A → High value
  - B → Medium value
  - C → Low value

### 2. XYZ Classification
- Based on **demand variability (Coefficient of Variation)**
  - X → Stable demand
  - Y → Moderate variability
  - Z → Highly uncertain demand

### 3. Combined ABC-XYZ Matrix
- 9 categories (AX, AY, ..., CZ)
- Each mapped to:
  - Service level
  - Priority
  - Review frequency

### 4. Demand Forecasting (AI)
- Uses **Prophet time-series model**
- Forecasts future demand for **priority items**
- Handles:
  - Seasonality
  - Trend changes
  - Uncertainty intervals

### 5. Inventory Optimization
For each item:
- Safety Stock calculation
- Reorder Point
- EOQ
- Lead-time aware adjustments

### 6. Full Inventory Table
- Generates optimized metrics for **all 9 categories**
- Works even without forecasting (fallback logic)

### 7. Visualization
- ABC distribution
- XYZ distribution
- Heatmap matrix
- Value contribution pie chart
- Forecast plots

---

## 🛠️ Tech Stack

- Core: Python 3.10+
- Data Processing:
  - pandas
  - numpy
- Machine Learning / Forecasting: prophet (Facebook Prophet)
- Visualization:
  - matplotlib
  - seaborn
- Statistical Modeling: scipy
---



## Input Dataset Requirements

Any csv can be used as input provided the csv contains:

| Column Name            | Description |
|-----------------------|------------|
| Item_ID               | Unique identifier |
| Total_Annual_Units    | Total yearly demand |
| Price_Per_Unit        | Cost per unit |
| *_Demand              | Monthly demand columns (12 months) |

---

## How It Works: Pipeline

1. Load dataset  
2. Perform **ABC classification**  
3. Perform **XYZ classification**  
4. Create **ABC-XYZ matrix**  
5. Forecast demand for priority items  
6. Compute inventory metrics  
7. Generate full inventory table  
8. Visualize results  
9. Export final report  

---

## Usage

```python
from abc_xyz_prophet_implementation import ABCXYZAnalyzer

analyzer = ABCXYZAnalyzer(
    file_path='abc_xyz_dataset.csv',
    base_year=2024
)

final_report = analyzer.run_complete_analysis(
    forecast_periods=6,
    max_forecast_items=15
)
