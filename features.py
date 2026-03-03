# ===============================
# FEATURE ENGINEERING - STAGE 1
# Demand Forecasting (quantity_sold)
# ===============================

import pandas as pd
import numpy as np

# ---------------------------------
# 1️⃣ LOAD DATASET
# ---------------------------------

# 🔴 CHANGE THIS PATH
file_path = "data/fact_consolidated.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully ✅")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)


# ---------------------------------
# 2️⃣ CONVERT POSSIBLE NUMERIC COLUMNS
# ---------------------------------

df_numeric = df.apply(pd.to_numeric, errors='coerce')

numeric_df = df_numeric.select_dtypes(include=['int64', 'float64'])

print("\nNumeric Columns Found:")
print(numeric_df.columns)


# ---------------------------------
# 3️⃣ CHECK IF TARGET EXISTS
# ---------------------------------

if 'quantity_sold' not in numeric_df.columns:
    print("\n❌ ERROR: quantity_sold is not numeric or missing.")
else:
    print("\n✅ quantity_sold found. Proceeding...")


# ---------------------------------
# 4️⃣ CORRELATION WITH TARGET
# ---------------------------------

correlation = numeric_df.corr()['quantity_sold'].sort_values(ascending=False)

print("\n📊 Correlation with quantity_sold:")
print(correlation)


# ---------------------------------
# 5️⃣ SAVE CORRELATION RESULT
# ---------------------------------

correlation.to_csv("correlation_results.csv")

print("\nCorrelation results saved as correlation_results.csv ✅")
