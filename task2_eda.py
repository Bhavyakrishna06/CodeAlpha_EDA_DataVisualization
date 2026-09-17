"""CodeAlpha Internship - Task 2: Exploratory Data Analysis (EDA)

Script: task2_eda.py Description: Explores data structure, investigates
anomalies, conducts summary
             statistics, and validates hypotheses on e-commerce transaction data.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


def load_or_create_data(filepath="data/ecommerce_sales_data.csv"):
  """Loads existing dataset or initializes sample data if not present."""
  os.makedirs("data", exist_ok=True)
  if os.path.exists(filepath):
    print(f"[+] Found existing dataset at: {filepath}")
    return pd.read_csv(filepath)

  print("[!] Dataset not found. Generating sample data for analysis...")
  np.random.seed(42)
  n_samples = 500

  categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Beauty"]
  payment_methods = [
      "Credit Card",
      "Debit Card",
      "UPI / Net Banking",
      "Cash on Delivery",
  ]
  genders = ["Male", "Female", "Other"]
  dates = pd.date_range(start="2025-01-01", periods=180, freq="D")

  data = {
      "Transaction_ID": [f"TXN_{1000 + i}" for i in range(n_samples)],
      "Date": np.random.choice(dates, size=n_samples),
      "Customer_Age": np.random.randint(18, 65, size=n_samples).astype(float),
      "Gender": np.random.choice(genders, size=n_samples, p=[0.48, 0.48, 0.04]),
      "Product_Category": np.random.choice(
          categories, size=n_samples, p=[0.25, 0.30, 0.20, 0.15, 0.10]
      ),
      "Quantity": np.random.randint(1, 6, size=n_samples),
      "Unit_Price": np.round(np.random.uniform(15.0, 500.0, size=n_samples), 2),
      "Payment_Method": np.random.choice(
          payment_methods, size=n_samples, p=[0.35, 0.25, 0.30, 0.10]
      ),
      "Customer_Rating": np.random.choice(
          [1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.10, 0.20, 0.40, 0.25]
      ).astype(float),
  }

  df = pd.DataFrame(data)
  df["Total_Amount"] = np.round(df["Quantity"] * df["Unit_Price"], 2)

  # Inject sample missing entries and an outlier
  df.loc[np.random.choice(df.index, 8, replace=False), "Customer_Age"] = np.nan
  df.loc[np.random.choice(df.index, 5, replace=False), "Customer_Rating"] = np.nan
  df.loc[0, "Total_Amount"] = 4850.00

  df.to_csv(filepath, index=False)
  print(f"[+] Dataset saved to {filepath}")
  return df


def run_task2_eda():
  print("=" * 60)
  print("CODEALPHA TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
  print("=" * 60)

  # 1. Load Data
  df = load_or_create_data()

  # 2. Explore Data Structure & Variables
  print("\n--- 1. DATA STRUCTURE & TYPES ---")
  print(f"Dataset Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
  print("\nColumn Information:")
  print(df.dtypes)

  print("\nFirst 5 Records:")
  print(df.head())

  # 3. Detect Data Issues (Missing Values & Duplicates)
  print("\n--- 2. DATA QUALITY AUDIT ---")
  missing_values = df.isnull().sum()
  print("Missing Values per Column:")
  print(missing_values[missing_values > 0])

  duplicate_count = df.duplicated().sum()
  print(f"Duplicate Rows Count: {duplicate_count}")

  # Clean and Impute
  df["Customer_Age"] = df["Customer_Age"].fillna(df["Customer_Age"].median())
  df["Customer_Rating"] = df["Customer_Rating"].fillna(
      df["Customer_Rating"].mode()[0]
  )
  print(
      "[+] Imputed 'Customer_Age' with median and 'Customer_Rating' with mode."
  )

  # 4. Statistical Summary
  print("\n--- 3. SUMMARY STATISTICS ---")
  print(df.describe())

  # 5. Outlier Detection (Interquartile Range - IQR)
  print("\n--- 4. OUTLIER & ANOMALY DETECTION ---")
  q1 = df["Total_Amount"].quantile(0.25)
  q3 = df["Total_Amount"].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr

  outliers = df[
      (df["Total_Amount"] < lower_bound) | (df["Total_Amount"] > upper_bound)
  ]
  print(f"Total Amount IQR: {iqr:.2f}")
  print(f"Upper Outlier Threshold: ${upper_bound:.2f}")
  print(
      f"Identified Outliers Count: {len(outliers)} "
      f"({len(outliers)/len(df)*100:.1f}% of total data)"
  )
  if not outliers.empty:
    print("\nSample Outlier Transactions:")
    print(outliers[["Transaction_ID", "Product_Category", "Total_Amount"]].head())

  # 6. Trends and Patterns Analysis
  print("\n--- 5. CATEGORY & BEHAVIORAL PATTERNS ---")
  category_metrics = (
      df.groupby("Product_Category")
      .agg(
          Total_Revenue=("Total_Amount", "sum"),
          Mean_Spend=("Total_Amount", "mean"),
          Median_Spend=("Total_Amount", "median"),
          Orders_Count=("Transaction_ID", "count"),
      )
      .sort_values(by="Total_Revenue", ascending=False)
  )
  print(category_metrics.round(2))

  # 7. Hypothesis Testing (Statistical Validation)
  print("\n--- 6. HYPOTHESIS TESTING ---")
  # Hypothesis 1: Is there a significant difference in spend across genders?
  male_spend = df[df["Gender"] == "Male"]["Total_Amount"]
  female_spend = df[df["Gender"] == "Female"]["Total_Amount"]
  t_stat, p_val = stats.ttest_ind(male_spend, female_spend, equal_var=False)

  print(
      "Hypothesis 1: Spending differs significantly between Male and Female"
      " customers."
  )
  print(f"  T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}")
  if p_val < 0.05:
    print("  Result: Statistically significant difference observed (p < 0.05).")
  else:
    print(
        "  Result: No statistically significant difference observed (p >= 0.05)."
    )

  # Hypothesis 2: Correlation between Customer Age and Total Spend
  corr, corr_p = stats.pearsonr(df["Customer_Age"], df["Total_Amount"])
  print(
      f"\nHypothesis 2: Correlation between Customer Age and Spend: {corr:.3f}"
      f" (p-value: {corr_p:.4f})"
  )

  # Save cleaned output for downstream visualization
  cleaned_path = "data/ecommerce_sales_cleaned.csv"
  df.to_csv(cleaned_path, index=False)
  print(f"\n[+] Cleaned dataset ready for Task 3 saved to: {cleaned_path}")


if __name__ == "__main__":
  run_task2_eda()
