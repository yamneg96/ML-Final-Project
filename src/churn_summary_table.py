"""
File: churn_summary_table.py
Purpose: Save a summary table of churn predictions (counts and mean probability) by churn class.
Output: results/tables/churn_summary_table.csv
"""

import os
import pandas as pd

# ---------------- Paths ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV = os.path.join(BASE_DIR, "results", "new_customer_predictions.csv")
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

# ---------------- Load data ----------------
df = pd.read_csv(INPUT_CSV)

# ---------------- Summary table ----------------
summary = df.groupby("Churn_Prediction").agg(
    Count=("Churn_Prediction", "count"),
    Mean_Probability=("Churn_Probability", "mean")
).reset_index()

# Save table
summary_path = os.path.join(TABLES_DIR, "churn_summary_table.csv")
summary.to_csv(summary_path, index=False)

print(f"[OK] Churn summary table saved to {summary_path}")
print(f"   Total customers: {len(df)}")
print(f"   Summary:")
for _, row in summary.iterrows():
    print(f"     Churn {int(row['Churn_Prediction'])}: {int(row['Count'])} customers (mean prob: {row['Mean_Probability']:.3f})")
