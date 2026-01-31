"""
File: visualize_feature_correlation.py
Purpose: Show correlation heatmap of numerical features vs churn.
Output: results/figures/feature_correlation_heatmap.png
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV = os.path.join(BASE_DIR, "results", "new_customer_predictions.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------- Load data ----------------
df = pd.read_csv(INPUT_CSV)

# Select only numerical columns
numerical_cols = df.select_dtypes(include='number').columns
corr = df[numerical_cols].corr()

# ---------------- Plot ----------------
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_correlation_heatmap.png"))
plt.close()

print(f"✅ Feature correlation heatmap saved to {FIGURES_DIR}/feature_correlation_heatmap.png")
