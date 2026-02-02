"""
File: visualize_churn_distribution.py
Purpose: Show distribution of predicted churn classes (0 vs 1).
Output: results/figures/churn_class_distribution.png
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

# Check if Churn_Prediction column exists
if "Churn_Prediction" not in df.columns:
    raise ValueError("Churn_Prediction column not found in input CSV")

# Warn if dataset is too small
if len(df) < 10:
    print(f"[WARNING] Only {len(df)} samples in dataset. Visualization may not be meaningful.")

# Get counts
churn_counts = df["Churn_Prediction"].value_counts().sort_index()
total = len(df)
churn_rate = (churn_counts.get(1, 0) / total * 100) if 1 in churn_counts.index else 0

# ---------------- Plot ----------------
plt.figure(figsize=(7, 5))
ax = sns.countplot(x="Churn_Prediction", data=df, hue="Churn_Prediction", 
                   palette=["#2ecc71", "#e74c3c"], 
                   order=[0, 1] if 1 in df["Churn_Prediction"].values else [0],
                   legend=False)
plt.title(f"Predicted Churn Distribution\n(Total: {total} customers, Churn Rate: {churn_rate:.1f}%)", 
          fontsize=12, fontweight='bold')
plt.xlabel("Churn Class (0=No Churn, 1=Churn)", fontsize=11)
plt.ylabel("Number of Customers", fontsize=11)

# Add count labels on bars
for i, (idx, count) in enumerate(churn_counts.items()):
    ax.text(i, count + max(churn_counts) * 0.01, f'{count}\n({count/total*100:.1f}%)', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "churn_class_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Churn class distribution saved to {FIGURES_DIR}/churn_class_distribution.png")
print(f"   No Churn (0): {churn_counts.get(0, 0)} customers")
print(f"   Churn (1): {churn_counts.get(1, 0)} customers")