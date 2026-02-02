"""
File: visualize_churn_probability.py
Purpose: Show histogram of churn probabilities predicted by the model.
Output: results/figures/churn_probability_hist.png
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Paths ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV = os.path.join(BASE_DIR, "results", "new_customer_predictions.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------- Load data ----------------
df = pd.read_csv(INPUT_CSV)

# Check if Churn_Probability column exists
if "Churn_Probability" not in df.columns:
    raise ValueError("Churn_Probability column not found in input CSV")

# Warn if dataset is too small
if len(df) < 10:
    print(f"[WARNING] Only {len(df)} samples in dataset. Histogram may not be smooth.")

# Get probability statistics
probabilities = df["Churn_Probability"]
mean_prob = probabilities.mean()
median_prob = probabilities.median()
std_prob = probabilities.std()

# Determine optimal number of bins
n_samples = len(df)
n_bins = min(30, max(10, int(np.sqrt(n_samples))))

# ---------------- Plot ----------------
plt.figure(figsize=(8, 5))
ax = sns.histplot(data=df, x="Churn_Probability", bins=n_bins, kde=True, 
                  color='#e67e22', alpha=0.7, edgecolor='black', linewidth=0.5)

# Add vertical lines for mean and median
plt.axvline(mean_prob, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_prob:.3f}')
plt.axvline(median_prob, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_prob:.3f}')
plt.axvline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Threshold: 0.5', alpha=0.7)

plt.title(f"Predicted Churn Probability Distribution\n(n={n_samples}, μ={mean_prob:.3f}, σ={std_prob:.3f})", 
          fontsize=12, fontweight='bold')
plt.xlabel("Churn Probability", fontsize=11)
plt.ylabel("Number of Customers", fontsize=11)
plt.legend(loc='upper right', fontsize=9)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "churn_probability_hist.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Churn probability histogram saved to {FIGURES_DIR}/churn_probability_hist.png")
print(f"   Mean probability: {mean_prob:.3f}")
print(f"   Median probability: {median_prob:.3f}")
print(f"   Std deviation: {std_prob:.3f}")
print(f"   Range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")