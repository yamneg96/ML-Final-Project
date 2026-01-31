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

# ---------------- Plot ----------------
plt.figure(figsize=(5,4))
sns.countplot(x="Churn_Prediction", data=df, palette="Set2")
plt.title("Predicted Churn Distribution")
plt.xlabel("Churn Class (0=No, 1=Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "churn_class_distribution.png"))
plt.close()

print(f"✅ Churn class distribution saved to {FIGURES_DIR}/churn_class_distribution.png")
