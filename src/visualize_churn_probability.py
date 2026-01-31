"""
File: visualize_churn_probability.py
Purpose: Show histogram of churn probabilities predicted by the model.
Output: results/figures/churn_probability_hist.png
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

# Convert Series to DataFrame to avoid Seaborn type warning
prob_df = df[["Churn_Probability"]]

# ---------------- Plot ----------------
plt.figure(figsize=(6,4))
sns.histplot(data=prob_df, x="Churn_Probability", bins=20, kde=True, color='orange')
plt.title("Predicted Churn Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "churn_probability_hist.png"))
plt.close()

print(f"✅ Churn probability histogram saved to {FIGURES_DIR}/churn_probability_hist.png")
