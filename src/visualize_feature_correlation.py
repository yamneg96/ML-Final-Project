"""
File: visualize_feature_correlation.py
Purpose: Show correlation heatmap of numerical features vs churn.
Output: results/figures/feature_correlation_heatmap.png
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Paths ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Use processed data for correlation (more reliable than predictions)
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "churn_processed.csv")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "results", "new_customer_predictions.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------- Load data ----------------
# Try to use processed data first (better for correlation analysis)
if os.path.exists(PROCESSED_CSV):
    df = pd.read_csv(PROCESSED_CSV)
    print(f"Using processed data from {PROCESSED_CSV}")
else:
    # Fallback to predictions if processed data not available
    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Using predictions data from {PREDICTIONS_CSV}")

# Select only numerical columns (exclude target columns if present)
target_cols = ['Churn Value', 'Churn Label', 'Churn_Prediction', 'Churn_Probability']
numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                  if col not in target_cols]

# Add target if available for correlation analysis
if 'Churn Value' in df.columns:
    numerical_cols.append('Churn Value')
elif 'Churn Label' in df.columns:
    # Convert Churn Label to numeric if needed
    if df['Churn Label'].dtype == 'object':
        df['Churn Label'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    numerical_cols.append('Churn Label')
elif 'Churn_Probability' in df.columns:
    numerical_cols.append('Churn_Probability')

# Calculate correlation
corr = df[numerical_cols].corr()

# ---------------- Plot ----------------
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
            vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Feature Correlation Heatmap\n(Numerical Features)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Feature correlation heatmap saved to {FIGURES_DIR}/feature_correlation_heatmap.png")
print(f"   Analyzed {len(numerical_cols)} numerical features from {len(df)} samples")