import pandas as pd
import numpy as np
import glob
import os

analysis_dir = "analysis"
feature = 'perm_entropy'  # or 'delta', 'theta', etc.

preictal_files = glob.glob(os.path.join(analysis_dir, f'*_preictal_{feature}.csv'))
interictal_files = glob.glob(os.path.join(analysis_dir, f'*_interictal_{feature}.csv'))

pre_vals, inter_vals = [], []
for f in preictal_files:
    df = pd.read_csv(f, index_col=0)
    if feature in df.columns:
        pre_vals.extend(df[feature].dropna().values)

for f in interictal_files:
    df = pd.read_csv(f, index_col=0)
    if feature in df.columns:
        inter_vals.extend(df[feature].dropna().values)

pre_mean = np.mean(pre_vals)
inter_mean = np.mean(inter_vals)

# threshold halfway between means:
threshold = (pre_mean + inter_mean) / 2

print(f"Preictal mean: {pre_mean:.3f}, Interictal mean: {inter_mean:.3f}")
print(f"Suggested threshold for '{feature}': {threshold:.3f}")

# Optional: use this threshold in your rule-based detection!
