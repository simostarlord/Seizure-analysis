import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

def mad_filter(arr, thresh=2):  # LOWER THRESH = REMOVE MORE OUTLIERS
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return arr[np.abs(arr - med) < thresh * mad]

analysis_dir = "analysis"
bands = ['delta', 'theta', 'alpha', 'beta']

preictal_files = glob.glob(os.path.join(analysis_dir, '*_preictal_bandpower.csv'))
interictal_files = glob.glob(os.path.join(analysis_dir, '*_interictal_bandpower.csv'))

for band in bands:
    pre_vals = []
    inter_vals = []
    for f in preictal_files:
        df = pd.read_csv(f, index_col=0)
        if band in df.columns:
            pre_vals.extend(df[band].dropna().values)
    for f in interictal_files:
        df = pd.read_csv(f, index_col=0)
        if band in df.columns:
            inter_vals.extend(df[band].dropna().values)
    # Convert to numpy arrays and log-transform
    pre_vals = np.log1p(np.array(pre_vals))
    inter_vals = np.log1p(np.array(inter_vals))
    # STRONG percentile filtering (10th–90th)
    pre_lo, pre_hi = np.percentile(pre_vals, [10, 90])
    inter_lo, inter_hi = np.percentile(inter_vals, [10, 90])
    pre_vals = pre_vals[(pre_vals > pre_lo) & (pre_vals < pre_hi)]
    inter_vals = inter_vals[(inter_vals > inter_lo) & (inter_vals < inter_hi)]
    # Tighter MAD filter
    pre_vals = mad_filter(pre_vals)
    inter_vals = mad_filter(inter_vals)
    if len(pre_vals) > 0 and len(inter_vals) > 0:
        stat, pval = ttest_ind(pre_vals, inter_vals, equal_var=False)
        print(f"{band} | Log-Mean preictal: {np.mean(pre_vals):.3f}, Log-Mean interictal: {np.mean(inter_vals):.3f}, t={stat:.3f}, p={pval:.4f}")
        plt.figure(figsize=(6,4))
        plt.boxplot([pre_vals, inter_vals], labels=['Preictal', 'Interictal'])
        plt.title(f"{band.capitalize()} Bandpower Comparison (Filtered & Log)")
        plt.ylabel("Bandpower (log-transformed)")
        plt.show()
    else:
        print(f"{band} | Insufficient data for comparison.")
