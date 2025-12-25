import pandas as pd
import glob
import os
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

analysis_dir = "analysis"
feature = 'perm_entropy'  # or 'delta', 'theta', etc.

preictal_files = glob.glob(os.path.join(analysis_dir, '*_preictal_perm_entropy.csv'))  # set correct file pattern

results = []
for pre_file in preictal_files:
    inter_file = pre_file.replace('preictal', 'interictal')
    if not os.path.exists(inter_file):
        continue
    pre_df = pd.read_csv(pre_file, index_col=0)
    inter_df = pd.read_csv(inter_file, index_col=0)
    # Merge values for paired comparison
    channel_values = []
    for ch in pre_df.index:
        if ch in inter_df.index:
            channel_values.append((pre_df.loc[ch][feature], inter_df.loc[ch][feature]))
    if not channel_values:
        continue
    pre_v, inter_v = zip(*channel_values)
    # Paired t-test
    stat, pval = ttest_rel(pre_v, inter_v)
    print(f"{pre_file}: t={stat:.3f}, p={pval:.4f}")
    # Boxplot visualization
    plt.figure(figsize=(6,4))
    plt.boxplot([pre_v, inter_v], labels=['Preictal', 'Interictal'])
    plt.title(f"Permutation Entropy Comparison\nFile: {os.path.basename(pre_file)}")
    plt.ylabel("Permutation Entropy")
    plt.show()
