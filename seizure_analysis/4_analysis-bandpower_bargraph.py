import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

analysis_dir = "analysis"  # use the folder where you saved your bandpower CSVs
bands = ['delta', 'theta', 'alpha', 'beta']

preictal_files = glob.glob(os.path.join(analysis_dir, '*_preictal_bandpower.csv'))

for pre_file in preictal_files:
    inter_file = pre_file.replace('preictal', 'interictal')
    if not os.path.exists(inter_file):
        print(f"Missing interictal file for {pre_file}, skipping.")
        continue

    # Parse patient and seizure indices for title
    base = os.path.basename(pre_file)
    patient = base.split('_')[0]
    seizure = base.split('_')[2].replace('seizure','')

    pre_df = pd.read_csv(pre_file, index_col=0)
    inter_df = pd.read_csv(inter_file, index_col=0)

    for band in bands:
        plt.figure(figsize=(14, 6))
        x = range(len(pre_df.index))
        plt.bar(x, pre_df[band], width=0.4, label="Preictal", color="red", alpha=0.6)
        plt.bar(x, inter_df[band], width=0.4, label="Interictal", color="blue", alpha=0.4, bottom=0)
        plt.xticks(x, pre_df.index, rotation=90)
        plt.title(f"{patient} Seizure {seizure}: Channel-wise {band} Band Power\nPreictal vs Interictal")
        plt.xlabel("EEG Channel")
        plt.ylabel("Band Power")
        plt.legend()
        plt.tight_layout()
        plt.show()
