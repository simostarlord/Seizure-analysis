#Reads all processed bandpower and entropy results for preictal segments

import pandas as pd
import glob
import os

analysis_dir = "analysis"
band = 'delta'         # or 'theta', 'alpha', 'beta'
bandpower_threshold = 1e-10    # Use your calculated value
entropy_threshold = 0.593      # From your previous analysis

preictal_bandpower_files = glob.glob(os.path.join(analysis_dir, '*_preictal_bandpower.csv'))
preictal_entropy_files = glob.glob(os.path.join(analysis_dir, '*_preictal_perm_entropy.csv'))

results = []

for bp_file in preictal_bandpower_files:
    # Find matching entropy file (change pattern to match naming if needed)
    ent_file = bp_file.replace('bandpower.csv', 'perm_entropy.csv')
    if not os.path.exists(ent_file):
        print(f"Missing entropy file for {bp_file}")
        continue
    bp_df = pd.read_csv(bp_file, index_col=0)
    ent_df = pd.read_csv(ent_file, index_col=0)
    for ch in bp_df.index:
        if ch in ent_df.index and band in bp_df.columns:
            bandpower_val = bp_df.loc[ch][band]
            entropy_val = ent_df.loc[ch]['perm_entropy']
            # Flag as preictal/seizure-prone by either rule (customize as needed)
            if bandpower_val < bandpower_threshold and entropy_val < entropy_threshold:
                results.append((bp_file, ch, bandpower_val, entropy_val))
                print(f"Preictal flagged in {bp_file}, channel: {ch}, bandpower: {bandpower_val:.2e}, entropy: {entropy_val:.3f}")

# Optional: Save flagged segments to a CSV for further analysis
flag_df = pd.DataFrame(results, columns=['file', 'channel', 'bandpower', 'entropy'])
flag_df.to_csv('preictal_flagged_segments.csv', index=False)
