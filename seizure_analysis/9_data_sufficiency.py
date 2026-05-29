"""
9_data_sufficiency.py
=====================
Visualises the data scarcity problem in the Siena Scalp EEG dataset.

Produces two plots saved to ml_results/:
  1. segments_per_patient.png  — bar chart of how many segments each patient has
  2. segments_vs_auc.png       — scatter of segments available vs best AUC achieved
                                 (subject-dependent), with a trend line

Run AFTER 8_subject_dependent.py (needs ml_results/sd_per_patient_results.csv).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "analysis")
SD_RESULTS   = os.path.join(SCRIPT_DIR, "ml_results", "sd_per_patient_results.csv")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "ml_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── COUNT SEGMENTS PER PATIENT ──────────────────────────────────────────────

def count_segments(analysis_dir):
    """Count preictal+interictal segment pairs per patient."""
    bp_files = glob.glob(os.path.join(analysis_dir, '*_preictal_bandpower.csv'))
    counts = {}
    for f in bp_files:
        pid = os.path.basename(f).split('_')[0]
        counts[pid] = counts.get(pid, 0) + 1  # each file = 1 preictal+interictal pair
    # Total segments = pairs × 2
    return {pid: n * 2 for pid, n in sorted(counts.items())}

# ─── PLOT 1: SEGMENTS PER PATIENT ────────────────────────────────────────────

def plot_segments_per_patient(seg_counts, output_dir):
    patients = list(seg_counts.keys())
    counts   = list(seg_counts.values())

    # Colour bars by whether they meet a "useful" threshold (≥8 segments)
    colors = ['#2ecc71' if c >= 8 else '#e74c3c' for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(patients, counts, color=colors, edgecolor='white', linewidth=0.8)

    # Threshold line
    ax.axhline(8, color='black', linestyle='--', linewidth=1.2,
               label='Minimum useful threshold (8 segments)', zorder=3)

    # Annotate bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel("Patient ID", fontsize=12)
    ax.set_ylabel("Total Labelled Segments\n(preictal + interictal)", fontsize=12)
    ax.set_title("Data Scarcity in Siena Scalp EEG\nSegments Available per Patient",
                 fontsize=13)
    ax.legend(fontsize=10)

    # Custom legend for colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='≥8 segments (sufficient)'),
        Patch(facecolor='#e74c3c', label='<8 segments (scarce)'),
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.2,
                   label='Minimum useful threshold')
    ], fontsize=9, loc='upper right')

    ax.set_ylim(0, max(counts) + 3)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(output_dir, 'segments_per_patient.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📊 Saved → {out}")

# ─── PLOT 2: SEGMENTS vs AUC SCATTER ─────────────────────────────────────────

def plot_segments_vs_auc(seg_counts, sd_results_path, output_dir):
    if not os.path.exists(sd_results_path):
        print(f"  [SKIP] {sd_results_path} not found — run script 8 first.")
        return

    sd_df = pd.read_csv(sd_results_path)

    # Best AUC per patient across all models
    best_auc = sd_df.groupby('patient')['auc'].max().reset_index()
    best_auc.columns = ['patient', 'best_auc']

    # Merge with segment counts
    seg_df = pd.DataFrame(list(seg_counts.items()), columns=['patient', 'n_segments'])
    merged = pd.merge(best_auc, seg_df, on='patient')

    if merged.empty:
        print("  [SKIP] No overlapping patients between SD results and segment counts.")
        return

    x = merged['n_segments'].values
    y = merged['best_auc'].values

    # Pearson correlation
    if len(x) > 2:
        r, pval = pearsonr(x, y)
        corr_label = f"Pearson r = {r:.2f}, p = {pval:.3f}"
    else:
        r, pval, corr_label = None, None, "Too few points for correlation"

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter points coloured by AUC
    scatter = ax.scatter(x, y, c=y, cmap='RdYlGn', s=120,
                         edgecolors='grey', linewidth=0.7,
                         vmin=0.3, vmax=1.0, zorder=3)
    plt.colorbar(scatter, ax=ax, label='Best AUC')

    # Annotate each point with patient ID
    for _, row in merged.iterrows():
        ax.annotate(row['patient'],
                    (row['n_segments'], row['best_auc']),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color='#333')

    # Trend line
    if r is not None and len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=1.2,
                label=corr_label, alpha=0.7)

    # Chance line
    ax.axhline(0.5, color='red', linestyle=':', linewidth=1,
               label='Chance (AUC = 0.5)', alpha=0.6)

    ax.set_xlabel("Number of Labelled Segments per Patient", fontsize=12)
    ax.set_ylabel("Best AUC Achieved\n(Subject-Dependent, any model)", fontsize=12)
    ax.set_title("Data Scarcity vs Classification Performance\nSiena Scalp EEG — Subject-Dependent",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    out = os.path.join(output_dir, 'segments_vs_auc.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📊 Saved → {out}")

    # Print summary
    print(f"\n   Correlation: {corr_label}")
    print(f"   Patients with ≥8 segments: "
          f"{(merged['n_segments'] >= 8).sum()} / {len(merged)}")
    print(f"   Median segments per patient: {np.median(x):.0f}")

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📊 Data Sufficiency Analysis — Siena Scalp EEG")
    print("=" * 50)

    seg_counts = count_segments(ANALYSIS_DIR)
    print(f"\nSegments per patient:")
    for pid, n in seg_counts.items():
        flag = "✅" if n >= 8 else "⚠️ "
        print(f"  {flag} {pid}: {n} segments")

    print(f"\nTotal patients : {len(seg_counts)}")
    print(f"Total segments : {sum(seg_counts.values())}")
    print(f"Median per patient: {np.median(list(seg_counts.values())):.0f}")

    print("\nGenerating plots...")
    plot_segments_per_patient(seg_counts, OUTPUT_DIR)
    plot_segments_vs_auc(seg_counts, SD_RESULTS, OUTPUT_DIR)

    print("\n✅ Done! Outputs in ml_results/:")
    print("   - segments_per_patient.png  (bar chart, red = data scarce)")
    print("   - segments_vs_auc.png       (scatter: more data → better AUC?)")