"""
8_subject_dependent.py
======================
Subject-DEPENDENT classification baseline for Siena Scalp EEG.

Contrast with 7_ml_classification.py (LOSO = subject-independent).

For each patient who has ≥4 segments:
  - Train & test on that patient's data only (stratified 80/20 split)
  - Run all 3 models (RF, SVM, XGBoost)

Then produces a side-by-side comparison plot:
  Subject-Dependent AUC  vs  LOSO (Subject-Independent) AUC
showing exactly why inter-subject variability is the core challenge.

Run AFTER 7_ml_classification.py (needs ml_results/per_fold_results.csv).
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# All paths anchored to THIS script's directory so they work regardless
# of which directory you run the script from.
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR  = os.path.join(SCRIPT_DIR, "analysis")
LOSO_RESULTS  = os.path.join(SCRIPT_DIR, "ml_results", "per_fold_results.csv")
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "ml_results")
BANDS         = ['delta', 'theta', 'alpha', 'beta']
TEST_SIZE     = 0.2   # 80/20 split within each patient
RANDOM_STATE  = 42
MIN_SEGMENTS  = 4     # skip patients with too few segments to split
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── FEATURE LOADING (same logic as script 7) ────────────────────────────────

def extract_patient_id(filename):
    return os.path.basename(filename).split('_')[0]

def build_feature_vector(bp_df, ent_df, bands):
    bp_feats = []
    for band in bands:
        if band in bp_df.columns:
            bp_feats.extend(bp_df[band].values.tolist())
        else:
            bp_feats.extend([np.nan] * len(bp_df))
    if 'perm_entropy' in ent_df.columns:
        ent_feats = ent_df['perm_entropy'].values.tolist()
    else:
        ent_feats = [np.nan] * len(ent_df)
    return bp_feats + ent_feats

def load_features_by_patient(analysis_dir):
    """
    Returns a dict: patient_id → (X, y)
    Each patient's X is already padded to uniform length within that patient.
    """
    preictal_bp_files = glob.glob(os.path.join(analysis_dir, '*_preictal_bandpower.csv'))
    patient_data = defaultdict(lambda: {'X': [], 'y': []})

    for bp_file in preictal_bp_files:
        patient_id    = extract_patient_id(bp_file)
        inter_bp_file = bp_file.replace('preictal', 'interictal')
        pre_ent_file  = bp_file.replace('bandpower', 'perm_entropy')
        inter_ent_file= inter_bp_file.replace('bandpower', 'perm_entropy')

        missing = [f for f in [inter_bp_file, pre_ent_file, inter_ent_file]
                   if not os.path.exists(f)]
        if missing:
            continue

        try:
            pre_bp    = pd.read_csv(bp_file, index_col=0)
            inter_bp  = pd.read_csv(inter_bp_file, index_col=0)
            pre_ent   = pd.read_csv(pre_ent_file, index_col=0)
            inter_ent = pd.read_csv(inter_ent_file, index_col=0)
        except Exception as e:
            print(f"  [ERROR] {bp_file}: {e}")
            continue

        pre_vec   = build_feature_vector(pre_bp,   pre_ent,   BANDS)
        inter_vec = build_feature_vector(inter_bp,  inter_ent, BANDS)

        patient_data[patient_id]['X'].append(pre_vec)
        patient_data[patient_id]['y'].append(1)
        patient_data[patient_id]['X'].append(inter_vec)
        patient_data[patient_id]['y'].append(0)

    # Pad within each patient & impute NaN
    result = {}
    for pid, data in patient_data.items():
        rows = data['X']
        max_len = max(len(r) for r in rows)
        padded  = [r + [np.nan] * (max_len - len(r)) for r in rows]
        X = np.array(padded, dtype=float)
        y = np.array(data['y'], dtype=int)

        # Median imputation
        col_medians = np.nanmedian(X, axis=0)
        nan_mask    = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        # Drop all-NaN columns
        valid_cols = ~np.all(np.isnan(X), axis=0)
        X = X[:, valid_cols]

        result[pid] = (X, y)

    return result

# ─── MODELS (same as script 7) ────────────────────────────────────────────────

def get_models(k_neighbors=1):
    rf = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200, max_depth=6,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    svm = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel='rbf', C=1.0, gamma='scale',
            probability=True, class_weight='balanced', random_state=RANDOM_STATE
        ))
    ])
    xgboost = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    return {'Random Forest': rf, 'SVM (RBF)': svm, 'XGBoost': xgboost}

# ─── SUBJECT-DEPENDENT EVALUATION ────────────────────────────────────────────

def subject_dependent_eval(patient_data):
    """
    For each patient with enough data: stratified 80/20 split, fit, evaluate.
    Returns: DataFrame with columns [patient, model, accuracy, sensitivity,
                                      specificity, auc, f1]
    """
    results = []
    splitter = StratifiedShuffleSplit(
        n_splits=5, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    for patient_id, (X, y) in sorted(patient_data.items()):
        n_samples     = len(y)
        n_preictal    = int(y.sum())
        n_interictal  = int((y == 0).sum())

        if n_samples < MIN_SEGMENTS:
            print(f"  [SKIP] {patient_id}: only {n_samples} segments (need ≥{MIN_SEGMENTS})")
            continue
        if n_preictal < 1 or n_interictal < 1:
            print(f"  [SKIP] {patient_id}: only one class present")
            continue

        # Dynamically set test_size so test set always has ≥2 samples
        # (one per class minimum), needed for stratified split to work
        min_test_size = max(2, int(np.ceil(n_samples * TEST_SIZE)))
        min_test_size = min(min_test_size, n_samples - 2)  # leave ≥2 for train
        actual_test_size = min_test_size / n_samples

        print(f"\n  Patient {patient_id}: {n_samples} segments "
              f"(preictal={n_preictal}, interictal={n_interictal})")

        # k_neighbors for SMOTE — can't exceed minority class size - 1
        k = max(1, min(5, min(n_preictal, n_interictal) - 1))
        models = get_models(k_neighbors=k)

        local_splitter = StratifiedShuffleSplit(
            n_splits=5, test_size=actual_test_size, random_state=RANDOM_STATE
        )

        for model_name, pipeline in models.items():
            fold_aucs, fold_accs, fold_sens, fold_spec, fold_f1s = [], [], [], [], []

            for train_idx, test_idx in local_splitter.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Need both classes in both splits for meaningful eval
                if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                    continue

                try:
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    y_prob = pipeline.predict_proba(X_test)[:, 1]

                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
                    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

                    fold_aucs.append(roc_auc_score(y_test, y_prob))
                    fold_accs.append(accuracy_score(y_test, y_pred))
                    fold_sens.append(sens)
                    fold_spec.append(spec)
                    fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))
                except Exception as e:
                    print(f"    [{model_name}] fold error: {e}")

            if fold_aucs:
                results.append({
                    'patient':     patient_id,
                    'model':       model_name,
                    'accuracy':    np.mean(fold_accs),
                    'sensitivity': np.mean(fold_sens),
                    'specificity': np.mean(fold_spec),
                    'auc':         np.mean(fold_aucs),
                    'f1':          np.mean(fold_f1s),
                })
                print(f"    {model_name:<18} AUC={np.mean(fold_aucs):.3f}  "
                      f"Sens={np.mean(fold_sens):.3f}  Spec={np.mean(fold_spec):.3f}")

    return pd.DataFrame(results)

# ─── COMPARISON PLOT ─────────────────────────────────────────────────────────

def plot_comparison(sd_df, loso_path, output_dir):
    """
    Side-by-side bar chart: Subject-Dependent vs LOSO AUC per model.
    Also draws a horizontal chance line at 0.5.
    """
    if not os.path.exists(loso_path):
        print(f"  [WARN] LOSO results not found at {loso_path}, skipping comparison plot.")
        return

    loso_df = pd.read_csv(loso_path)

    models      = ['Random Forest', 'SVM (RBF)', 'XGBoost']
    colors_sd   = ['#2ecc71', '#3498db', '#e74c3c']
    colors_loso = ['#27ae60', '#2980b9', '#c0392b']

    sd_means, sd_stds, loso_means, loso_stds = [], [], [], []

    for model in models:
        sd_sub   = sd_df[sd_df['model'] == model]['auc']
        loso_sub = loso_df[loso_df['model'] == model]['auc']
        sd_means.append(sd_sub.mean()   if len(sd_sub)   > 0 else np.nan)
        sd_stds.append(sd_sub.std()     if len(sd_sub)   > 1 else 0)
        loso_means.append(loso_sub.mean() if len(loso_sub) > 0 else np.nan)
        loso_stds.append(loso_sub.std()   if len(loso_sub) > 1 else 0)

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(x - width/2, sd_means,   width, yerr=sd_stds,
                   label='Subject-Dependent (80/20 split)',
                   color=colors_sd, alpha=0.85, capsize=5, zorder=3)
    bars2 = ax.bar(x + width/2, loso_means, width, yerr=loso_stds,
                   label='Subject-Independent (LOSO-CV)',
                   color=colors_loso, alpha=0.45, capsize=5, zorder=3)

    # Chance line
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.2,
               label='Chance (AUC=0.5)', zorder=2)

    # Annotate bars with values
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9, color='#555')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Mean AUC (± std)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Subject-Dependent vs Subject-Independent Classification\n"
                 "Preictal vs Interictal | Siena Scalp EEG", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'sd_vs_loso_comparison.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n📊 Comparison plot saved → {out_path}")

# ─── SUMMARY TABLE ───────────────────────────────────────────────────────────

def print_summary(sd_df):
    print("\n" + "="*65)
    print("SUBJECT-DEPENDENT SUMMARY (mean ± std across patients)")
    print(f"{'MODEL':<18} {'ACC':>6} {'SENS':>6} {'SPEC':>6} {'AUC':>6} {'F1':>6}")
    print("="*65)
    summary_rows = []
    for model in ['Random Forest', 'SVM (RBF)', 'XGBoost']:
        sub = sd_df[sd_df['model'] == model]
        if sub.empty:
            continue
        m = sub[['accuracy','sensitivity','specificity','auc','f1']].mean()
        s = sub[['accuracy','sensitivity','specificity','auc','f1']].std().fillna(0)
        print(f"{model:<18} "
              f"{m['accuracy']:.3f}±{s['accuracy']:.2f}  "
              f"{m['sensitivity']:.3f}±{s['sensitivity']:.2f}  "
              f"{m['specificity']:.3f}±{s['specificity']:.2f}  "
              f"{m['auc']:.3f}±{s['auc']:.2f}  "
              f"{m['f1']:.3f}±{s['f1']:.2f}")
        summary_rows.append({
            'Model': model,
            'Accuracy (mean)': round(m['accuracy'], 3),   'Accuracy (std)': round(s['accuracy'], 3),
            'Sensitivity (mean)': round(m['sensitivity'], 3), 'Sensitivity (std)': round(s['sensitivity'], 3),
            'Specificity (mean)': round(m['specificity'], 3), 'Specificity (std)': round(s['specificity'], 3),
            'AUC (mean)': round(m['auc'], 3),              'AUC (std)': round(s['auc'], 3),
            'F1 (mean)': round(m['f1'], 3),                'F1 (std)': round(s['f1'], 3),
        })
    print("="*65)
    return pd.DataFrame(summary_rows)

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧠 Siena Scalp EEG — Subject-Dependent Baseline")
    print("=" * 55)

    # 1. Load per-patient features
    print("\n📂 Loading features by patient...")
    print(f"   Script location : {os.path.abspath(__file__)}")
    print(f"   Working dir     : {os.getcwd()}")
    print(f"   Looking for     : {os.path.abspath(ANALYSIS_DIR)}")
    found_csvs = glob.glob(os.path.join(ANALYSIS_DIR, '*_preictal_bandpower.csv'))
    print(f"   Preictal BP CSVs found: {len(found_csvs)}")
    for f in found_csvs[:5]:
        print(f"     {f}")
    if not found_csvs:
        print("\n❌ No CSVs found. Fix ANALYSIS_DIR at the top of this script.")
        print("   It should point to the folder containing your bandpower/entropy CSVs.")
        print(f"   Current value: ANALYSIS_DIR = '{ANALYSIS_DIR}'")
        exit(1)

    patient_data = load_features_by_patient(ANALYSIS_DIR)
    print(f"   Found {len(patient_data)} patients: {sorted(patient_data.keys())}")

    # 2. Subject-dependent evaluation
    print("\n🔁 Running subject-dependent evaluation (5×80/20 splits)...\n")
    sd_df = subject_dependent_eval(patient_data)

    if sd_df.empty:
        print("❌ No patients had enough segments. Check MIN_SEGMENTS setting.")
        exit(1)

    # 3. Summary
    summary_df = print_summary(sd_df)

    # 4. Save
    sd_df.to_csv(os.path.join(OUTPUT_DIR, 'sd_per_patient_results.csv'), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'sd_summary.csv'), index=False)
    print(f"\n📄 Saved → ml_results/sd_per_patient_results.csv")
    print(f"📄 Saved → ml_results/sd_summary.csv")

    # 5. Comparison plot vs LOSO
    plot_comparison(sd_df, LOSO_RESULTS, OUTPUT_DIR)

    print("\n✅ Done! Key outputs in /ml_results/:")
    print("   - sd_summary.csv              (subject-dependent mean ± std)")
    print("   - sd_per_patient_results.csv  (per-patient breakdown)")
    print("   - sd_vs_loso_comparison.png   (the key comparison figure)")
    print("\n💡 Expected: subject-dependent AUC >> LOSO AUC")
    print("   This gap IS our finding — it motivates personalised models.")