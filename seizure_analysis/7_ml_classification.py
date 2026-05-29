"""
7_ml_classification.py
=======================
Preictal vs Interictal classification using bandpower + entropy features.
Dataset: Siena Scalp EEG (PhysioNet)

Pipeline:
  - Loads all preictal/interictal bandpower + entropy CSVs from /analysis
  - Builds a 95-feature vector per segment (19ch × 4 bands + 19ch entropy)
  - Runs Leave-One-Subject-Out (LOSO) cross-validation
  - Compares: Random Forest, SVM (RBF), XGBoost
  - Handles class imbalance with SMOTE (applied only on training folds)
  - Reports: Accuracy, Sensitivity, Specificity, AUC, F1
  - Saves: results table CSV + ROC curve plot
"""

import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

from collections import defaultdict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    f1_score, RocCurveDisplay
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "analysis")
BANDS        = ['delta', 'theta', 'alpha', 'beta']
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "ml_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── STEP 1: LOAD & BUILD FEATURE MATRIX ─────────────────────────────────────

def extract_patient_id(filename):
    """Extract patient ID from filename like 'PN00_seizure1_preictal_bandpower.csv'"""
    base = os.path.basename(filename)
    # Patient ID is everything before the first underscore
    return base.split('_')[0]

def load_features(analysis_dir):
    """
    For each seizure segment (preictal + interictal), build one feature row.
    Feature vector: [bp_ch1_delta, ..., bp_ch19_beta, entropy_ch1, ..., entropy_ch19]
    Returns: X (n_samples, 95), y (n_samples,), patients (n_samples,)
    """
    preictal_bp_files = glob.glob(os.path.join(analysis_dir, '*_preictal_bandpower.csv'))

    X_rows, y_rows, patient_rows = [], [], []

    for bp_file in preictal_bp_files:
        patient_id = extract_patient_id(bp_file)

        # Corresponding files
        inter_bp_file  = bp_file.replace('preictal', 'interictal')
        pre_ent_file   = bp_file.replace('bandpower', 'perm_entropy')
        inter_ent_file = inter_bp_file.replace('bandpower', 'perm_entropy')

        # Skip if any file is missing
        missing = [f for f in [inter_bp_file, pre_ent_file, inter_ent_file]
                   if not os.path.exists(f)]
        if missing:
            print(f"  [SKIP] Missing files for {bp_file}: {missing}")
            continue

        try:
            pre_bp   = pd.read_csv(bp_file, index_col=0)
            inter_bp = pd.read_csv(inter_bp_file, index_col=0)
            pre_ent  = pd.read_csv(pre_ent_file, index_col=0)
            inter_ent= pd.read_csv(inter_ent_file, index_col=0)
        except Exception as e:
            print(f"  [ERROR] Could not load {bp_file}: {e}")
            continue

        def build_feature_vector(bp_df, ent_df):
            """Flatten bandpower across channels + append entropy values."""
            bp_feats = []
            for band in BANDS:
                if band in bp_df.columns:
                    bp_feats.extend(bp_df[band].values.tolist())
                else:
                    bp_feats.extend([np.nan] * len(bp_df))

            if 'perm_entropy' in ent_df.columns:
                ent_feats = ent_df['perm_entropy'].values.tolist()
            else:
                ent_feats = [np.nan] * len(ent_df)

            return bp_feats + ent_feats

        pre_vec   = build_feature_vector(pre_bp, pre_ent)
        inter_vec = build_feature_vector(inter_bp, inter_ent)

        X_rows.append(pre_vec)
        y_rows.append(1)  # preictal = positive class
        patient_rows.append(patient_id)

        X_rows.append(inter_vec)
        y_rows.append(0)  # interictal = negative class
        patient_rows.append(patient_id)

    # ── Pad all rows to the same length (different patients may have
    #    different channel counts, so vectors won't all be equal length) ──
    max_len = max(len(r) for r in X_rows)
    print(f"  Feature vector lengths: min={min(len(r) for r in X_rows)}, "
          f"max={max_len}  (padding shorter rows with NaN)")
    X_rows_padded = [r + [np.nan] * (max_len - len(r)) for r in X_rows]

    X = np.array(X_rows_padded, dtype=float)
    y = np.array(y_rows, dtype=int)
    patients = np.array(patient_rows)

    # Drop columns that are entirely NaN
    valid_cols = ~np.all(np.isnan(X), axis=0)
    X = X[:, valid_cols]

    # Impute remaining NaNs with column median
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    print(f"\n✅ Feature matrix built: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Patients: {sorted(set(patients))}")
    print(f"   Class balance — Preictal: {y.sum()}, Interictal: {(y==0).sum()}\n")
    return X, y, patients


# ─── STEP 2: MODELS ──────────────────────────────────────────────────────────

def get_models():
    """
    Returns dict of model name → imbalanced-learn Pipeline.
    SMOTE is applied only to training folds (inside the pipeline).
    StandardScaler is included for SVM; RF and XGB don't strictly need it
    but it doesn't hurt and keeps the pipeline uniform.
    """
    rf = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=min(3, 1))),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    svm = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=min(3, 1))),
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        ))
    ])

    xgboost = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=min(3, 1))),
        ('scaler', StandardScaler()),
        ('clf', xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        ))
    ])

    return {
        'Random Forest': rf,
        'SVM (RBF)': svm,
        'XGBoost': xgboost
    }


# ─── STEP 3: LOSO CROSS-VALIDATION ───────────────────────────────────────────

def loso_evaluate(X, y, patients, models):
    """
    Leave-One-Subject-Out CV.
    For each patient, train on all others, test on the left-out patient.
    SMOTE is applied inside the pipeline (only on train fold).
    """
    unique_patients = sorted(set(patients))
    n_patients = len(unique_patients)

    if n_patients < 2:
        raise ValueError(f"Need ≥2 patients for LOSO. Found: {unique_patients}")

    print(f"Running LOSO-CV over {n_patients} patients...\n")

    # Store fold-level results per model
    fold_results = defaultdict(list)  # model_name → list of fold metric dicts
    all_probs    = defaultdict(list)  # model_name → list of (y_true, y_prob)

    for i, test_patient in enumerate(unique_patients):
        train_mask = patients != test_patient
        test_mask  = patients == test_patient

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # Skip if test fold has only one class (can't compute AUC)
        if len(set(y_test)) < 2:
            print(f"  Fold {i+1}/{n_patients} [{test_patient}] — skipped (only one class in test)")
            continue

        # Adjust SMOTE k_neighbors based on minority class size in training
        min_class_count = min(np.bincount(y_train))
        k = max(1, min(5, min_class_count - 1))

        print(f"  Fold {i+1}/{n_patients} — Test: {test_patient} | "
              f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        for model_name, pipeline in models.items():
            # Dynamically set SMOTE k_neighbors
            pipeline.set_params(smote__k_neighbors=k)

            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1]

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                auc = roc_auc_score(y_test, y_prob)
                acc = accuracy_score(y_test, y_pred)
                f1  = f1_score(y_test, y_pred, zero_division=0)

                fold_results[model_name].append({
                    'patient': test_patient,
                    'accuracy': acc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'auc': auc,
                    'f1': f1
                })
                all_probs[model_name].append((y_test, y_prob))

            except Exception as e:
                print(f"    [{model_name}] Error in fold {test_patient}: {e}")

    return fold_results, all_probs


# ─── STEP 4: SUMMARISE RESULTS ───────────────────────────────────────────────

def summarise_results(fold_results):
    """Print and return a summary DataFrame across all folds."""
    summary_rows = []
    print("\n" + "="*65)
    print(f"{'MODEL':<18} {'ACC':>6} {'SENS':>6} {'SPEC':>6} {'AUC':>6} {'F1':>6}")
    print("="*65)

    for model_name, folds in fold_results.items():
        if not folds:
            print(f"{model_name:<18}  No valid folds.")
            continue
        df = pd.DataFrame(folds)
        means = df[['accuracy','sensitivity','specificity','auc','f1']].mean()
        stds  = df[['accuracy','sensitivity','specificity','auc','f1']].std()

        print(f"{model_name:<18} "
              f"{means['accuracy']:.3f}±{stds['accuracy']:.2f}  "
              f"{means['sensitivity']:.3f}±{stds['sensitivity']:.2f}  "
              f"{means['specificity']:.3f}±{stds['specificity']:.2f}  "
              f"{means['auc']:.3f}±{stds['auc']:.2f}  "
              f"{means['f1']:.3f}±{stds['f1']:.2f}")

        summary_rows.append({
            'Model': model_name,
            'Accuracy (mean)': round(means['accuracy'], 3),
            'Accuracy (std)':  round(stds['accuracy'], 3),
            'Sensitivity (mean)': round(means['sensitivity'], 3),
            'Sensitivity (std)':  round(stds['sensitivity'], 3),
            'Specificity (mean)': round(means['specificity'], 3),
            'Specificity (std)':  round(stds['specificity'], 3),
            'AUC (mean)': round(means['auc'], 3),
            'AUC (std)':  round(stds['auc'], 3),
            'F1 (mean)': round(means['f1'], 3),
            'F1 (std)':  round(stds['f1'], 3),
        })

    print("="*65 + "\n")
    return pd.DataFrame(summary_rows)


# ─── STEP 5: ROC CURVE PLOT ──────────────────────────────────────────────────

def plot_roc_curves(all_probs, output_dir):
    """
    Aggregate all LOSO test folds and plot one ROC curve per model.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Random Forest': 'forestgreen', 'SVM (RBF)': 'royalblue', 'XGBoost': 'tomato'}

    for model_name, fold_list in all_probs.items():
        if not fold_list:
            continue
        # Concatenate all folds
        y_true_all = np.concatenate([yt for yt, yp in fold_list])
        y_prob_all = np.concatenate([yp for yt, yp in fold_list])
        auc = roc_auc_score(y_true_all, y_prob_all)

        RocCurveDisplay.from_predictions(
            y_true_all, y_prob_all,
            name=f"{model_name} (AUC={auc:.3f})",
            ax=ax,
            color=colors.get(model_name, 'gray')
        )

    ax.plot([0,1],[0,1], 'k--', lw=1, label='Chance')
    ax.set_title("LOSO-CV ROC Curves\nPreictal vs Interictal (Siena Scalp EEG)", fontsize=13)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'roc_curves_loso.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📊 ROC curve saved → {out_path}")


# ─── STEP 6: PER-FOLD BREAKDOWN ──────────────────────────────────────────────

def save_per_fold_results(fold_results, output_dir):
    """Save per-patient fold results to CSV for inspection."""
    all_rows = []
    for model_name, folds in fold_results.items():
        for fold in folds:
            fold['model'] = model_name
            all_rows.append(fold)
    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df[['model','patient','accuracy','sensitivity','specificity','auc','f1']]
        out_path = os.path.join(output_dir, 'per_fold_results.csv')
        df.to_csv(out_path, index=False)
        print(f"📄 Per-fold results saved → {out_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧠 Siena Scalp EEG — Preictal ML Classification")
    print("=" * 55)

    # 1. Load features
    X, y, patients = load_features(ANALYSIS_DIR)

    if len(X) == 0:
        print("❌ No feature data found. Make sure the /analysis folder contains "
              "bandpower and entropy CSVs from scripts 3a and 3b.")
        exit(1)

    # 2. Get models
    models = get_models()

    # 3. LOSO evaluation
    fold_results, all_probs = loso_evaluate(X, y, patients, models)

    # 4. Summary table
    summary_df = summarise_results(fold_results)
    summary_path = os.path.join(OUTPUT_DIR, 'classification_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"📄 Summary saved → {summary_path}")

    # 5. ROC curves
    plot_roc_curves(all_probs, OUTPUT_DIR)

    # 6. Per-fold breakdown
    save_per_fold_results(fold_results, OUTPUT_DIR)

    print("\n✅ Done! Results in /ml_results/")
    print("   - classification_summary.csv  (mean ± std per model)")
    print("   - per_fold_results.csv         (per-patient breakdown)")
    print("   - roc_curves_loso.png          (ROC curves)")