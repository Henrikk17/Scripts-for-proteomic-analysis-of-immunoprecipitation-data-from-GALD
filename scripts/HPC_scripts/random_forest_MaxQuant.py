# 12/11-2026
# Deterministic Random Forest Analysis on MaxQuant Data (Runs 4, 5, 6)

import os
import pickle
from pathlib import Path
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from matplotlib.backends.backend_pdf import PdfPages

from helper_functions.load_maxQuant_proteomic_data import (
    load_maxQuant_proteomic_data,
    find_maxquant_txt_dir
)

# Directories for cache and plots
CACHE_DIR = Path("../cache_maxquant")
PLOT_DIR = Path("../plots/random_forest")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Reference Top 20 proteins from FragPipe workflows for direct cross-engine comparison
FRAGPIPE_3_OF_3 = {'C4BPA', 'FKBP11', 'PKM', 'PROS1', 'TAGLN2', 'TFRC', 'TKFC', 'TLN1', 'UGP2'}
FRAGPIPE_2_OF_3 = {'ACTR3', 'CP', 'DHRS7', 'MTTP', 'RPL21', 'RSU1'}


def get_maxquant_datasets(use_cache=True, base_dir="~/DTU-GALD"):
    """
    Loads and filters MaxQuant Run 4 (R0876), Run 5 (R1101), and Run 6 (R1344 - No LB),
    applying identical sample exclusions (removing KON QC sample_14/sample_56 and Buffer controls).
    """
    cache_file = CACHE_DIR / "maxquant_filtered_prot.pkl"
    if use_cache and cache_file.exists():
        print(f"⚡ [CACHE] Loading MaxQuant datasets from {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Configuration matching your experiment_annotation structure
    run_specs = {
        "mq_run4": {
            "req_id": "0876",
            "annot": "../experiment_annotation/experiment_annotation_run4_R876.tsv",
            "label": "MaxQuant | Run 4"
        },
        "mq_run5": {
            "req_id": "1101",
            "annot": "../experiment_annotation/experiment_annotation_run5_R1101.tsv",
            "label": "MaxQuant | Run 5"
        },
        "mq_run6_no_LB": {
            "req_id": "1344",
            "annot": "../experiment_annotation/experiment_annotation_run6_R1344.tsv",
            "label": "MaxQuant | Run 6 (No LB)"
        }
    }

    mq_datasets = {}
    for key, spec in run_specs.items():
        txt_dir = find_maxquant_txt_dir(spec["req_id"], base_dir=base_dir)
        print(f"\n[*] Located {spec['label']} at: {txt_dir}")

        prot_df, _ = load_maxQuant_proteomic_data(
            ROOT=txt_dir,
            annotation_file=spec["annot"],
            load_peptides=False
        )

        # Standard quality/cohort filtering identical to FragPipe manage_datasets.py:
        # 1. Remove GALD_14 (KON internal repeat control: sample_14 and sample_56)
        prot_df = prot_df[~prot_df['Sample'].isin(['sample_14', 'sample_56'])].copy()

        # 2. Exclude Lysate Buffer controls (keep only Liver Lysate IP samples)
        if 'Sample_type' in prot_df.columns:
            prot_df = prot_df[prot_df['Sample_type'].astype(str).str.strip() == 'Liver Lysate'].copy()

        mq_datasets[key] = {
            "label": spec["label"],
            "prot": prot_df
        }

    with open(cache_file, "wb") as f:
        pickle.dump(mq_datasets, f)
    print(f"\n✅ Cached filtered MaxQuant datasets to {cache_file}")

    return mq_datasets


def run_maxquant_random_forest(mq_datasets, pdf_obj):
    """
    Merges MaxQuant R4, R5, and R6 (No LB), applies within-run deterministic Perseus
    imputation and Z-scoring, performs a 70/30 stratified Train/Test split, and fits RF.
    """
    print(f"\n{'='*70}")
    print("🌲 RANDOM FOREST ANALYSIS: MAXQUANT (R4 + R5 + R6)")
    print(f"{'='*70}")

    run_configs = [
        ("mq_run4", "Run 4"),
        ("mq_run5", "Run 5"),
        ("mq_run6_no_LB", "Run 6")
    ]

    wide_runs = []
    sample_to_group = {}

    for key, run_label in run_configs:
        df = mq_datasets[key]["prot"].dropna(subset=['Gene']).copy()
        quant_col = 'Razor Intensity'

        # Unique Sample ID to prevent sample_1 collisions across runs
        df['Unique_Sample'] = f"{run_label.replace(' ', '')}_" + df['Sample'].astype(str)

        for _, row in df.drop_duplicates('Unique_Sample').iterrows():
            u_samp = row['Unique_Sample']
            grp = str(row['Group']).strip()
            sample_to_group[u_samp] = 'GALD' if grp == 'GALD' else 'Control'

        # Pivot to [Unique_Sample x Gene]
        pivot_df = df.pivot_table(index='Unique_Sample', columns='Gene', values=quant_col, aggfunc='sum')
        pivot_df.replace(0, np.nan, inplace=True)
        pivot_df = pivot_df.dropna(thresh=3, axis=1)

        log2_df = np.log2(pivot_df)

        # Deterministic Perseus Imputation BEFORE Z-scoring
        np.random.seed(42)
        imputed_run_df = log2_df.copy()
        for samp in imputed_run_df.index:
            row_vals = imputed_run_df.loc[samp].dropna()
            if len(row_vals) > 2:
                mean_val = row_vals.mean()
                std_val = row_vals.std()
                na_mask = imputed_run_df.loc[samp].isna()
                imputed_run_df.loc[samp, na_mask] = np.random.normal(
                    loc=mean_val - (1.8 * std_val),
                    scale=std_val * 0.3,
                    size=na_mask.sum()
                )

        # Within-Run Z-Score
        scaler = StandardScaler()
        z_array = scaler.fit_transform(imputed_run_df)
        z_df = pd.DataFrame(z_array, index=imputed_run_df.index, columns=imputed_run_df.columns)

        wide_runs.append(z_df)

    # Strict Inner Join across all 3 runs
    X = pd.concat(wide_runs, axis=0, join='inner')
    y = np.array([sample_to_group[s] for s in X.index])
    y_binary = (y == 'GALD').astype(int)

    print(f"  [*] Matrix Shape: {X.shape[0]} samples x {X.shape[1]} proteins")

    # 70% Train / 30% Test Split (Stratified, Seed = 42)
    X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = train_test_split(
        X, y, y_binary, test_size=0.3, random_state=42, stratify=y
    )

    print(f"  [*] Training Set: {X_train.shape[0]} samples")
    print(f"  [*] Testing Set:  {X_test.shape[0]} samples")

    # Fit Random Forest
    rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    train_probs = rf.predict_proba(X_train)[:, 1]
    test_probs = rf.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_bin_train, train_probs)
    test_auc = roc_auc_score(y_bin_test, test_probs)

    print(f"\n  📊 Model Performance:")
    print(f"     Train ROC-AUC: {train_auc:.4f}")
    print(f"     Test ROC-AUC:  {test_auc:.4f}")

    # Extract Top 20 Features
    feat_imp_df = pd.DataFrame({
        'Gene': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(20).reset_index(drop=True)

    print(f"\n  🏆 Top 20 Predictive Proteins (MAXQUANT):")
    for idx, row in feat_imp_df.iterrows():
        gene = row['Gene']
        tag = ""
        if gene in FRAGPIPE_3_OF_3:
            tag = " <-- [In 3/3 FragPipe Top 20!]"
        elif gene in FRAGPIPE_2_OF_3:
            tag = " <-- [In 2/3 FragPipe Top 20!]"
        print(f"    {idx+1:2d}. {gene:<12} (Importance: {row['Importance']:.5f}){tag}")

    # Plot Top 20 Bar Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        x='Importance', y='Gene', hue='Gene',
        data=feat_imp_df, palette='viridis',
        legend=False, ax=ax
    )

    ax.set_title(f"Top 20 Predictive Proteins - MAXQUANT\n(Test AUC: {test_auc:.2f})", weight='bold')
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
    ax.set_ylabel("")
    ax.grid(axis='x', alpha=0.3)

    # Highlight proteins that also appeared in FragPipe Top 20
    for tick in ax.get_yticklabels():
        if tick.get_text() in (FRAGPIPE_3_OF_3 | FRAGPIPE_2_OF_3 | {'HP', 'VASP'}):
            tick.set_color('firebrick')
            tick.set_fontweight('bold')

    plt.tight_layout()
    pdf_obj.savefig(fig, bbox_inches='tight')
    plt.close()

    return feat_imp_df


if __name__ == "__main__":
    mq_data = get_maxquant_datasets(use_cache=True, base_dir="~/DTU-GALD")

    pdf_path = PLOT_DIR / "RF_Analysis_MaxQuant.pdf"
    with PdfPages(pdf_path) as pdf:
        mq_top20 = run_maxquant_random_forest(mq_data, pdf)

    mq_set = set(mq_top20['Gene'])
    overlap_3 = mq_set & FRAGPIPE_3_OF_3
    overlap_2 = mq_set & FRAGPIPE_2_OF_3

    print("\n" + "="*70)
    print("🔄 CROSS-ENGINE OVERLAP: MAXQUANT vs. FRAGPIPE TOP 20")
    print("="*70)
    print(f"  • Shared with 3/3 FragPipe workflows: {', '.join(sorted(overlap_3)) if overlap_3 else 'None'}")
    print(f"  • Shared with 2/3 FragPipe workflows: {', '.join(sorted(overlap_2)) if overlap_2 else 'None'}")
    print("="*70)
    print(f"✅ MaxQuant Random Forest plot saved to: {pdf_path}")

