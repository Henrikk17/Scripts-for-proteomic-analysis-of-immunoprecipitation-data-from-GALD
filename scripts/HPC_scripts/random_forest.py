# 12/11-2026
# Deterministic Random Forest Analysis with Train/Test Split

import os
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
from sklearn.metrics import roc_auc_score, accuracy_score
from matplotlib.backends.backend_pdf import PdfPages

from helper_functions.manage_datasets import get_all_fragpipe_datasets

# Ensure the new dedicated plot directory exists
os.makedirs("../plots/random_forest", exist_ok=True)

def process_rf_workflow(datasets_dict, wf_prefix, pdf_obj):
    """
    Merges R4, R5, and R6, applies within-run imputation and Z-scoring,
    splits data into train/test sets, and trains a Random Forest.
    Saves the top 20 features plot to the provided PDF object.
    """
    print(f"\n{'='*70}")
    print(f"🌲 RANDOM FOREST ANALYSIS: {wf_prefix.upper()}")
    print(f"{'='*70}")

    run_configs = [
        (f"{wf_prefix}_run4", "Run 4"),
        (f"{wf_prefix}_run5", "Run 5"),
        (f"{wf_prefix}_run6_no_LB", "Run 6")
    ]

    wide_runs = []
    sample_to_group = {}

    for key, run_label in run_configs:
        if key not in datasets_dict:
            continue

        df = datasets_dict[key]["prot"].dropna(subset=['Gene']).copy()

        if 'Razor Intensity' in df.columns:
            quant_col = 'Razor Intensity'
        elif 'MaxLFQ Intensity' in df.columns:
            quant_col = 'MaxLFQ Intensity'
        else:
            quant_col = 'Intensity'

        # Unique Sample ID to prevent collisions
        df['Unique_Sample'] = f"{run_label.replace(' ', '')}_" + df['Sample'].astype(str)

        for _, row in df.drop_duplicates('Unique_Sample').iterrows():
            u_samp = row['Unique_Sample']
            grp = str(row['Group']).strip()
            sample_to_group[u_samp] = 'GALD' if grp == 'GALD' else 'Control'

        # Pivot to [Unique_Sample x Gene]
        pivot_df = df.pivot_table(index='Unique_Sample', columns='Gene', values=quant_col, aggfunc='sum')
        pivot_df.replace(0, np.nan, inplace=True)
        pivot_df = pivot_df.dropna(thresh=3, axis=1) # Filter out extreme noise

        log2_df = np.log2(pivot_df)

        # Deterministic Imputation BEFORE Z-scoring
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

    # Strict Inner Join to evaluate only consistently detected proteins
    X = pd.concat(wide_runs, axis=0, join='inner')
    y = np.array([sample_to_group[s] for s in X.index])
    y_binary = (y == 'GALD').astype(int)

    print(f"  [*] Matrix Shape: {X.shape[0]} samples x {X.shape[1]} proteins")

    # ==========================================
    # Train / Test Split
    # ==========================================
    # 70% Train, 30% Test, stratified to maintain GALD/Control ratios
    X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = train_test_split(
        X, y, y_binary, test_size=0.3, random_state=42, stratify=y
    )

    print(f"  [*] Training Set: {X_train.shape[0]} samples")
    print(f"  [*] Testing Set:  {X_test.shape[0]} samples")

    # ==========================================
    # Random Forest Training & Evaluation
    # ==========================================
    rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    # Evaluate for Overfitting
    train_probs = rf.predict_proba(X_train)[:, 1]
    test_probs = rf.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_bin_train, train_probs)
    test_auc = roc_auc_score(y_bin_test, test_probs)

    print(f"\n  📊 Model Performance:")
    print(f"     Train ROC-AUC: {train_auc:.4f}")
    print(f"     Test ROC-AUC:  {test_auc:.4f}")

    if test_auc < 0.6:
        print("     ⚠️ WARNING: Low Test AUC indicates the model is struggling to generalize (Overfitting to noise).")
    elif train_auc - test_auc > 0.3:
        print("     ⚠️ WARNING: Large gap between Train and Test AUC suggests some overfitting.")
    else:
        print("     ✅ Model shows stable generalization.")

    # ==========================================
    # Feature Extraction & Plotting
    # ==========================================
    feat_imp_df = pd.DataFrame({
        'Gene': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(20).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        x='Importance', y='Gene', hue='Gene',
        data=feat_imp_df, palette='viridis',
        legend=False, ax=ax
    )

    ax.set_title(f"Top 20 Predictive Proteins - {wf_prefix.upper()}\n(Test AUC: {test_auc:.2f})", weight='bold')
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
    ax.set_ylabel("")
    ax.grid(axis='x', alpha=0.3)

    # Highlight known candidates
    for tick in ax.get_yticklabels():
        if tick.get_text() in ['CP', 'HP', 'VASP', 'C4BPA', 'PROS1', 'TAGLN2', 'TFRC', 'UGP2']:
            tick.set_color('firebrick')
            tick.set_fontweight('bold')

    plt.tight_layout()
    pdf_obj.savefig(fig, bbox_inches='tight')
    plt.close()

    return feat_imp_df

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("Loading cached datasets...")
    datasets = get_all_fragpipe_datasets(use_cache=True, load_peptides=False)

    workflows = ["imms", "comp", "glyco"]
    top20_by_wf = {}

    pdf_path = "../plots/random_forest/RF_Analysis_All_Workflows.pdf"
    print(f"\nSaving all plots to compiled PDF: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        for wf in workflows:
            top20_df = process_rf_workflow(datasets, wf, pdf)
            top20_by_wf[wf] = top20_df

    # Overlap Summary
    print("\n" + "="*70)
    print("🔄 OVERLAP OF TOP 20 RANDOM FOREST PROTEINS ACROSS WORKFLOWS")
    print("="*70)

    gene_to_workflows = {}
    for wf, df_top in top20_by_wf.items():
        for rank, row in df_top.iterrows():
            gene = row['Gene']
            gene_to_workflows.setdefault(gene, []).append(f"{wf.upper()}(#{rank+1})")

    overlapping_genes = {g: wfs for g, wfs in gene_to_workflows.items() if len(wfs) >= 2}

    if overlapping_genes:
        sorted_overlaps = sorted(overlapping_genes.items(), key=lambda x: (-len(x[1]), x[0]))
        for gene, wfs in sorted_overlaps:
            print(f"  • {gene:<12} -> Appears in {len(wfs)}/3 workflows: {', '.join(wfs)}")
    else:
        print("  None of the Top 20 proteins overlapped across workflows.")
    print("="*70)
