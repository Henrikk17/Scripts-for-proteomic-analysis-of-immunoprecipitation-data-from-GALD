# 11/11-2026
# Strictly Verified Multivariate Signature Analysis (PCA & Random Forest)

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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from helper_functions.manage_datasets import get_all_fragpipe_datasets

os.makedirs("../plots/multivariate", exist_ok=True)

def process_multivariate_workflow(datasets_dict, wf_prefix):
    """
    Merges R4, R5, and R6 (No LB) with unique Run-Sample IDs, performs within-run
    Perseus imputation followed by within-run Z-scoring, and runs PCA & Random Forest.
    Returns the list of Top 20 predictive genes.
    """
    print(f"\n{'='*70}")
    print(f"🚀 RUNNING MULTIVARIATE ANALYSIS: {wf_prefix.upper()}")
    print(f"{'='*70}")

    run_configs = [
        (f"{wf_prefix}_run4", "Run 4"),
        (f"{wf_prefix}_run5", "Run 5"),
        (f"{wf_prefix}_run6_no_LB", "Run 6")
    ]

    wide_runs = []
    sample_to_group = {}
    sample_to_run = {}

    for key, run_label in run_configs:
        if key not in datasets_dict:
            continue

        df = datasets_dict[key]["prot"].dropna(subset=['Gene']).copy()

        # Auto-detect quantification column
        if 'Razor Intensity' in df.columns:
            quant_col = 'Razor Intensity'
        elif 'MaxLFQ Intensity' in df.columns:
            quant_col = 'MaxLFQ Intensity'
        else:
            quant_col = 'Intensity'

        # Create a globally unique Sample ID so sample_1 in R4 doesn't collide with sample_1 in R6
        df['Unique_Sample'] = f"{run_label.replace(' ', '')}_" + df['Sample'].astype(str)

        for _, row in df.drop_duplicates('Unique_Sample').iterrows():
            u_samp = row['Unique_Sample']
            grp = str(row['Group']).strip()
            sample_to_group[u_samp] = 'GALD' if grp == 'GALD' else 'Control'
            sample_to_run[u_samp] = run_label

        # Pivot to [Unique_Sample x Gene]
        pivot_df = df.pivot_table(index='Unique_Sample', columns='Gene', values=quant_col, aggfunc='sum')
        pivot_df.replace(0, np.nan, inplace=True)

        # Drop proteins detected in fewer than 3 samples in this run to avoid zero-variance scaling
        pivot_df = pivot_df.dropna(thresh=3, axis=1)

        # Log2 Transform
        log2_df = np.log2(pivot_df)

        # Deterministic Perseus-style Imputation (per sample within run) BEFORE Z-scoring
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

        # Within-Run Z-Score across samples (removes MS batch intensity offsets)
        scaler = StandardScaler()
        z_array = scaler.fit_transform(imputed_run_df)
        z_df = pd.DataFrame(z_array, index=imputed_run_df.index, columns=imputed_run_df.columns)

        wide_runs.append(z_df)

    # 2. Merge all runs — use 'inner' join so we only evaluate proteins consistently quantified in all 3 runs
    # (Using 'outer' join forces cross-batch imputation where an entire run is missing a protein)
    X = pd.concat(wide_runs, axis=0, join='inner')

    y = np.array([sample_to_group[s] for s in X.index])
    runs_vec = [sample_to_run[s] for s in X.index]

    n_gald = np.sum(y == 'GALD')
    n_ctrl = np.sum(y == 'Control')
    print(f"  [*] Merged Matrix (Strict Inner Join across R4, R5, R6): {X.shape[0]} samples ({n_gald} GALD, {n_ctrl} Control) x {X.shape[1]} proteins")

    # ==========================================
    # 3. Principal Component Analysis (PCA)
    # ==========================================
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)

    pca_df = pd.DataFrame(data=pcs, columns=['PC1', 'PC2'], index=X.index)
    pca_df['Group'] = y
    pca_df['Run'] = runs_vec

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='Group', style='Run',
        data=pca_df,
        palette={'GALD': 'firebrick', 'Control': 'steelblue'},
        markers={'Run 4': 'o', 'Run 5': 's', 'Run 6': '^'},
        s=110, alpha=0.85, ax=ax
    )

    ax.set_title(f"PCA: GALD vs Control Across Runs 4, 5, 6 ({wf_prefix.upper()})", weight='bold')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f"../plots/multivariate/PCA_{wf_prefix}.png", dpi=300)
    plt.close()

    # ==========================================
    # 4. Random Forest & Cross-Validated Accuracy
    # ==========================================
    rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
    
    # Evaluate true out-of-bag / cross-validated ROC-AUC to see if the model actually generalizes
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(rf, X, (y == 'GALD').astype(int), cv=cv, scoring='roc_auc')
    print(f"  [*] 5-Fold Cross-Validated ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    rf.fit(X, y)

    # Extract Top 20 Features
    feat_imp_df = pd.DataFrame({
        'Gene': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(20).reset_index(drop=True)

    # Fix Seaborn v0.14+ deprecation warning by assigning hue='Gene' and legend=False
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        x='Importance', y='Gene', hue='Gene',
        data=feat_imp_df, palette='viridis',
        legend=False, ax=ax
    )
    ax.set_title(f"Top 20 Predictive Proteins (Random Forest) - {wf_prefix.upper()}", weight='bold')
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
    ax.set_ylabel("")
    ax.grid(axis='x', alpha=0.3)

    # Highlight known candidates if present
    for tick in ax.get_yticklabels():
        if tick.get_text() in ['CP', 'HP', 'VASP', 'C4BPA', 'PROS1', 'TAGLN2']:
            tick.set_color('firebrick')
            tick.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(f"../plots/multivariate/RF_Importance_{wf_prefix}.png", dpi=300)
    plt.close()

    # Print all Top 20 Features cleanly
    print(f"\n  🏆 Top 20 Predictive Proteins ({wf_prefix.upper()}):")
    for idx, row in feat_imp_df.iterrows():
        print(f"    {idx+1:2d}. {row['Gene']:<12} (Importance: {row['Importance']:.5f})")

    return feat_imp_df


# ==========================================
# Main Execution & Cross-Workflow Overlap
# ==========================================
if __name__ == "__main__":
    print("Loading cached datasets...")
    datasets = get_all_fragpipe_datasets(use_cache=True, load_peptides=False)

    workflows = ["imms", "comp", "glyco"]
    top20_by_wf = {}

    for wf in workflows:
        top20_df = process_multivariate_workflow(datasets, wf)
        top20_by_wf[wf] = top20_df

    # Compute Overlap Across Workflows
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
