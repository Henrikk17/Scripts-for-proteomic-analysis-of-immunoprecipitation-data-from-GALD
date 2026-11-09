# 09/11-2026
# Multivariate Signature Analysis (PCA & Random Forest) across merged runs

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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from helper_functions.manage_datasets import get_all_fragpipe_datasets

os.makedirs("../plots/multivariate", exist_ok=True)

def process_multivariate_workflow(datasets_dict, wf_prefix):
    """
    Merges R4, R5, and R6 (No LB) for a given workflow, applies within-run Z-scoring,
    imputes missing data, and runs PCA & Random Forest.
    """
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING MULTIVARIATE ANALYSIS: {wf_prefix}")
    print(f"{'='*60}")

    # 1. Gather and pivot the 3 independent runs
    run_keys = [f"{wf_prefix}_run4", f"{wf_prefix}_run5", f"{wf_prefix}_run6_no_LB"]
    
    wide_runs = []
    sample_to_group = {}
    
    for key in run_keys:
        if key not in datasets_dict:
            continue
            
        df = datasets_dict[key]["prot"].dropna(subset=['Gene']).copy()
        
        # Determine quant column
        quant_col = 'Razor Intensity' if 'Razor Intensity' in df.columns else 'Intensity'
        if 'MaxLFQ Intensity' in df.columns: quant_col = 'MaxLFQ Intensity'
        
        # Build master sample dictionary (Control group simplified to 'Control')
        for _, row in df.drop_duplicates('Sample').iterrows():
            grp = row['Group']
            sample_to_group[row['Sample']] = 'GALD' if grp == 'GALD' else 'Control'

        # Pivot to [Samples x Genes]
        pivot_df = df.pivot_table(index='Sample', columns='Gene', values=quant_col, aggfunc='sum')
        pivot_df.replace(0, np.nan, inplace=True)
        
        # Log2 Transform
        log2_df = np.log2(pivot_df)
        
        # WITHIN-RUN Z-SCORE (Crucial for removing batch effects before merging)
        scaler = StandardScaler()
        z_array = scaler.fit_transform(log2_df)
        z_df = pd.DataFrame(z_array, index=log2_df.index, columns=log2_df.columns)
        
        wide_runs.append(z_df)

    # 2. Merge all runs into one master matrix
    master_df = pd.concat(wide_runs, axis=0, join='outer')
    
    # Filter for proteins present in at least 30% of all samples to reduce extreme noise
    thresh = int(0.30 * len(master_df))
    master_df = master_df.dropna(thresh=thresh, axis=1)
    
    # 3. Impute remaining missing values (using median of the standardized values)
    imputer = SimpleImputer(strategy='median')
    imputed_array = imputer.fit_transform(master_df)
    X = pd.DataFrame(imputed_array, index=master_df.index, columns=master_df.columns)
    
    # Target vector (y)
    y = np.array([sample_to_group[s] for s in X.index])
    
    print(f"  [*] Merged Matrix: {X.shape[0]} samples x {X.shape[1]} proteins")

    # ==========================================
    # 4. Principal Component Analysis (PCA)
    # ==========================================
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    
    pca_df = pd.DataFrame(data=pcs, columns=['PC1', 'PC2'])
    pca_df['Group'] = y
    pca_df['Sample'] = X.index
    # Map back original run for visualization
    pca_df['Run'] = pca_df['Sample'].apply(lambda x: 'Run 4' if 'R876' in x or 'run4' in x.lower() 
                                           else ('Run 5' if 'R1101' in x or 'run5' in x.lower() else 'Run 6'))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Group', style='Run', data=pca_df, 
                    palette={'GALD': 'firebrick', 'Control': 'steelblue'}, s=100, alpha=0.8, ax=ax)
    
    ax.set_title(f"PCA: GALD vs Control Across All Runs ({wf_prefix})", weight='bold')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"../plots/multivariate/PCA_{wf_prefix}.png", dpi=300)
    plt.close()

    # ==========================================
    # 5. Random Forest Feature Importance
    # ==========================================
    rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    
    # Extract and sort feature importances
    importances = rf.feature_importances_
    feat_imp_df = pd.DataFrame({'Gene': X.columns, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Gene', data=feat_imp_df, palette='viridis', ax=ax)
    ax.set_title(f"Top 20 Predictive Proteins (Random Forest) - {wf_prefix}", weight='bold')
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
    ax.set_ylabel("")
    ax.grid(axis='x', alpha=0.3)
    
    # Highlight specific proteins of interest if they appear in the top 20
    for i, tick in enumerate(ax.get_yticklabels()):
        if tick.get_text() in ['CP', 'HP', 'VASP', 'C4BPA']:
            tick.set_color('firebrick')
            tick.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(f"../plots/multivariate/RF_Importance_{wf_prefix}.png", dpi=300)
    plt.close()
    
    print(f"  ✅ Top 5 Features: {', '.join(feat_imp_df['Gene'].head(5).tolist())}")
    print(f"  ✅ Plots saved to ../plots/multivariate/")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("Loading cached datasets...")
    datasets = get_all_fragpipe_datasets(use_cache=True, load_peptides=False)

    process_multivariate_workflow(datasets, "imms")
    process_multivariate_workflow(datasets, "comp")
    process_multivariate_workflow(datasets, "glyco")
