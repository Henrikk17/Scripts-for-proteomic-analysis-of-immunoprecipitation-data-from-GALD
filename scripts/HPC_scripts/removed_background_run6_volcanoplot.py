# 18/11-2026
# Delta Volcano Test: (Liver - LB) GALD vs Controls

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import loaders to use the fast caches
from helper_functions.master_data_handlers import get_maxquant_datasets, get_fragpipe_imms_datasets

# Configuration
P_CUTOFF = 0.05
LOG2FC_CUTOFF = 1.0
GLOBAL_SEED = 42
HIGHLIGHT_GENES = ["CP", "HP", "TFRC", "TFR2"]

CACHE_DIR = Path("../cache_master_pipeline")
OUT_DIR = Path("../plots/delta_volcano_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_delta_volcano(prot_df, engine_name):
    print(f"\n{'='*70}\n🚀 RUNNING DELTA VOLCANO: {engine_name}\n{'='*70}")
    
    quant_col = 'Razor Intensity' if 'Razor Intensity' in prot_df.columns else ('MaxLFQ Intensity' if 'MaxLFQ Intensity' in prot_df.columns else 'Intensity')
    df = prot_df.dropna(subset=['Gene']).copy()
    df['Gene'] = df['Gene'].astype(str).str.split(';').str[0].str.strip()
    
    # 1. Map Subject IDs and Matrix types
    df['Sample_Num'] = df['Sample'].astype(str).str.lower().str.replace('sample_', '', regex=False).astype(int)
    df['Subject_ID'] = df['Sample_Num'].apply(lambda n: n if n <= 42 else n - 42)
    df['Matrix'] = df['Sample_Num'].apply(lambda n: 'Liver' if n <= 42 else 'Buffer')
    
    if 'Buffer' not in df['Matrix'].values:
        print(f"⚠️ No LB samples found for {engine_name}. Skipping.")
        return []
        
    subject_to_group = df[['Subject_ID', 'Group']].dropna().drop_duplicates().set_index('Subject_ID')['Group'].to_dict()

    # 2. Pivot and Impute
    wide_df = df.pivot_table(index='Gene', columns='Sample', values=quant_col, aggfunc='sum').replace(0, np.nan).dropna(thresh=5, axis=0)
    
    rng = np.random.default_rng(GLOBAL_SEED)
    imputed_df = np.log2(wide_df).copy()
    for col in imputed_df.columns:
        valid = imputed_df[col].dropna()
        if len(valid) > 2:
            m, s = valid.mean(), valid.std()
            mask = imputed_df[col].isna()
            imputed_df.loc[mask, col] = rng.normal(loc=m - 1.8 * s, scale=s * 0.3, size=mask.sum())

    # 3. Pair subjects and calculate Delta: (Liver - LB)
    liver_cols = [c for c in imputed_df.columns if int(c.replace('sample_', '')) <= 42]
    buffer_cols = [c for c in imputed_df.columns if int(c.replace('sample_', '')) > 42]
    
    liver_df = pd.DataFrame(index=imputed_df.index)
    buffer_df = pd.DataFrame(index=imputed_df.index)
    
    for c in liver_cols: liver_df[int(c.replace('sample_', ''))] = imputed_df[c]
    for c in buffer_cols: buffer_df[int(c.replace('sample_', '')) - 42] = imputed_df[c]
        
    # Only keep subjects with BOTH a liver and a buffer sample
    common_subjects = sorted(list(set(liver_df.columns) & set(buffer_df.columns)))
    delta_df = liver_df[common_subjects] - buffer_df[common_subjects]
    
    # 4. Group Delta arrays by GALD vs Control
    gald_subjs = [s for s in common_subjects if subject_to_group.get(s, '') == 'GALD']
    ctrl_subjs = [s for s in common_subjects if subject_to_group.get(s, '') != 'GALD']
    
    gald_data = delta_df[gald_subjs].values
    ctrl_data = delta_df[ctrl_subjs].values
    genes = delta_df.index.values

    # 5. Calculate statistics on the Deltas
    log2fc = np.mean(gald_data, axis=1) - np.mean(ctrl_data, axis=1)
    _, p_vals = ttest_ind(gald_data, ctrl_data, axis=1, equal_var=False)
    
    v_df = pd.DataFrame({'Gene': genes, 'Log2FC_of_Delta': log2fc, 'p_value': p_vals}).dropna()
    v_df['FDR'] = multipletests(v_df['p_value'], method='fdr_bh')[1]
    
    # 6. Strict direction: GALD Delta must be significantly HIGHER than Control Delta
    sig = v_df[(v_df['FDR'] < P_CUTOFF) & (v_df['Log2FC_of_Delta'] >= LOG2FC_CUTOFF)].copy()
    down = v_df[(v_df['FDR'] < P_CUTOFF) & (v_df['Log2FC_of_Delta'] <= -LOG2FC_CUTOFF)].copy()
    
    # Print hits to terminal
    top_hits = sig.sort_values(by='FDR').head(10)['Gene'].tolist()
    hl_hits = [g for g in HIGHLIGHT_GENES if g in sig['Gene'].values]
    
    print(f"    [{engine_name} Delta Volcano Top 10] -> {', '.join(top_hits)}")
    if hl_hits: print(f"      * Target Hits heavily enriched above background: {', '.join(hl_hits)}")
    
    # Export significant hits
    out_txt = OUT_DIR / f"{engine_name}_Significantly_Enriched_Deltas.txt"
    if not sig.empty:
        sig.sort_values(by='FDR').to_csv(out_txt, sep='\t', index=False)
        print(f"    💾 Saved {len(sig)} significant GALD-enriched delta hits to {out_txt.name}")
    else:
        print(f"    ⚠️ No significant GALD-enriched delta hits found for {engine_name}.")

    # 7. Generate Volcano Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(v_df['Log2FC_of_Delta'], -np.log10(v_df['FDR']), alpha=0.5, color='grey', s=10, zorder=1)
    ax.scatter(sig['Log2FC_of_Delta'], -np.log10(sig['FDR']), alpha=0.8, color='firebrick', s=20, zorder=2)
    ax.scatter(down['Log2FC_of_Delta'], -np.log10(down['FDR']), alpha=0.8, color='steelblue', s=20, zorder=2)
    
    ax.set_title(f"{engine_name} Delta Volcano (Liver - LB): GALD vs Controls")
    ax.axvline(x=LOG2FC_CUTOFF, color='firebrick', linestyle=':', linewidth=1.5)
    ax.axvline(x=-LOG2FC_CUTOFF, color='steelblue', linestyle=':', linewidth=1.5)
    ax.axhline(y=-np.log10(P_CUTOFF), color='black', linestyle=':', linewidth=1.5)
    
    for _, row in sig.iterrows():
        if row['Gene'] not in HIGHLIGHT_GENES:
            ax.text(row['Log2FC_of_Delta'] + 0.05, -np.log10(row['FDR']), str(row['Gene']), 
                    fontsize=6, color='firebrick', va='bottom', ha='left')
            
    for _, row in down.iterrows():
        if row['Gene'] not in HIGHLIGHT_GENES:
            ax.text(row['Log2FC_of_Delta'] - 0.05, -np.log10(row['FDR']), str(row['Gene']), 
                    fontsize=6, color='steelblue', va='bottom', ha='right')

    # Highlight specific targets
    target_markers = ['*', '^', 'D', 's']
    target_colors = ['gold', 'lime', 'magenta', 'cyan']
    
    for idx, target in enumerate(HIGHLIGHT_GENES):
        if target in v_df['Gene'].values:
            row = v_df[v_df['Gene'] == target].iloc[0]
            m = target_markers[idx % len(target_markers)]
            c = target_colors[idx % len(target_colors)]
            
            ax.scatter(row['Log2FC_of_Delta'], -np.log10(row['FDR']), color=c, marker=m, 
                       s=120, edgecolor='black', linewidth=1, zorder=10)
            
            align_ha = 'left' if row['Log2FC_of_Delta'] >= 0 else 'right'
            offset = 0.08 if row['Log2FC_of_Delta'] >= 0 else -0.08
            
            ax.text(row['Log2FC_of_Delta'] + offset, -np.log10(row['FDR']), str(row['Gene']), 
                    fontsize=7, color='black', fontweight='bold', va='bottom', ha=align_ha, zorder=11)

    ax.set_xlabel(r'$\log_2(\text{Delta Fold Change})$ (GALD vs Control)')
    ax.set_ylabel(r'$-\log_{10}(\text{FDR})$')
    
    out_pdf = OUT_DIR / f"{engine_name}_Delta_Volcano.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    
    return sig['Gene'].tolist() if not sig.empty else []

if __name__ == "__main__":
    print("Loading cached Run 6 (with LB) datasets...")
    mq_all = get_maxquant_datasets(CACHE_DIR)
    fp_all = get_fragpipe_imms_datasets(CACHE_DIR)
    
    mq_genes = run_delta_volcano(mq_all["run6_with_LB"]["prot"], "MaxQuant")
    fp_genes = run_delta_volcano(fp_all["run6_with_LB"]["prot"], "FragPipe_IMMS")
    
    # 8. Compute and print the overlap
    overlap = set(mq_genes).intersection(set(fp_genes))
    
    print(f"\n{'='*70}\n🤝 CROSS-ENGINE OVERLAP (Significantly Abundant GALD Deltas)\n{'='*70}")
    if overlap:
        print(f"    Found {len(overlap)} overlapping proteins.")
        print(f"    Proteins: {', '.join(sorted(overlap))}")
        
        hl_overlap = [g for g in HIGHLIGHT_GENES if g in overlap]
        if hl_overlap:
            print(f"    ⭐ Target Hits in Overlap: {', '.join(hl_overlap)}")
    else:
        print("    No overlapping proteins found between the two engines for this specific test.")
        
    print(f"\n📁 Tests complete. Check {OUT_DIR.resolve()} for outputs.")