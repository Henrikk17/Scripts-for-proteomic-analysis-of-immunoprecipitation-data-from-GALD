# 09/11-2026
# Targeted Immunoglobulin (IG) Verification & LB Sample Analysis

import os
import re
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from helper_functions.manage_datasets import get_all_fragpipe_datasets
from helper_functions.load_fragPipe_proteomic_data import load_fragPipe_proteomic_data

os.makedirs("../plots", exist_ok=True)

TARGET_IGS = ['IGLV1-50', 'IGLV1-44', 'IGKV1-37', 'IGLV1-36', 'IGHV1-69D']

def alphanum_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(s))]

def calculate_stats(df, group_case, group_ctrl):
    quant_col = 'Razor Intensity' if 'Razor Intensity' in df.columns else 'Intensity'
    if 'MaxLFQ Intensity' in df.columns: quant_col = 'MaxLFQ Intensity'
    
    df_clean = df.dropna(subset=['Gene'])
    wide_df = df_clean.pivot_table(index='Gene', columns='Sample', values=quant_col, aggfunc='sum')
    wide_df.replace(0, np.nan, inplace=True)
    log2_df = np.log2(wide_df)
    
    np.random.seed(42)
    imputed_df = log2_df.copy()
    for col in imputed_df.columns:
        valid_values = imputed_df[col].dropna()
        if len(valid_values) > 2:
            mean_val = valid_values.mean()
            std_val = valid_values.std()
            null_count = imputed_df[col].isna().sum()
            imputed_df.loc[imputed_df[col].isna(), col] = np.random.normal(
                loc=mean_val - (1.8 * std_val), scale=std_val * 0.3, size=null_count)

    sample_to_group = df.drop_duplicates('Sample').set_index('Sample')['Group'].to_dict()
    case_samples = [s for s in imputed_df.columns if sample_to_group.get(s) == group_case]
    ctrl_samples = [s for s in imputed_df.columns if sample_to_group.get(s) == group_ctrl]

    if not case_samples or not ctrl_samples:
        return None

    imputed_df = imputed_df[case_samples + ctrl_samples].dropna()

    results = []
    for gene, row in imputed_df.iterrows():
        case_vals = row[case_samples].values
        ctrl_vals = row[ctrl_samples].values
        log2fc = case_vals.mean() - ctrl_vals.mean()
        _, p_val = ttest_ind(case_vals, ctrl_vals, equal_var=False)
        results.append({'Gene': gene, 'Log2FC': log2fc, 'p_value': p_val})

    res_df = pd.DataFrame(results).dropna()
    if res_df.empty: return None
    
    res_df['FDR'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
    res_df['-log10(p_value)'] = -np.log10(res_df['p_value'])
    return res_df

def plot_volcano(res_df, title, pdf):
    plt.figure(figsize=(10, 8))
    
    sig_up = (res_df['FDR'] < 0.05) & (res_df['Log2FC'] > 1)
    sig_down = (res_df['FDR'] < 0.05) & (res_df['Log2FC'] < -1)
    
    plt.scatter(res_df['Log2FC'], res_df['-log10(p_value)'], color='grey', alpha=0.3, s=20)
    plt.scatter(res_df.loc[sig_up, 'Log2FC'], res_df.loc[sig_up, '-log10(p_value)'], color='red', alpha=0.6, s=30)
    plt.scatter(res_df.loc[sig_down, 'Log2FC'], res_df.loc[sig_down, '-log10(p_value)'], color='blue', alpha=0.6, s=30)
    
    # Highlight Target IGs forcefully
    for gene in TARGET_IGS:
        row = res_df[res_df['Gene'] == gene]
        if not row.empty:
            x, y = row['Log2FC'].values[0], row['-log10(p_value)'].values[0]
            plt.scatter(x, y, color='gold', edgecolor='black', s=250, marker='*', zorder=5)
            plt.annotate(gene, (x, y), xytext=(6, 6), textcoords='offset points', fontweight='bold', fontsize=10)

    plt.axvline(x=1, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=-1, color='black', linestyle='--', alpha=0.5)
    
    # Draw FDR 0.05 line if it exists
    sig_pvals = res_df.loc[res_df['FDR'] < 0.05, 'p_value']
    if not sig_pvals.empty:
        plt.axhline(y=-np.log10(sig_pvals.max()), color='black', linestyle='--', alpha=0.5)

    plt.title(title, fontweight='bold')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 p-value')
    plt.grid(True, alpha=0.2)
    pdf.savefig(bbox_inches='tight')
    plt.close()

def plot_heatmap(pep_df, gene, run_name, pdf, group_palette):
    df_filtered = pep_df[pep_df.get("Gene", pd.Series(dtype=str)).astype(str).str.strip() == gene].copy()
    if df_filtered.empty: return
    
    pep_col = 'Peptide Sequence' if 'Peptide Sequence' in pep_df.columns else 'Peptide'
    quant_col = 'Razor Intensity' if 'Razor Intensity' in pep_df.columns else ('MaxLFQ Intensity' if 'MaxLFQ Intensity' in pep_df.columns else 'Intensity')

    df_filtered[quant_col] = pd.to_numeric(df_filtered[quant_col], errors='coerce')
    df_filtered['Sample_Label'] = df_filtered['Sample'].astype(str) + '\n' + df_filtered['Group'].astype(str).str.strip()
    
    sample_group_map = pep_df[['Sample', 'Group']].drop_duplicates()
    sample_group_map['Sample_Label'] = sample_group_map['Sample'].astype(str) + '\n' + sample_group_map['Group'].astype(str).str.strip()
    sample_group_dict = sample_group_map.set_index('Sample_Label')['Group'].str.strip().to_dict()
    
    sorted_samples = sorted(sample_group_dict.keys(), key=lambda x: (0 if sample_group_dict[x]=='GALD' else 1, alphanum_key(x)))

    pivot_df = df_filtered.pivot_table(index=pep_col, columns='Sample_Label', values=quant_col, aggfunc='sum')
    for s in sorted_samples:
        if s not in pivot_df.columns: pivot_df[s] = np.nan
    
    pivot_df = pivot_df[sorted_samples]
    log_df = np.log2(pivot_df.replace(0, np.nan))
    with np.errstate(invalid='ignore'):
        z_df = log_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1).dropna(how='all')
    
    if z_df.empty: return
    
    z_df = z_df.assign(na_count=z_df.isna().sum(axis=1), mean_int=log_df.mean(axis=1)).sort_values(by=['na_count', 'mean_int'], ascending=[True, False]).drop(columns=['na_count', 'mean_int'])
    
    fig, ax = plt.subplots(figsize=(max(12, len(sorted_samples)*0.45), max(4, len(z_df)*0.35))) 
    ax.set_facecolor('#404040')
    sns.heatmap(z_df, cmap='Oranges', center=0, vmin=-2.5, vmax=2.5, ax=ax, cbar_kws={'label': 'log2 Z-Score'}, linewidths=0.5, linecolor='white')
    
    ax.set_title(f"{gene} Peptides - {run_name}", fontweight='bold', pad=40)
    labels = ax.get_xticklabels()
    for tick in labels:
        lbl = tick.get_text()
        grp = sample_group_dict.get(lbl, "")
        tick.set_text(lbl.split('\n')[0])
        if grp in group_palette:
            tick.set_color("black")
            tick.set_fontweight('bold')
            tick.set_bbox(dict(facecolor=group_palette[grp], edgecolor='none', alpha=0.3, boxstyle='round,pad=0.3'))
    
    ax.set_xticklabels(labels, rotation=45, ha="right")
    legend_patches = [mpatches.Patch(color=color, alpha=0.6, label=grp) for grp, color in group_palette.items()]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ==========================================
# Main Execution
# ==========================================
print("Loading LFQ-Comparable standard datasets...")
all_datasets = get_all_fragpipe_datasets(use_cache=True, load_peptides=True)

# Extract standard runs for LFQ-Comparable
runs_to_analyze = {
    "R4_vs_Healthy": (all_datasets["comp_run4"]["prot"], all_datasets["comp_run4"]["pep"], 'GALD', 'Healthy donor'),
    "R5_vs_Spouse": (all_datasets["comp_run5"]["prot"], all_datasets["comp_run5"]["pep"], 'GALD', 'GALD_spouse'),
    "R6_vs_Healthy": (all_datasets["comp_run6_no_LB"]["prot"], all_datasets["comp_run6_no_LB"]["pep"], 'GALD', 'Healthy donor'),
    "R6_vs_Spouse": (all_datasets["comp_run6_no_LB"]["prot"], all_datasets["comp_run6_no_LB"]["pep"], 'GALD', 'GALD_spouse')
}

# Manually Load Run 6 LB Samples (Bypassing standard filter)
print("Loading raw Run 6 data to extract LB (Buffer) samples...")
root_r6 = "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run6_results_with_LB"
annot_r6 = "../experiment_annotation/experiment_annotation_run6_R1344.tsv"
raw_prot, raw_pep = load_fragPipe_proteomic_data(root_r6, annot_r6)

# Filter explicitly for Lysate buffer
prot_lb = raw_prot[(raw_prot['Sample_type'] != 'Liver Lysate') & (~raw_prot['Sample'].isin(['sample_14', 'sample_56']))].copy()
pep_lb = raw_pep[(raw_pep['Sample_type'] != 'Liver Lysate') & (~raw_pep['Sample'].isin(['sample_14', 'sample_56']))].copy()

runs_to_analyze["R6_LB_vs_Healthy"] = (prot_lb, pep_lb, 'GALD', 'Healthy donor')
runs_to_analyze["R6_LB_vs_Spouse"] = (prot_lb, pep_lb, 'GALD', 'GALD_spouse')

group_palette = {'GALD': 'firebrick', 'Healthy donor': 'steelblue', 'GALD_spouse': 'forestgreen'}
pdf_path = "../plots/IG_Target_Analysis.pdf"
summary_records = []

with PdfPages(pdf_path) as pdf:
    for name, (prot, pep, case, ctrl) in runs_to_analyze.items():
        print(f"\n[*] Processing {name}...")
        res_df = calculate_stats(prot, case, ctrl)
        
        if res_df is not None:
            # 1. Plot Volcano
            plot_volcano(res_df, f"Volcano: {name.replace('_', ' ')}", pdf)
            
            # 2. Extract stats for print table
            for gene in TARGET_IGS:
                row = res_df[res_df['Gene'] == gene]
                if not row.empty:
                    d = row.iloc[0]
                    summary_records.append({
                        'Run': name, 'Gene': gene, 'Log2FC': d['Log2FC'], 
                        'P-value': d['p_value'], 'FDR': d['FDR'],
                        'Sig': '*' if d['p_value'] < 0.05 else ''
                    })
            
            # 3. Plot Heatmaps
            for gene in TARGET_IGS:
                plot_heatmap(pep, gene, name.replace('_', ' '), pdf, group_palette)

print("\n" + "="*70)
print(" TARGET IMMUNOGLOBULIN (IG) SUMMARY TABLE")
print("="*70)
if summary_records:
    df_sum = pd.DataFrame(summary_records)
    df_sum['Log2FC'] = df_sum['Log2FC'].map("{:.2f}".format)
    df_sum['P-value'] = df_sum['P-value'].map("{:.4e}".format)
    df_sum['FDR'] = df_sum['FDR'].map("{:.4e}".format)
    # Pivot for easier reading
    print(df_sum.to_string(index=False))
else:
    print("None of the target IGs were found in any dataset.")
print("="*70)
print(f"✅ All Volcano plots and Heatmaps saved to: {pdf_path}")
