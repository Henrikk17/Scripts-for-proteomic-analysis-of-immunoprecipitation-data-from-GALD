# 09/11-2026
# Automated Volcano Plot Analysis with Multi-page PDF & CP Tracking

import os
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
from matplotlib.backends.backend_pdf import PdfPages

# Import the new data loading helper
from helper_functions.manage_datasets import get_all_fragpipe_datasets

# Ensure output directory exists for saving plots
os.makedirs("../plots", exist_ok=True)

def generate_volcano_plot(df, group_case, group_ctrl, dataset_name, pdf_obj):
    """
    Generates a volcano plot, saves it to a multi-page PDF, 
    and returns a dictionary tracking Ceruloplasmin (CP) significance.
    """
    # Auto-detect the correct intensity column
    if 'Razor Intensity' in df.columns:
        quant_col = 'Razor Intensity'
    elif 'MaxLFQ Intensity' in df.columns:
        quant_col = 'MaxLFQ Intensity'
    elif 'Intensity' in df.columns:
        quant_col = 'Intensity'
    else:
        print(f"  [-] Could not find an intensity column in {dataset_name}. Skipping.")
        return None

    # Ensure the Gene column exists and drop unmapped proteins
    if 'Gene' not in df.columns:
        print(f"  [-] Missing 'Gene' column in {dataset_name}. Skipping.")
        return None
        
    df_clean = df.dropna(subset=['Gene'])
    wide_df = df_clean.pivot_table(index='Gene', columns='Sample', values=quant_col, aggfunc='sum')
    
    # Replace 0 with NaN first to avoid -inf in log2
    wide_df.replace(0, np.nan, inplace=True)
    log2_df = np.log2(wide_df)
    
    # Perseus-style Imputation for Missing Values
    imputed_df = log2_df.copy()
    for col in imputed_df.columns:
        valid_values = imputed_df[col].dropna()
        if len(valid_values) > 2:
            mean_val = valid_values.mean()
            std_val = valid_values.std()
            
            # Generate random values from down-shifted distribution
            null_count = imputed_df[col].isna().sum()
            imputed_values = np.random.normal(loc=mean_val - (1.8 * std_val), 
                                              scale=std_val * 0.3, 
                                              size=null_count)
            imputed_df.loc[imputed_df[col].isna(), col] = imputed_values

    # Map Samples to Groups
    sample_to_group = df.drop_duplicates('Sample').set_index('Sample')['Group'].to_dict()
    case_samples = [s for s in imputed_df.columns if sample_to_group.get(s) == group_case]
    ctrl_samples = [s for s in imputed_df.columns if sample_to_group.get(s) == group_ctrl]

    if not case_samples or not ctrl_samples:
        print(f"  [-] Missing samples for {group_case} or {group_ctrl}. Skipping plot.")
        return None

    imputed_df = imputed_df[case_samples + ctrl_samples].dropna()

    # Calculate Fold Change and P-values
    results = []
    for gene, row in imputed_df.iterrows():
        case_vals = row[case_samples].values
        ctrl_vals = row[ctrl_samples].values
        
        log2fc = case_vals.mean() - ctrl_vals.mean()
        t_stat, p_val = ttest_ind(case_vals, ctrl_vals, equal_var=False)
        results.append({'Gene': gene, 'Log2FC': log2fc, 'p_value': p_val})

    res_df = pd.DataFrame(results).dropna()
    
    if res_df.empty:
        print(f"  [-] No valid statistics calculated. Skipping plot.")
        return None
    
    # Multiple testing correction (FDR / Benjamini-Hochberg)
    res_df['FDR'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
    res_df['-log10(p_value)'] = -np.log10(res_df['p_value'])

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    sig_up = (res_df['FDR'] < 0.05) & (res_df['Log2FC'] > 1)
    sig_down = (res_df['FDR'] < 0.05) & (res_df['Log2FC'] < -1)
    
    plt.scatter(res_df['Log2FC'], res_df['-log10(p_value)'], color='grey', alpha=0.3, s=20, label='Not Sig')
    plt.scatter(res_df.loc[sig_up, 'Log2FC'], res_df.loc[sig_up, '-log10(p_value)'], color='red', alpha=0.6, s=30, label=f'Up in {group_case}')
    plt.scatter(res_df.loc[sig_down, 'Log2FC'], res_df.loc[sig_down, '-log10(p_value)'], color='blue', alpha=0.6, s=30, label=f'Down in {group_case}')
    
    # Highlight CP and check significance
    cp_info = {
        'Dataset': dataset_name,
        'Comparison': f"{group_case} vs {group_ctrl}",
        'Status': 'Not Found',
        'Log2FC': np.nan,
        'FDR': np.nan
    }
    
    cp_row = res_df[res_df['Gene'] == 'CP']
    if not cp_row.empty:
        cp_lfc = cp_row['Log2FC'].values[0]
        cp_fdr = cp_row['FDR'].values[0]
        cp_info['Log2FC'] = cp_lfc
        cp_info['FDR'] = cp_fdr
        
        if cp_fdr < 0.05 and cp_lfc > 1:
            cp_info['Status'] = 'Sig Up'
        elif cp_fdr < 0.05 and cp_lfc < -1:
            cp_info['Status'] = 'Sig Down'
        else:
            cp_info['Status'] = 'Not Sig'

        plt.scatter(cp_lfc, cp_row['-log10(p_value)'].values[0], color='gold', edgecolor='black', s=200, marker='*', label='Ceruloplasmin (CP)', zorder=5)
        plt.annotate('CP', (cp_lfc, cp_row['-log10(p_value)'].values[0]), 
                     xytext=(5, 5), textcoords='offset points', fontweight='bold', fontsize=12)

    # Label Top 5 Up and Down genes (excluding CP to avoid double labeling)
    top_up = res_df[sig_up].nsmallest(5, 'FDR')
    top_down = res_df[sig_down].nsmallest(5, 'FDR')
    top_genes = pd.concat([top_up, top_down])
    
    for _, row in top_genes.iterrows():
        if row['Gene'] != 'CP':
            plt.annotate(row['Gene'], (row['Log2FC'], row['-log10(p_value)']),
                         xytext=(3, 3), textcoords='offset points', fontsize=9, alpha=0.8)

    plt.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=-1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.title(f'Volcano Plot: {group_case} vs {group_ctrl}\n({dataset_name})')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 p-value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the current figure to the PDF object
    pdf_obj.savefig(bbox_inches='tight')
    plt.close()
    
    return cp_info


# ==========================================
# Main Execution Loop
# ==========================================

print("Loading cached datasets...")
datasets = get_all_fragpipe_datasets(use_cache=True, load_peptides=False)

cp_tracker = []
pdf_path = "../plots/All_Volcano_Plots.pdf"

print(f"\nGenerating Volcano Plots into a single multi-page PDF: {pdf_path}")

# Open the multi-page PDF object
with PdfPages(pdf_path) as pdf:
    for key, data in datasets.items():
        label = data["label"]
        prot_df = data["prot"]
        comparisons = data["comparisons"]

        for group_case, group_ctrl in comparisons:
            print(f"[*] Processing {label}: {group_case} vs {group_ctrl}...")
            cp_result = generate_volcano_plot(
                df=prot_df, 
                group_case=group_case, 
                group_ctrl=group_ctrl, 
                dataset_name=label,
                pdf_obj=pdf
            )
            if cp_result:
                cp_tracker.append(cp_result)

# --- Print CP Significance Summary ---
print("\n" + "="*80)
print(" CERULOPLASMIN (CP) SIGNIFICANCE SUMMARY ACROSS ALL RUNS")
print("="*80)

cp_summary_df = pd.DataFrame(cp_tracker)

# Format the floats for cleaner terminal output
cp_summary_df['Log2FC'] = cp_summary_df['Log2FC'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "NaN")
cp_summary_df['FDR'] = cp_summary_df['FDR'].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "NaN")

print(cp_summary_df.to_string(index=False))
print("="*80)
print(f"✅ All plots successfully saved to: {pdf_path}")
