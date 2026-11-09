# 09/11-2026
# Automated Volcano Plot Analysis across all FragPipe Workflows

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

# Import the new data loading helper
from helper_functions.manage_datasets import get_all_fragpipe_datasets

# Ensure output directory exists for saving plots
os.makedirs("../plots", exist_ok=True)

def generate_volcano_plot(df, group_case='GALD', group_ctrl='Healthy donor', dataset_name='Run 4'):
    # 1. Pivot to Wide Format (Rows = Genes/Proteins, Columns = Samples)
    # Note: Update 'Intensity' to 'MaxLFQ Intensity' if you are relying strictly on the LFQ metric
    quant_col = 'Razor Intensity'

    # Auto-detect the correct intensity column
    if 'Razor Intensity' in df.columns:
        quant_col = 'Razor Intensity'
    elif 'MaxLFQ Intensity' in df.columns:
        quant_col = 'MaxLFQ Intensity'
    elif 'Intensity' in df.columns:
        quant_col = 'Intensity'
    else:
        print(f"  [-] Could not find an intensity column in {dataset_name}. Available columns: {list(df.columns)}")
        return None

    # Ensure the Gene column exists and drop unmapped proteins to keep the plot clean
    if 'Gene' not in df.columns:
        print(f"  [-] Missing 'Gene' column in {dataset_name}. Skipping.")
        return None

    df_clean = df.dropna(subset=['Gene'])
    wide_df = df_clean.pivot_table(index='Gene', columns='Sample', values=quant_col, aggfunc='sum')

    # 2. Log2 Transform
    # Replace 0 with NaN first to avoid -inf in log2
    wide_df.replace(0, np.nan, inplace=True)
    log2_df = np.log2(wide_df)

    # 3. Perseus-style Imputation for Missing Values
    # Shift mean by -1.8 standard deviations, width of 0.3 standard deviations
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

    # 4. Map Samples to Groups
    sample_to_group = df.drop_duplicates('Sample').set_index('Sample')['Group'].to_dict()
    case_samples = [s for s in imputed_df.columns if sample_to_group.get(s) == group_case]
    ctrl_samples = [s for s in imputed_df.columns if sample_to_group.get(s) == group_ctrl]

    # Filter to only proteins that have data in these groups (prevents errors if a group is entirely missing)
    if not case_samples or not ctrl_samples:
        print(f"  [-] Missing samples for {group_case} or {group_ctrl} in {dataset_name}. Skipping plot.")
        return None

    imputed_df = imputed_df[case_samples + ctrl_samples].dropna()

    # 5. Calculate Fold Change and P-values
    results = []
    for gene, row in imputed_df.iterrows():
        case_vals = row[case_samples].values
        ctrl_vals = row[ctrl_samples].values

        # Log2 Fold Change (Case - Control, because values are already log2)
        log2fc = case_vals.mean() - ctrl_vals.mean()

        # T-test (Welch's t-test for unequal variances)
        t_stat, p_val = ttest_ind(case_vals, ctrl_vals, equal_var=False)

        results.append({'Gene': gene, 'Log2FC': log2fc, 'p_value': p_val})

    res_df = pd.DataFrame(results).dropna()

    if res_df.empty:
        print(f"  [-] No valid statistics calculated for {dataset_name}. Skipping plot.")
        return None

    # Multiple testing correction (FDR / Benjamini-Hochberg)
    res_df['FDR'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
    res_df['-log10(p_value)'] = -np.log10(res_df['p_value'])

    # 6. Plotting
    plt.figure(figsize=(10, 8))

    # Color mapping: Significant = FDR < 0.05 AND absolute Log2FC > 1
    sig_up = (res_df['FDR'] < 0.05) & (res_df['Log2FC'] > 1)
    sig_down = (res_df['FDR'] < 0.05) & (res_df['Log2FC'] < -1)

    plt.scatter(res_df['Log2FC'], res_df['-log10(p_value)'], color='grey', alpha=0.5, s=20, label='Not Sig')
    plt.scatter(res_df.loc[sig_up, 'Log2FC'], res_df.loc[sig_up, '-log10(p_value)'], color='red', alpha=0.7, s=30, label=f'Up in {group_case}')
    plt.scatter(res_df.loc[sig_down, 'Log2FC'], res_df.loc[sig_down, '-log10(p_value)'], color='blue', alpha=0.7, s=30, label=f'Down in {group_case}')

    # Highlight Ceruloplasmin (CP)
    cp_row = res_df[res_df['Gene'] == 'CP']
    if not cp_row.empty:
        plt.scatter(cp_row['Log2FC'], cp_row['-log10(p_value)'], color='gold', edgecolor='black', s=200, marker='*', label='Ceruloplasmin (CP)', zorder=5)
        plt.annotate('CP', (cp_row['Log2FC'].values[0], cp_row['-log10(p_value)'].values[0]),
                     xytext=(5, 5), textcoords='offset points', fontweight='bold', fontsize=12)

    # Draw threshold lines
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.axvline(x=-1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1, alpha=0.7) # Visual aid for raw p=0.05

    plt.title(f'Volcano Plot: {group_case} vs {group_ctrl}\n({dataset_name})')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 p-value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Clean up dataset name for the output file name
    safe_name = dataset_name.replace(' | ', '_').replace(' ', '_').replace('(', '').replace(')', '')
    safe_case = group_case.replace(' ', '_')
    safe_ctrl = group_ctrl.replace(' ', '_')

    out_path = f"../plots/Volcano_{safe_name}_{safe_case}_vs_{safe_ctrl}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    return res_df


# ==========================================
# Main Execution Loop
# ==========================================

print("Loading cached datasets...")
# Load datasets (using fast cache)
datasets = get_all_fragpipe_datasets(use_cache=True, load_peptides=False)

print("\nGenerating Volcano Plots...")
for key, data in datasets.items():
    label = data["label"]
    prot_df = data["prot"]
    comparisons = data["comparisons"]

    for group_case, group_ctrl in comparisons:
        print(f"[*] Processing {label}: {group_case} vs {group_ctrl}...")
        generate_volcano_plot(
            df=prot_df,
            group_case=group_case,
            group_ctrl=group_ctrl,
            dataset_name=label
        )

print("\n✅ All Volcano plots successfully generated in the '../plots/' directory.")
