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
import seaborn as sns

#########################

import os
import matplotlib
# Use 'TkAgg' only if an X11 display is available on the HPC, otherwise fallback to headless 'Agg'
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

from helper_functions.load_fragPipe_proteomic_data import load_fragPipe_proteomic_data

# Ensure output directory exists for saving files/plots
os.makedirs("../plots", exist_ok=True)

#######################
##  1. Loading FragPipe Data (Run 4, 5, 6)
#######################

# Annotation files
annotation_file_run4 = "../experiment_annotation/experiment_annotation_run4_R876.tsv"
annotation_file_run5 = "../experiment_annotation/experiment_annotation_run5_R1101.tsv"
annotation_file_run6 = "../experiment_annotation/experiment_annotation_run6_R1344.tsv"

# --- A. Loading data processed with N-glycosylation workflow ---
print("=== Loading Glyco-N Workflow Data ===")

print("\n[Glyco-N] Loading Run 4...")
ROOT_glyco_run4 = "/home/local/s215065/results/glyco_N_processed/run4_results_no_GALD14"
master_prot_run4_glyco_df, master_pep_run4_glyco_df = load_fragPipe_proteomic_data(ROOT_glyco_run4, annotation_file_run4)

print("\n[Glyco-N] Loading Run 5...")
ROOT_glyco_run5 = "/home/local/s215065/results/glyco_N_processed/run5_results_no_GALD14"
master_prot_run5_glyco_df, master_pep_run5_glyco_df = load_fragPipe_proteomic_data(ROOT_glyco_run5, annotation_file_run5)

print("\n[Glyco-N] Loading Run 6 (No Lysate Buffer)...")
ROOT_glyco_no_LB_run6 = "/home/local/s215065/results/glyco_N_processed/run6_results_no_GALD14_no_lysate_buffer"
master_prot_run6_no_LB_glyco_df, master_pep_run6_no_LB_glyco_df = load_fragPipe_proteomic_data(ROOT_glyco_no_LB_run6, annotation_file_run6)

print("\n[Glyco-N] Loading Run 6 (With Lysate Buffer)...")
ROOT_glyco_with_LB_run6 = "/home/local/s215065/results/glyco_N_processed/run6_results_no_GALD14_including_lysate_buffer"
master_prot_run6_with_LB_glyco_df, master_pep_run6_with_LB_glyco_df = load_fragPipe_proteomic_data(ROOT_glyco_with_LB_run6, annotation_file_run6)


# --- B. Loading data processed with LFQ_MBR_comparable_with_glyco_N ---
print("\n=== Loading LFQ-MBR Comparable Workflow Data ===")

print("\n[LFQ-MBR Comparable] Loading Run 4...")
ROOT_comp_run4 = "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run4_results"
master_prot_run4_comp_df, master_pep_run4_comp_df = load_fragPipe_proteomic_data(ROOT_comp_run4, annotation_file_run4)

print("\n[LFQ-MBR Comparable] Loading Run 5...")
ROOT_comp_run5 = "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run5_results"
master_prot_run5_comp_df, master_pep_run5_comp_df = load_fragPipe_proteomic_data(ROOT_comp_run5, annotation_file_run5)

print("\n[LFQ-MBR Comparable] Loading Run 6 (No Lysate Buffer)...")
ROOT_comp_no_LB_run6 = "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run6_results_no_LB"
master_prot_run6_no_LB_comp_df, master_pep_run6_no_LB_comp_df = load_fragPipe_proteomic_data(ROOT_comp_no_LB_run6, annotation_file_run6)

print("\n[LFQ-MBR Comparable] Loading Run 6 (With Lysate Buffer)...")
ROOT_comp_with_LB_run6 = "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run6_results_with_LB"
master_prot_run6_with_LB_comp_df, master_pep_run6_with_LB_comp_df = load_fragPipe_proteomic_data(ROOT_comp_with_LB_run6, annotation_file_run6)


# --- C. Loading data processed with LFQ_MBR_IM-MS ---
print("\n=== Loading LFQ-MBR IM-MS Workflow Data ===")

print("\n[LFQ-MBR IM-MS] Loading Run 4...")
ROOT_imms_run4 = "/home/local/s215065/results/LFQ_MBR_IM-MS/run4_results"
master_prot_run4_imms_df, master_pep_run4_imms_df = load_fragPipe_proteomic_data(ROOT_imms_run4, annotation_file_run4)

print("\n[LFQ-MBR IM-MS] Loading Run 5...")
ROOT_imms_run5 = "/home/local/s215065/results/LFQ_MBR_IM-MS/run5_results"
master_prot_run5_imms_df, master_pep_run5_imms_df = load_fragPipe_proteomic_data(ROOT_imms_run5, annotation_file_run5)

print("\n[LFQ-MBR IM-MS] Loading Run 6 (No Lysate Buffer)...")
ROOT_imms_no_LB_run6 = "/home/local/s215065/results/LFQ_MBR_IM-MS/run6_results_no_LB"
master_prot_run6_no_LB_imms_df, master_pep_run6_no_LB_imms_df = load_fragPipe_proteomic_data(ROOT_imms_no_LB_run6, annotation_file_run6)

print("\n[LFQ-MBR IM-MS] Loading Run 6 (With Lysate Buffer)...")
ROOT_imms_with_LB_run6 = "/home/local/s215065/results/LFQ_MBR_IM-MS/run6_results_with_LB"
master_prot_run6_with_LB_imms_df, master_pep_run6_with_LB_imms_df = load_fragPipe_proteomic_data(ROOT_imms_with_LB_run6, annotation_file_run6)


#######################
##  2. Filtering Out Irrelevant Samples (GALD14)
#######################
print("\n" + "="*60)
print("CHECKING & FILTERING IRRELEVANT SAMPLES AND LYSATE BUFFERS")
print("="*60)

def check_and_filter_df(df, name, df_type="Protein"):
    """
    Checks for unexpected samples (sample_14, sample_56, Lysate buffer),
    alerts the user if found, and returns a clean filtered DataFrame.
    """
    # 1. Check for sample_14 and sample_56
    unexpected_samples = {'sample_14', 'sample_56'}
    found_unexpected = set(df['Sample'].unique()).intersection(unexpected_samples)
    if found_unexpected:
        print(f"(!) WARNING [{name} - {df_type}]: Unexpectedly found {found_unexpected}! Filtering them out...")

    # 2. Check for Lysate buffer
    if 'Sample_type' in df.columns:
        non_liver = df[df['Sample_type'] != 'Liver Lysate']['Sample_type'].unique()
        if len(non_liver) > 0:
            print(f"ℹ️  INFO [{name} - {df_type}]: Found non-Liver Lysate sample types {non_liver}. Filtering to 'Liver Lysate' only...")

    # Apply standard filters
    cond = (~df['Sample'].isin(unexpected_samples))
    if 'Sample_type' in df.columns:
        cond = cond & (df['Sample_type'] == 'Liver Lysate')

    return df[cond].copy()

# Apply filtering to all DataFrames
master_prot_run4_glyco_df = check_and_filter_df(master_prot_run4_glyco_df, "Glyco Run 4")
master_prot_run5_glyco_df = check_and_filter_df(master_prot_run5_glyco_df, "Glyco Run 5")
master_prot_run6_no_LB_glyco_df = check_and_filter_df(master_prot_run6_no_LB_glyco_df, "Glyco Run 6 (No LB)")
master_prot_run6_with_LB_glyco_df = check_and_filter_df(master_prot_run6_with_LB_glyco_df, "Glyco Run 6 (With LB)")

master_prot_run4_comp_df = check_and_filter_df(master_prot_run4_comp_df, "Comparable Run 4")
master_prot_run5_comp_df = check_and_filter_df(master_prot_run5_comp_df, "Comparable Run 5")
master_prot_run6_no_LB_comp_df = check_and_filter_df(master_prot_run6_no_LB_comp_df, "Comparable Run 6 (No LB)")
master_prot_run6_with_LB_comp_df = check_and_filter_df(master_prot_run6_with_LB_comp_df, "Comparable Run 6 (With LB)")

master_prot_run4_imms_df = check_and_filter_df(master_prot_run4_imms_df, "IM-MS Run 4")
master_prot_run5_imms_df = check_and_filter_df(master_prot_run5_imms_df, "IM-MS Run 5")
master_prot_run6_no_LB_imms_df = check_and_filter_df(master_prot_run6_no_LB_imms_df, "IM-MS Run 6 (No LB)")
master_prot_run6_with_LB_imms_df = check_and_filter_df(master_prot_run6_with_LB_imms_df, "IM-MS Run 6 (With LB)")

print("\n✅ Successfully verified and filtered all DataFrames.")


#########################

# Assuming you already have 'master_prot_run4_imms_df' loaded
# and it contains columns like 'Protein', 'Gene', 'Sample', 'Group', 'Intensity' (or MaxLFQ): 


def generate_volcano_plot(df, group_case='GALD', group_ctrl='Healthy donor', dataset_name='Run 4'):
    # 1. Pivot to Wide Format (Rows = Genes/Proteins, Columns = Samples)
    # Use your actual quantitative column name here (e.g., 'Intensity' or 'MaxLFQ Intensity')
    quant_col = 'Intensity' 
    
    wide_df = df.pivot_table(index='Gene', columns='Sample', values=quant_col, aggfunc='sum')
    
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

    # Filter to only proteins that have data in these groups
    imputed_df = imputed_df[case_samples + ctrl_samples].dropna()

    # 5. Calculate Fold Change and P-values
    results = []
    for gene, row in imputed_df.iterrows():
        case_vals = row[case_samples].values
        ctrl_vals = row[ctrl_samples].values
        
        # Log2 Fold Change (Case - Control, because values are already log2)
        log2fc = case_vals.mean() - ctrl_vals.mean()
        
        # T-test
        t_stat, p_val = ttest_ind(case_vals, ctrl_vals, equal_var=False)
        
        results.append({'Gene': gene, 'Log2FC': log2fc, 'p_value': p_val})

    res_df = pd.DataFrame(results).dropna()
    
    # Multiple testing correction (FDR / Benjamini-Hochberg)
    res_df['FDR'] = multipletests(res_df['p_value'], method='fdr_bh')[1]
    res_df['-log10(p_value)'] = -np.log10(res_df['p_value'])

    # 6. Plotting
    plt.figure(figsize=(10, 8))
    
    # Color mapping: Significant = FDR < 0.05 AND absolute Log2FC > 1
    sig_up = (res_df['FDR'] < 0.05) & (res_df['Log2FC'] > 1)
    sig_down = (res_df['FDR'] < 0.05) & (res_df['Log2FC'] < -1)
    
    plt.scatter(res_df['Log2FC'], res_df['-log10(p_value)'], color='grey', alpha=0.5, s=20, label='Not Sig')
    plt.scatter(res_df.loc[sig_up, 'Log2FC'], res_df.loc[sig_up, '-log10(p_value)'], color='red', alpha=0.7, s=30, label='Up in GALD')
    plt.scatter(res_df.loc[sig_down, 'Log2FC'], res_df.loc[sig_down, '-log10(p_value)'], color='blue', alpha=0.7, s=30, label='Down in GALD')
    
    # Highlight Ceruloplasmin (CP)
    cp_row = res_df[res_df['Gene'] == 'CP']
    if not cp_row.empty:
        plt.scatter(cp_row['Log2FC'], cp_row['-log10(p_value)'], color='gold', edgecolor='black', s=150, marker='*', label='Ceruloplasmin (CP)', zorder=5)
        plt.annotate('CP', (cp_row['Log2FC'].values[0], cp_row['-log10(p_value)'].values[0]), 
                     xytext=(5, 5), textcoords='offset points', fontweight='bold')

    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=-1, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1) # Note: line drawn at raw p=0.05 for visual aid
    
    plt.title(f'Volcano Plot: {group_case} vs {group_ctrl} ({dataset_name})')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 p-value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"../plots/Volcano_{dataset_name.replace(' ', '_')}_{group_case}_vs_{group_ctrl}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return res_df

# Example Execution:
# volcano_stats_run4 = generate_volcano_plot(master_prot_run4_imms_df, group_case='GALD', group_ctrl='Healthy donor', dataset_name='LFQ-IM-MS Run 4')