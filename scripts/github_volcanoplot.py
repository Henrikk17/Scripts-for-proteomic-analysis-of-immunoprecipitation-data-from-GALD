#03/08-2026

#This script is made for generating a volcano plot from the FragPipe output data. 

import matplotlib
matplotlib.use('TkAgg')  # Forces Matplotlib to use an interactive X11 window
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import os

###

from load_fragPipe_proteomic_data import load_fragPipe_proteomic_data #Costum function for loading FragPipe output data

# Ensure output directory exists for saving files
os.makedirs("output", exist_ok=True)

#######################
##  Loading FragPipe data (from generated on the HPC for Run 4,5,6):
#######################

###

#Run 4: 
print("=== Loading Run4 FragPipe Data ===") 
ROOT_run4 = "/home/local/s215065/results/run4_results" #(!) Change this value accordingly (!) 
annotation_file_run4 = "experiment_annotation/experiment_annotation_run4_R876.tsv" 

master_prot_run4_fragPipe_df, master_pep_run4_fragPipe_df = load_fragPipe_proteomic_data(ROOT_run4, annotation_file_run4) 

###

#Run 5: 
print("\n=== Loading Run5 FragPipe Data ===") 
ROOT_run5 = "/home/local/s215065/results/run5_results" #(!) Change this value accordingly (!) 
annotation_file_run5 = "experiment_annotation/experiment_annotation_run5_R1101.tsv" 

master_prot_run5_fragPipe_df, master_pep_run5_fragPipe_df = load_fragPipe_proteomic_data(ROOT_run5, annotation_file_run5) 

###

#Run 6: 
print("\n=== Loading Run6 FragPipe Data ===") 
ROOT_run6 = "/home/local/s215065/results/run6_results" #(!) Change this value accordingly (!) 
annotation_file_run6 = "experiment_annotation/experiment_annotation_run6_R1344.tsv" 

master_prot_run6_fragPipe_df, master_pep_run6_fragPipe_df = load_fragPipe_proteomic_data(ROOT_run6, annotation_file_run6) 



#######################
##  Filtering out irrelevant parts of the data (e.g. the contaminanted GALD sample 14): 
#######################

#GALD14 is the contaminated GALD sample, we will filter it out in the downstream analysis.

master_prot_run4_fragPipe_df = master_prot_run4_fragPipe_df[master_prot_run4_fragPipe_df['Sample'] != 'sample_14'].copy()
master_pep_run4_fragPipe_df = master_pep_run4_fragPipe_df[master_pep_run4_fragPipe_df['Sample'] != 'sample_14'].copy()

master_prot_run5_fragPipe_df = master_prot_run5_fragPipe_df[master_prot_run5_fragPipe_df['Sample'] != 'sample_14'].copy()
master_pep_run5_fragPipe_df = master_pep_run5_fragPipe_df[master_pep_run5_fragPipe_df['Sample'] != 'sample_14'].copy()

master_prot_run6_fragPipe_df = master_prot_run6_fragPipe_df[(master_prot_run6_fragPipe_df['Sample'] != 'sample_14') & (master_prot_run6_fragPipe_df['Sample'] != 'sample_56')].copy()
master_pep_run6_fragPipe_df = master_pep_run6_fragPipe_df[(master_pep_run6_fragPipe_df['Sample'] != 'sample_14') & (master_pep_run6_fragPipe_df['Sample'] != 'sample_56')].copy()

print("Successfully filtered out sample_14 (and sample_56 from run6) from all DataFrames.")


#######################
##  Extracting Ceruloplasmin (CP) RAW data (Long Format):
#######################

print("\n--- Extracting CP raw data ---")

cp_run4 = master_prot_run4_fragPipe_df[master_prot_run4_fragPipe_df.get('Gene', pd.Series(dtype=str)) == 'CP'].copy()
if not cp_run4.empty: cp_run4.insert(0, 'Experiment_Run', 'Run 4')

cp_run5 = master_prot_run5_fragPipe_df[master_prot_run5_fragPipe_df.get('Gene', pd.Series(dtype=str)) == 'CP'].copy()
if not cp_run5.empty: cp_run5.insert(0, 'Experiment_Run', 'Run 5')

cp_run6 = master_prot_run6_fragPipe_df[master_prot_run6_fragPipe_df.get('Gene', pd.Series(dtype=str)) == 'CP'].copy()
if not cp_run6.empty: cp_run6.insert(0, 'Experiment_Run', 'Run 6')

# Combine and save as semicolon-separated CSV in long format
cp_combined_raw = pd.concat([cp_run4, cp_run5, cp_run6], ignore_index=True)
if not cp_combined_raw.empty:
    cp_raw_out = "output/CP_raw_data_combined.csv"
    cp_combined_raw.to_csv(cp_raw_out, sep=';', index=False)
    print(f"✅ Saved long-format RAW CP data to {cp_raw_out}")


#######################
##  Making the volcano plots: 
#######################

try:
    from adjustText import adjust_text
    adjust_text_available = True
except ImportError:
    print("\nWarning: adjustText module not found. Overlapping labels will not be automatically repelled.")
    adjust_text_available = False


print("\n--- Running Volcano Pipeline (Left-shifted Gaussian Imputation) ---")

# ==========================================
# CONFIGURATION
# ==========================================
SHIFT = 1.8
SCALE = 0.3
lfc_threshold = 1.0
p_adj_threshold = 0.05
log10_p_threshold = -np.log10(p_adj_threshold)

runs = [
    ('Run 4', master_prot_run4_fragPipe_df),
    ('Run 5', master_prot_run5_fragPipe_df),
    ('Run 6', master_prot_run6_fragPipe_df)
]

comparisons = [
    ('Healthy donor', 'Healthy mother', 'steelblue', 'Up in Healthy'),
    ('GALD_spouse', 'Spouses', 'forestgreen', 'Up in Spouse')
]

all_significant_hits = []
all_cp_stats = []
all_cp_imputed_data = [] # Container for the actual IMPUTED values in long format

# ==========================================
# MASTER LOOP
# ==========================================
for run_name, df_raw in runs:
    
    print(f"\n==========================================")
    print(f"⚙️ PREPPING & IMPUTING: {run_name}")
    print(f"==========================================")
    
    # Ensure numerical intensities
    df_raw['Razor Intensity'] = pd.to_numeric(df_raw['Razor Intensity'], errors='coerce')
    
    # 1. Extract Descriptions once per run
    cols_to_keep = ['Protein Group', 'Protein Description']
    if 'Gene' in df_raw.columns:
        cols_to_keep.insert(1, 'Gene')
    info_dict = df_raw[cols_to_keep].drop_duplicates(subset='Protein Group')
    
    # 2. Pivot & Log2 Transform the ENTIRE run (Global Imputation)
    raw_df = df_raw.pivot_table(index='Protein Group', columns='Sample', values='Razor Intensity', aggfunc='sum')
    
    # Explicitly replace 0.0 with NaN so they are successfully log2 transformed into NaNs for imputation
    raw_df = raw_df.replace(0, np.nan)
    log_df = np.log2(raw_df)
    
    # Count how many missing values exist BEFORE we overwrite them!
    missing_values_count = log_df.isna().sum().sum()
    
    # 3. Stochastic Imputation (Globally)
    np.random.seed(42) 
    imputed_df = log_df.copy()
    
    for col in imputed_df.columns:
        sample_data = imputed_df[col].dropna()
        if sample_data.empty: continue
        
        sample_median = sample_data.median()
        sample_sd = sample_data.std()
        num_nans = imputed_df[col].isna().sum()
        
        if num_nans > 0:
            fake_data = np.random.normal(
                loc=sample_median - (SHIFT * sample_sd),
                scale=sample_sd * SCALE,
                size=num_nans
            )
            imputed_df.loc[imputed_df[col].isna(), col] = fake_data
            
    print(f"Imputed {missing_values_count} missing values for {run_name}.")

    # EXTRACT IMPUTED CP DATA (Log2 scale - Long Format)
    cp_groups = info_dict[info_dict['Gene'] == 'CP']['Protein Group'].tolist() if 'Gene' in info_dict.columns else []
    if cp_groups:
        cp_imp = imputed_df.loc[cp_groups].copy()
        cp_imp.reset_index(inplace=True)
        # Merge gene and description info back in
        cp_imp = pd.merge(cp_imp, info_dict, on='Protein Group', how='left')
        
        # Melt wide format back into long format to match raw format
        all_samples_in_run = df_raw['Sample'].unique().tolist()
        sample_cols = [c for c in cp_imp.columns if c in all_samples_in_run]
        meta_cols = [c for c in cp_imp.columns if c not in sample_cols]
        
        cp_imp_long = cp_imp.melt(id_vars=meta_cols, value_vars=sample_cols, var_name='Sample', value_name='Log2_Imputed_Intensity')
        cp_imp_long.insert(0, 'Experiment_Run', run_name)
        
        # Recover Sample_type and Group from raw data
        sample_metadata = df_raw[['Sample', 'Group', 'Sample_type']].drop_duplicates()
        cp_imp_long = pd.merge(cp_imp_long, sample_metadata, on='Sample', how='left')
        
        all_cp_imputed_data.append(cp_imp_long)

    # ==========================================
    # PAIRWISE COMPARISONS
    # ==========================================
    for ctrl_group, ctrl_display, ctrl_color, ctrl_label in comparisons:
        
        gald_cols = df_raw[df_raw['Group'] == 'GALD']['Sample'].unique().tolist()
        ctrl_cols = df_raw[df_raw['Group'] == ctrl_group]['Sample'].unique().tolist()
        
        if not gald_cols or not ctrl_cols:
            print(f"⏩ Skipping GALD vs {ctrl_display} (Cohort missing)")
            continue
            
        print(f"\n📊 Plotting {run_name}: GALD vs {ctrl_display}...")

        # 4. Statistical Testing
        log2_fc = imputed_df[gald_cols].mean(axis=1) - imputed_df[ctrl_cols].mean(axis=1)
        
        with np.errstate(invalid='ignore'):
            _, p_vals = ttest_ind(imputed_df[gald_cols], imputed_df[ctrl_cols], axis=1, equal_var=False)
            
        _, p_adj, _, _ = multipletests(pd.Series(p_vals).fillna(1.0), method='fdr_bh')

        results_df = pd.DataFrame({
            'Protein Group': imputed_df.index,
            'Log2_Fold_Change': log2_fc.values,
            'P_Value': p_vals,
            'Adjusted_P_Value': p_adj,
            '-Log10_P_Adj': -np.log10(p_adj)
        })

        results_df = pd.merge(results_df, info_dict, on='Protein Group', how='left')

        # 5. Significance Categorization
        def categorize(row):
            if row['-Log10_P_Adj'] >= log10_p_threshold and row['Log2_Fold_Change'] >= lfc_threshold:
                return 'Up in GALD'
            elif row['-Log10_P_Adj'] >= log10_p_threshold and row['Log2_Fold_Change'] <= -lfc_threshold:
                return ctrl_label
            else:
                return 'Not Significant'

        results_df['Significance'] = results_df.apply(categorize, axis=1)

        # 6. Save Significant Hits & CP Stats
        sig_df = results_df[results_df['Significance'] != 'Not Significant'].copy()
        if not sig_df.empty:
            sig_df['Experiment_Run'] = run_name
            sig_df['Comparison'] = f"GALD vs {ctrl_display}"
            all_significant_hits.append(sig_df)

        # Save CP specific statistical stats regardless of significance
        if 'Gene' in results_df.columns:
            cp_stats = results_df[results_df['Gene'] == 'CP'].copy()
            if not cp_stats.empty:
                cp_stats.insert(0, 'Comparison', f"GALD vs {ctrl_display}")
                cp_stats.insert(0, 'Experiment_Run', run_name)
                all_cp_stats.append(cp_stats)

        # ==========================================
        # 7. DRAW THE VOLCANO PLOT
        # ==========================================
        fig, ax = plt.subplots(figsize=(12, 9))
        palette = {'Not Significant': 'lightgray', 'Up in GALD': 'firebrick', ctrl_label: ctrl_color}

        cp_mask = results_df['Gene'] == 'CP' if 'Gene' in results_df.columns else pd.Series([False] * len(results_df))
        skp1_mask = results_df['Gene'] == 'SKP1' if 'Gene' in results_df.columns else pd.Series([False] * len(results_df))
        combined_mask = cp_mask | skp1_mask

        cp_row = results_df[cp_mask]
        skp1_row = results_df[skp1_mask]
        non_target_df = results_df[~combined_mask]

        sns.scatterplot(
            data=non_target_df, x='Log2_Fold_Change', y='-Log10_P_Adj',
            hue='Significance', palette=palette, s=70, alpha=0.8,
            edgecolor=None, ax=ax
        )

        if not cp_row.empty:
            cp_sig = cp_row['Significance'].values[0]
            ax.scatter(cp_row['Log2_Fold_Change'], cp_row['-Log10_P_Adj'], color='darkorchid', marker='D', s=130, zorder=5, edgecolor='black', linewidth=0.6, label=f'CP ({cp_sig})')
            ax.annotate('CP', xy=(cp_row['Log2_Fold_Change'].values[0], cp_row['-Log10_P_Adj'].values[0]),
                        xytext=(cp_row['Log2_Fold_Change'].values[0] + 0.3, cp_row['-Log10_P_Adj'].values[0] + 0.3),
                        fontsize=10, weight='bold', color='darkorchid', arrowprops=dict(arrowstyle='->', color='darkorchid', lw=0.8))

        if not skp1_row.empty:
            skp1_sig = skp1_row['Significance'].values[0]
            ax.scatter(skp1_row['Log2_Fold_Change'], skp1_row['-Log10_P_Adj'], color='dodgerblue', marker='D', s=130, zorder=5, edgecolor='black', linewidth=0.6, label=f'SKP1 ({skp1_sig})')
            ax.annotate('SKP1', xy=(skp1_row['Log2_Fold_Change'].values[0], skp1_row['-Log10_P_Adj'].values[0]),
                        xytext=(skp1_row['Log2_Fold_Change'].values[0] + 0.3, skp1_row['-Log10_P_Adj'].values[0] + 0.3),
                        fontsize=10, weight='bold', color='dodgerblue', arrowprops=dict(arrowstyle='->', color='dodgerblue', lw=0.8))

        ax.axvline(x=lfc_threshold, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=-lfc_threshold, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=log10_p_threshold, color='black', linestyle='--', alpha=0.5)

        top_hits_gald = results_df[results_df['Significance'] == 'Up in GALD'].sort_values(by=['-Log10_P_Adj', 'Log2_Fold_Change'], ascending=[False, False]).head(40)
        texts = []
        for _, row in top_hits_gald.iterrows():
            label_name = row['Gene'] if ('Gene' in row and pd.notna(row['Gene'])) else row['Protein Group']
            if label_name in ['CP', 'SKP1']: continue 
            texts.append(ax.text(row['Log2_Fold_Change'], row['-Log10_P_Adj'], str(label_name), fontsize=9, weight='bold', color='darkred'))

        top_hits_ctrl = results_df[results_df['Significance'] == ctrl_label].sort_values(by=['-Log10_P_Adj', 'Log2_Fold_Change'], ascending=[False, False]).head(40)
        for _, row in top_hits_ctrl.iterrows():
            label_name = row['Gene'] if ('Gene' in row and pd.notna(row['Gene'])) else row['Protein Group']
            if label_name in ['CP', 'SKP1']: continue 
            texts.append(ax.text(row['Log2_Fold_Change'], row['-Log10_P_Adj'], str(label_name), fontsize=9, weight='bold', color=ctrl_color))

        if adjust_text_available:
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5), ax=ax)

        ax.set_title(f"Volcano plot - GALD vs {ctrl_display} - Protein-level - {run_name} \nWith imputation", fontsize=16, weight='bold')
        ax.set_xlabel(r'$\log_2$ Fold Change (GALD / Control)', fontsize=14)
        ax.set_ylabel(r'$-\log_{10}$ Adjusted p-value (FDR)', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        ax.grid(alpha=0.2)
        plt.tight_layout()

        safe_ctrl = ctrl_display.replace(" ", "_")
        filename = f"output/Volcano_{run_name.replace(' ', '')}_GALD_vs_{safe_ctrl}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot locally to {filename}")

        plt.show(block=False)

# ==========================================
# 8. COMPILE MASTER RESULTS DATAFRAME
# ==========================================
print("\n====================================================================")
print("🏆 MASTER SIGNIFICANT GENES LIST (IMPUTED - ALL RUNS)")
print("====================================================================")

if all_significant_hits:
    master_sig_df = pd.concat(all_significant_hits, ignore_index=True)
    
    master_sig_df = master_sig_df[['Experiment_Run', 'Comparison', 'Protein Group', 'Gene', 'Significance', 'Log2_Fold_Change', 'P_Value', 'Adjusted_P_Value']]
    master_sig_df = master_sig_df.sort_values(by=['Experiment_Run', 'Comparison', 'Adjusted_P_Value'], ascending=[True, True, True])
    print(master_sig_df)
else:
    print("No significant hits found across any experiments using the current thresholds.")
    master_sig_df = pd.DataFrame()


if not master_sig_df.empty:
    output_csv = "output/All_Runs_Significant_Proteins.csv"
    master_sig_df.to_csv(output_csv, sep=';', index=False)
    print(f"✅ Master results table saved successfully to: {output_csv}")

if all_cp_stats:
    cp_stats_df = pd.concat(all_cp_stats, ignore_index=True)
    cp_stats_out = "output/CP_Statistics_All_Runs.csv"
    cp_stats_df.to_csv(cp_stats_out, sep=';', index=False)
    print(f"✅ CP statistical results across all runs saved to {cp_stats_out}")

if all_cp_imputed_data:
    cp_imputed_df = pd.concat(all_cp_imputed_data, ignore_index=True)
    cp_imputed_out = "output/CP_imputed_data_combined.csv"
    cp_imputed_df.to_csv(cp_imputed_out, sep=';', index=False)
    print(f"✅ Imputed (Log2) long-format CP intensity data across all runs saved to {cp_imputed_out}")

# plt.show()