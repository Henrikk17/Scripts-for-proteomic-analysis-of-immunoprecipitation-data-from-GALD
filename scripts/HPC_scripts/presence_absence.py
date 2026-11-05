# 31/10-2026
#Overhauled: 4/11-2026


#######
# Quick intro: 
# This script is made to quickly evaluate the presence/absence of Ceruloplasmin (CP) 
# in the runs processed with the FragPipe Glyco-N workflow and the normal LFQ-MBR workflow. 
#######

# Description:
# Given that FragPipe's processing of CP and the samples in general might vary depending on 
# whether the N-glyco-LFQ or the LFQ-MBR (normal) workflow setting is used, we want to 
# analyse the presence/absence of CP in both workflows, and assess how much the different 
# workflows affect CP detection.
#
# Additionally, the script calculates the P-value based on Fisher's Exact Test 
# (CP present/absent in cohort 1 vs. CP present/absent in cohort 2).
#######

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
master_prot_run4_fragPipe_glyco_df, master_pep_run4_fragPipe_glyco_df = load_fragPipe_proteomic_data(ROOT_glyco_run4, annotation_file_run4)

print("\n[Glyco-N] Loading Run 5...")
ROOT_glyco_run5 = "/home/local/s215065/results/glyco_N_processed/run5_results_no_GALD14"
master_prot_run5_fragPipe_glyco_df, master_pep_run5_fragPipe_glyco_df = load_fragPipe_proteomic_data(ROOT_glyco_run5, annotation_file_run5)

print("\n[Glyco-N] Loading Run 6 (No Lysate Buffer)...")
ROOT_glyco_no_LB_run6 = "/home/local/s215065/results/glyco_N_processed/run6_results_no_GALD14_no_lysate_buffer/"
master_prot_run6_fragPipe_no_LB_glyco_df, master_pep_run6_fragPipe_no_LB_glyco_df = load_fragPipe_proteomic_data(ROOT_glyco_no_LB_run6, annotation_file_run6)

print("\n[Glyco-N] Loading Run 6 (With Lysate Buffer)...")
ROOT_glyco_with_LB_run6 = "/home/local/s215065/results/glyco_N_processed/run6_results_no_GALD14_including_lysate_buffer/"
master_prot_run6_fragPipe_with_LB_glyco_df, master_pep_run6_fragPipe_with_LB_glyco_df = load_fragPipe_proteomic_data(ROOT_glyco_with_LB_run6, annotation_file_run6)

print("\n[Glyco-N] Loading Run 6 (Sanity Check)...")
ROOT_glyco_run6_sanityCheck = "/home/local/s215065/results/glyco_N_processed/run6_sanityCheck_results"
master_prot_run6_fragPipe_sanityCheck_glyco_df, master_pep_run6_fragPipe_sanityCheck_glyco_df = load_fragPipe_proteomic_data(ROOT_glyco_run6_sanityCheck, annotation_file_run6)


# --- B. Loading data processed with LFQ-MBR (normal) workflow ---
print("\n=== Loading Normal (LFQ-MBR) Workflow Data ===")

print("\n[Normal LFQ-MBR] Loading Run 4...")
ROOT_normal_run4 = "/home/local/s215065/results/normal_processed_MBR_no_GALD14/run4_results"
master_prot_run4_fragPipe_normal_df, master_pep_run4_fragPipe_normal_df = load_fragPipe_proteomic_data(ROOT_normal_run4, annotation_file_run4)

print("\n[Normal LFQ-MBR] Loading Run 5...")
ROOT_normal_run5 = "/home/local/s215065/results/normal_processed_MBR_no_GALD14/run5_results"
master_prot_run5_fragPipe_normal_df, master_pep_run5_fragPipe_normal_df = load_fragPipe_proteomic_data(ROOT_normal_run5, annotation_file_run5)

print("\n[Normal LFQ-MBR] Loading Run 6 (No Lysate Buffer)...")
ROOT_normal_no_LB_run6 = "/home/local/s215065/results/normal_processed_MBR_no_GALD14/run6_no_lysate_buffer"
master_prot_run6_fragPipe_no_LB_normal_df, master_pep_run6_fragPipe_no_LB_normal_df = load_fragPipe_proteomic_data(ROOT_normal_no_LB_run6, annotation_file_run6)

print("\n[Normal LFQ-MBR] Loading Run 6 (With Lysate Buffer)...")
ROOT_normal_with_LB_run6 = "/home/local/s215065/results/normal_processed_MBR_no_GALD14/run6_with_lysate_buffer"
master_prot_run6_fragPipe_with_LB_normal_df, master_pep_run6_fragPipe_with_LB_normal_df = load_fragPipe_proteomic_data(ROOT_normal_with_LB_run6, annotation_file_run6)

print("\n[Normal LFQ-MBR] Loading Run 6 (Sanity Check)...")
ROOT_normal_run6_sanityCheck = "/home/local/s215065/results/normal_processed_MBR_no_GALD14/run6_sanityCheck_results"
master_prot_run6_fragPipe_sanityCheck_normal_df, master_pep_run6_fragPipe_sanityCheck_normal_df = load_fragPipe_proteomic_data(ROOT_normal_run6_sanityCheck, annotation_file_run6)


#######################
##  2. Filtering Out Irrelevant / Contaminated Samples
#######################
print("\n" + "="*60)
print("CHECKING & FILTERING CONTAMINATED SAMPLES AND LYSATE BUFFERS")
print("="*60)

def check_and_filter_df(df, name, df_type="Protein", remove_spouses=False):
    """
    Checks for unexpected samples (sample_14, sample_56, Lysate buffer, or Spouses in sanity check),
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

    # 3. Optional spouse removal for sanity check datasets
    if remove_spouses and 'Group' in df.columns:
        if 'GALD_spouse' in df['Group'].values:
            print(f"(!) WARNING [{name} - {df_type}]: Found 'GALD_spouse' in sanity check dataset! Filtering them out...")
        cond = cond & (df['Group'] != 'GALD_spouse')

    return df[cond].copy()

# Apply filtering to all Glyco-N DataFrames
master_prot_run4_fragPipe_glyco_df = check_and_filter_df(master_prot_run4_fragPipe_glyco_df, "Glyco Run 4", "Protein")
master_pep_run4_fragPipe_glyco_df  = check_and_filter_df(master_pep_run4_fragPipe_glyco_df,  "Glyco Run 4", "Peptide")

master_prot_run5_fragPipe_glyco_df = check_and_filter_df(master_prot_run5_fragPipe_glyco_df, "Glyco Run 5", "Protein")
master_pep_run5_fragPipe_glyco_df  = check_and_filter_df(master_pep_run5_fragPipe_glyco_df,  "Glyco Run 5", "Peptide")

master_prot_run6_fragPipe_no_LB_glyco_df = check_and_filter_df(master_prot_run6_fragPipe_no_LB_glyco_df, "Glyco Run 6 (No LB)", "Protein")
master_pep_run6_fragPipe_no_LB_glyco_df  = check_and_filter_df(master_pep_run6_fragPipe_no_LB_glyco_df,  "Glyco Run 6 (No LB)", "Peptide")

master_prot_run6_fragPipe_with_LB_glyco_df = check_and_filter_df(master_prot_run6_fragPipe_with_LB_glyco_df, "Glyco Run 6 (With LB)", "Protein")
master_pep_run6_fragPipe_with_LB_glyco_df  = check_and_filter_df(master_pep_run6_fragPipe_with_LB_glyco_df,  "Glyco Run 6 (With LB)", "Peptide")

master_prot_run6_fragPipe_sanityCheck_glyco_df = check_and_filter_df(master_prot_run6_fragPipe_sanityCheck_glyco_df, "Glyco Run 6 (Sanity Check)", "Protein", remove_spouses=True)
master_pep_run6_fragPipe_sanityCheck_glyco_df  = check_and_filter_df(master_pep_run6_fragPipe_sanityCheck_glyco_df,  "Glyco Run 6 (Sanity Check)", "Peptide", remove_spouses=True)

# Apply filtering to all Normal (LFQ-MBR) DataFrames
master_prot_run4_fragPipe_normal_df = check_and_filter_df(master_prot_run4_fragPipe_normal_df, "Normal Run 4", "Protein")
master_pep_run4_fragPipe_normal_df  = check_and_filter_df(master_pep_run4_fragPipe_normal_df,  "Normal Run 4", "Peptide")

master_prot_run5_fragPipe_normal_df = check_and_filter_df(master_prot_run5_fragPipe_normal_df, "Normal Run 5", "Protein")
master_pep_run5_fragPipe_normal_df  = check_and_filter_df(master_pep_run5_fragPipe_normal_df,  "Normal Run 5", "Peptide")

master_prot_run6_fragPipe_no_LB_normal_df = check_and_filter_df(master_prot_run6_fragPipe_no_LB_normal_df, "Normal Run 6 (No LB)", "Protein")
master_pep_run6_fragPipe_no_LB_normal_df  = check_and_filter_df(master_pep_run6_fragPipe_no_LB_normal_df,  "Normal Run 6 (No LB)", "Peptide")

master_prot_run6_fragPipe_with_LB_normal_df = check_and_filter_df(master_prot_run6_fragPipe_with_LB_normal_df, "Normal Run 6 (With LB)", "Protein")
master_pep_run6_fragPipe_with_LB_normal_df  = check_and_filter_df(master_pep_run6_fragPipe_with_LB_normal_df,  "Normal Run 6 (With LB)", "Peptide")

master_prot_run6_fragPipe_sanityCheck_normal_df = check_and_filter_df(master_prot_run6_fragPipe_sanityCheck_normal_df, "Normal Run 6 (Sanity Check)", "Protein", remove_spouses=True)
master_pep_run6_fragPipe_sanityCheck_normal_df  = check_and_filter_df(master_pep_run6_fragPipe_sanityCheck_normal_df,  "Normal Run 6 (Sanity Check)", "Peptide", remove_spouses=True)

print("\n✅ Successfully verified and filtered all DataFrames.")


#######################
##  3. Extracting Ceruloplasmin (CP) & Running Fisher's Exact Test
#######################

print("\n" + "="*60)
print("CERULOPLASMIN (CP) PRESENCE/ABSENCE & FISHER'S EXACT TEST")
print("="*60)

# Expected total cohort sizes (after removing sample_14, sample_56, and Lysate buffer)
EXPECTED_TOTALS = {
    'GALD': 14,
    'Healthy donor': 15,
    'GALD_spouse': 12
}

# Organize protein DataFrames and the comparisons we want to run for each
datasets_to_analyse = [
    # Glyco-N Runs
    ("Glyco-N | Run 4",                master_prot_run4_fragPipe_glyco_df,             [('GALD', 'Healthy donor')]),
    ("Glyco-N | Run 5",                master_prot_run5_fragPipe_glyco_df,             [('GALD', 'GALD_spouse')]),
    ("Glyco-N | Run 6 (No LB)",        master_prot_run6_fragPipe_no_LB_glyco_df,       [('GALD', 'Healthy donor'), ('GALD', 'GALD_spouse')]),
    ("Glyco-N | Run 6 (With LB)",      master_prot_run6_fragPipe_with_LB_glyco_df,     [('GALD', 'Healthy donor'), ('GALD', 'GALD_spouse')]),
    ("Glyco-N | Run 6 (Sanity Check)", master_prot_run6_fragPipe_sanityCheck_glyco_df, [('GALD', 'Healthy donor')]),
    
    # Normal LFQ-MBR Runs
    ("Normal LFQ-MBR | Run 4",                master_prot_run4_fragPipe_normal_df,             [('GALD', 'Healthy donor')]),
    ("Normal LFQ-MBR | Run 5",                master_prot_run5_fragPipe_normal_df,             [('GALD', 'GALD_spouse')]),
    ("Normal LFQ-MBR | Run 6 (No LB)",        master_prot_run6_fragPipe_no_LB_normal_df,       [('GALD', 'Healthy donor'), ('GALD', 'GALD_spouse')]),
    ("Normal LFQ-MBR | Run 6 (With LB)",      master_prot_run6_fragPipe_with_LB_normal_df,     [('GALD', 'Healthy donor'), ('GALD', 'GALD_spouse')]),
    ("Normal LFQ-MBR | Run 6 (Sanity Check)", master_prot_run6_fragPipe_sanityCheck_normal_df, [('GALD', 'Healthy donor')]),
]

cp_extracted_dfs = {}
summary_records = []

for label, prot_df, comparisons in datasets_to_analyse:
    print(f"\n--- {label} ---")
    
    # Extract CP rows
    cp_df = prot_df[prot_df.get('Gene', pd.Series(dtype=str)) == 'CP'].copy()
    cp_df.insert(0, 'Dataset', label)
    cp_extracted_dfs[label] = cp_df
    
    # Total unique samples present in the entire run per group (denominator)
    total_samples_in_run = prot_df.groupby('Group')['Sample'].nunique().to_dict()
    
    # Unique samples where CP was detected per group (numerator)
    cp_present_in_run = cp_df.groupby('Group')['Sample'].nunique().to_dict()
    
    for group1, group2 in comparisons:
        # Use total samples detected in the run; fallback to EXPECTED_TOTALS if a group had 0 total proteins
        tot_g1 = total_samples_in_run.get(group1, EXPECTED_TOTALS[group1])
        tot_g2 = total_samples_in_run.get(group2, EXPECTED_TOTALS[group2])
        
        # Sanity check if total samples in the dataframe differs from expected cohort sizes
        if tot_g1 != EXPECTED_TOTALS[group1]:
            print(f"  (!) Note: Total valid samples in run for {group1} is {tot_g1} (expected {EXPECTED_TOTALS[group1]})")
        if tot_g2 != EXPECTED_TOTALS[group2]:
            print(f"  (!) Note: Total valid samples in run for {group2} is {tot_g2} (expected {EXPECTED_TOTALS[group2]})")

        pres_g1 = cp_present_in_run.get(group1, 0)
        abs_g1  = tot_g1 - pres_g1
        
        pres_g2 = cp_present_in_run.get(group2, 0)
        abs_g2  = tot_g2 - pres_g2
        
        # 2x2 Contingency Table:
        # [[Present_G1, Absent_G1],
        #  [Present_G2, Absent_G2]]
        contingency_table = [
            [pres_g1, abs_g1],
            [pres_g2, abs_g2]
        ]
        
        odds_ratio, p_val = fisher_exact(contingency_table, alternative='two-sided')
        
        print(f"Comparison: {group1} vs. {group2}")
        print(f"  • {group1:<14}: {pres_g1:>2}/{tot_g1:<2} present ({abs_g1} absent)")
        print(f"  • {group2:<14}: {pres_g2:>2}/{tot_g2:<2} present ({abs_g2} absent)")
        print(f"  • Fisher's Exact Test P-value: {p_val:.4e} (Odds Ratio: {odds_ratio:.2f})")
        
        summary_records.append({
            'Dataset': label,
            'Comparison': f"{group1} vs {group2}",
            'Group_1': group1,
            'G1_Present': pres_g1,
            'G1_Absent': abs_g1,
            'G1_Total': tot_g1,
            'Group_2': group2,
            'G2_Present': pres_g2,
            'G2_Absent': abs_g2,
            'G2_Total': tot_g2,
            'Odds_Ratio': odds_ratio,
            'Fisher_P_value': p_val
        })

# Print a clean summary table at the very end
summary_df = pd.DataFrame(summary_records)
print("\n" + "="*60)
print("SUMMARY TABLE OF ALL COMPARISONS")
print("="*60)
print(summary_df[['Dataset', 'Comparison', 'G1_Present', 'G1_Total', 'G2_Present', 'G2_Total', 'Fisher_P_value']].to_string(index=False))

