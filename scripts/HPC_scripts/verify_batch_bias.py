# 09/11-2026
# GALD IgG Purification Batch Bias & Origin Verification (Run 6)

import os
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from matplotlib.backends.backend_pdf import PdfPages

from helper_functions.load_fragPipe_proteomic_data import load_fragPipe_proteomic_data

os.makedirs("../plots", exist_ok=True)

# 1. Define Sample Sets based on 'Calculations for IP - RUN 6.xlsx'
new_prep_ll = [f"sample_{i}" for i in [1, 2, 3, 4, 5, 8, 12]]
old_prep_ll = [f"sample_{i}" for i in [6, 7, 9, 10, 11, 13, 15]] # Excluded sample_14 (KON QC)

new_prep_lb = [f"sample_{i+42}" for i in [1, 2, 3, 4, 5, 8, 12]] # 43, 44, 45, 46, 47, 50, 54
old_prep_lb = [f"sample_{i+42}" for i in [6, 7, 9, 10, 11, 13, 15]] # 48, 49, 51, 52, 53, 55, 57

healthy_ll = [f"sample_{i}" for i in range(28, 43)] # 28 to 42

def map_batch_cohort(s):
    if s in new_prep_ll: return "GALD (New Prep) + Liver"
    if s in old_prep_ll: return "GALD (Old Prep) + Liver"
    if s in new_prep_lb: return "GALD (New Prep) + Buffer"
    if s in old_prep_lb: return "GALD (Old Prep) + Buffer"
    if s in healthy_ll: return "Healthy Donor + Liver"
    return "Other"

# 2. Load Run 6 LFQ-Comparable Data (with LB samples)
print("Loading Run 6 (With LB) Data...")
root_r6 = "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run6_results_with_LB"
annot_r6 = "../experiment_annotation/experiment_annotation_run6_R1344.tsv"

prot_df, _ = load_fragPipe_proteomic_data(root_r6, annot_r6)

# Determine intensity column
quant_col = 'Razor Intensity' if 'Razor Intensity' in prot_df.columns else 'Intensity'
if 'MaxLFQ Intensity' in prot_df.columns: quant_col = 'MaxLFQ Intensity'

# 3. Filter and Prepare Data
prot_df = prot_df.dropna(subset=['Gene'])
prot_df['Batch_Cohort'] = prot_df['Sample'].apply(map_batch_cohort)
plot_df = prot_df[prot_df['Batch_Cohort'] != "Other"].copy()

# Calculate Log2 Intensity (treating 0 as NaN to ignore missing values in the mean)
plot_df[quant_col] = pd.to_numeric(plot_df[quant_col], errors='coerce')
plot_df['Log2_Intensity'] = np.log2(plot_df[quant_col].replace(0, np.nan))

TARGETS = ['CP', 'HP', 'C4BPA', 'PROS1', 'VASP', 'TAGLN2']

# 4. Statistical Testing & Output
print("\n" + "="*80)
print(" STATISTICAL CHECK: IgG BATCH BIAS & PROTEIN ORIGIN (Run 6)")
print("="*80)

summary_records = []
for gene in TARGETS:
    gene_df = plot_df[plot_df['Gene'] == gene]
    if gene_df.empty:
        continue
        
    def get_vals(cohort_name):
        return gene_df[gene_df['Batch_Cohort'] == cohort_name]['Log2_Intensity'].dropna().values
        
    n_ll = get_vals("GALD (New Prep) + Liver")
    o_ll = get_vals("GALD (Old Prep) + Liver")
    n_lb = get_vals("GALD (New Prep) + Buffer")
    o_lb = get_vals("GALD (Old Prep) + Buffer")
    
    # Test 1: Batch Bias (New vs Old Prep in Liver Lysate)
    p_batch_ll = ttest_ind(n_ll, o_ll, equal_var=False)[1] if len(n_ll) > 1 and len(o_ll) > 1 else np.nan
    # Test 2: Protein Origin (Liver Lysate vs Buffer Only in New Prep)
    p_origin_n = ttest_ind(n_ll, n_lb, equal_var=False)[1] if len(n_ll) > 1 and len(n_lb) > 1 else np.nan
    # Test 3: Protein Origin (Liver Lysate vs Buffer Only in Old Prep)
    p_origin_o = ttest_ind(o_ll, o_lb, equal_var=False)[1] if len(o_ll) > 1 and len(o_lb) > 1 else np.nan

    print(f"\n--- {gene} ---")
    print(f"  [Batch Bias] New vs Old Prep (Liver): P-value = {p_batch_ll:.4f}")
    if p_batch_ll < 0.05:
        print("     ⚠️ WARNING: Significant difference driven entirely by IgG purification date.")
        
    print(f"  [Origin Check - New] Liver vs Buffer: P-value = {p_origin_n:.4f}")
    if p_origin_n >= 0.05 or np.isnan(p_origin_n):
        print("     ⚠️ WARNING: Protein signal in Liver IP is indistinguishable from Buffer (Maternal Plasma Origin).")
        
    print(f"  [Origin Check - Old] Liver vs Buffer: P-value = {p_origin_o:.4f}")

# 5. Plotting Boxplots
pdf_path = "../plots/Verify_IgG_Batch_Bias.pdf"
cohort_order = [
    "Healthy Donor + Liver",
    "GALD (New Prep) + Liver",
    "GALD (Old Prep) + Liver",
    "GALD (New Prep) + Buffer",
    "GALD (Old Prep) + Buffer"
]

custom_palette = {
    "Healthy Donor + Liver": "steelblue",
    "GALD (New Prep) + Liver": "firebrick",
    "GALD (Old Prep) + Liver": "lightcoral",
    "GALD (New Prep) + Buffer": "darkorange",
    "GALD (Old Prep) + Buffer": "navajowhite"
}

with PdfPages(pdf_path) as pdf:
    for gene in TARGETS:
        gene_df = plot_df[plot_df['Gene'] == gene]
        if gene_df.empty:
            continue
            
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Batch_Cohort', y='Log2_Intensity', data=gene_df, order=cohort_order, palette=custom_palette, showfliers=False, width=0.6)
        sns.stripplot(x='Batch_Cohort', y='Log2_Intensity', data=gene_df, order=cohort_order, color='black', alpha=0.6, jitter=True, size=6)
        
        plt.title(f"{gene} Abundance Across Run 6 Isolation Subsets", fontweight='bold', fontsize=14)
        plt.ylabel("Log2 Intensity")
        plt.xlabel("")
        plt.xticks(rotation=30, ha='right', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        pdf.savefig(bbox_inches='tight')
        plt.close()

print("\n" + "="*80)
print(f"✅ Boxplots comparing subsets saved to {pdf_path}")
