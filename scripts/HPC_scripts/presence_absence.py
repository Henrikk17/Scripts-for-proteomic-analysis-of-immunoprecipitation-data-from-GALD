# 09/11-2026
# Proteome-Wide Presence/Absence Fisher's Exact Test & Overlap Analysis

import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from helper_functions.manage_datasets import get_all_fragpipe_datasets

# 1. Load all cached datasets
datasets = get_all_fragpipe_datasets(use_cache=True, force_reload=False, load_peptides=False)

EXPECTED_TOTALS = {
    'GALD': 14,
    'Healthy donor': 15,
    'GALD_spouse': 12
}

# Structure: workflow -> "Enriched_GALD"/"Enriched_Control" -> gene -> list of subset tags
workflow_overlaps = {
    "Glyco-N": {"Enriched_GALD": {}, "Enriched_Control": {}},
    "LFQ-Comparable": {"Enriched_GALD": {}, "Enriched_Control": {}},
    "LFQ-IM-MS": {"Enriched_GALD": {}, "Enriched_Control": {}}
}

cp_records = []
all_sig_records = []

print("=" * 80)
print(" PROTEOME-WIDE PRESENCE/ABSENCE FISHER'S EXACT TEST")
print("=" * 80)

for key, data in datasets.items():
    label = data["label"]
    prot_df = data["prot"]
    comparisons = data["comparisons"]
    wf_name, run_part = label.split(" | ")

    print(f"\n--- {label} ---")

    # Ensure Gene column exists and remove NaN gene rows
    if 'Gene' not in prot_df.columns:
        print(f"  [-] Missing 'Gene' column in {label}. Skipping.")
        continue

    valid_prot_df = prot_df.dropna(subset=['Gene'])

    # Determine unique samples detected per group in this run (denominators)
    total_samples_in_run = valid_prot_df.groupby('Group')['Sample'].nunique().to_dict()

    # Pre-compute unique samples per protein and group (numerators)
    presence_matrix = (
        valid_prot_df.groupby(['Gene', 'Group'])['Sample']
        .nunique()
        .unstack(fill_value=0)
    )

    for group1, group2 in comparisons:
        tot_g1 = total_samples_in_run.get(group1, EXPECTED_TOTALS.get(group1, 0))
        tot_g2 = total_samples_in_run.get(group2, EXPECTED_TOTALS.get(group2, 0))

        if tot_g1 == 0 or tot_g2 == 0:
            print(f"  [-] Missing group samples ({group1}: {tot_g1}, {group2}: {tot_g2}). Skipping.")
            continue

        comp_genes = []
        p_values = []
        odds_ratios = []
        g1_pres_list = []
        g2_pres_list = []

        # Run 2x2 Fisher's exact test for every gene
        for gene, row in presence_matrix.iterrows():
            pres_g1 = row.get(group1, 0)
            abs_g1 = tot_g1 - pres_g1

            pres_g2 = row.get(group2, 0)
            abs_g2 = tot_g2 - pres_g2

            # Skip proteins absent in both groups
            if pres_g1 == 0 and pres_g2 == 0:
                continue

            contingency_table = [[pres_g1, abs_g1], [pres_g2, abs_g2]]
            odds_ratio, p_val = fisher_exact(contingency_table, alternative='two-sided')

            comp_genes.append(gene)
            p_values.append(p_val)
            odds_ratios.append(odds_ratio)
            g1_pres_list.append(pres_g1)
            g2_pres_list.append(pres_g2)

        if not comp_genes:
            continue

        res_df = pd.DataFrame({
            'Gene': comp_genes,
            'G1_Present': g1_pres_list,
            'G1_Total': tot_g1,
            'G2_Present': g2_pres_list,
            'G2_Total': tot_g2,
            'Odds_Ratio': odds_ratios,
            'p_value': p_values
        })

        # Benjamini-Hochberg FDR correction across all proteins in this comparison
        res_df['FDR'] = multipletests(res_df['p_value'], method='fdr_bh')[1]

        # Extract CP specifically
        cp_row = res_df[res_df['Gene'] == 'CP']
        if not cp_row.empty:
            cp_data = cp_row.iloc[0]
            cp_records.append({
                'Dataset': label,
                'Comparison': f"{group1} vs {group2}",
                'G1_Present': f"{cp_data['G1_Present']}/{tot_g1}",
                'G2_Present': f"{cp_data['G2_Present']}/{tot_g2}",
                'Odds_Ratio': cp_data['Odds_Ratio'],
                'p_value': cp_data['p_value'],
                'FDR': cp_data['FDR'],
                'Sig': 'YES' if cp_data['FDR'] < 0.05 else 'NO'
            })
            print(f"Comparison: {group1} vs. {group2}")
            print(f"  • CP Presence: {group1} ({cp_data['G1_Present']}/{tot_g1}) vs {group2} ({cp_data['G2_Present']}/{tot_g2})")
            print(f"  • CP Fisher P: {cp_data['p_value']:.4e} | FDR: {cp_data['FDR']:.4e} (Sig: {'YES' if cp_data['FDR'] < 0.05 else 'NO'})")
        else:
            print(f"Comparison: {group1} vs. {group2} -> CP not detected.")

        # Filter significant proteins (FDR < 0.05)
        sig_df = res_df[res_df['FDR'] < 0.05].copy()
        print(f"  • Total significant presence/absence proteins (FDR < 0.05): {len(sig_df)}")

        # Track overlap on distinct biological subsets (ignoring redundant "With LB")
        if "With LB" not in label:
            short_run = run_part.replace(" (No LB)", "").replace("Run ", "R")
            short_ctrl = "Healthy" if "Healthy" in group2 else "Spouse"
            subset_tag = f"{short_run}:{short_ctrl}"

            for _, srow in sig_df.iterrows():
                sgene = srow['Gene']
                # Determine direction: higher detection rate in GALD vs Control
                prop_g1 = srow['G1_Present'] / tot_g1
                prop_g2 = srow['G2_Present'] / tot_g2

                if prop_g1 > prop_g2:
                    workflow_overlaps[wf_name]["Enriched_GALD"].setdefault(sgene, []).append(subset_tag)
                else:
                    workflow_overlaps[wf_name]["Enriched_Control"].setdefault(sgene, []).append(subset_tag)

# --- 1. Ceruloplasmin (CP) Dedicated Summary Table ---
print("\n" + "=" * 90)
print(" CERULOPLASMIN (CP) PRESENCE/ABSENCE SUMMARY TABLE")
print("=" * 90)
cp_summary_df = pd.DataFrame(cp_records)
if not cp_summary_df.empty:
    cp_summary_df['Odds_Ratio'] = cp_summary_df['Odds_Ratio'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "NaN")
    cp_summary_df['p_value'] = cp_summary_df['p_value'].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "NaN")
    cp_summary_df['FDR'] = cp_summary_df['FDR'].map(lambda x: f"{x:.4e}" if pd.notnull(x) else "NaN")
    print(cp_summary_df[['Dataset', 'Comparison', 'G1_Present', 'G2_Present', 'Odds_Ratio', 'p_value', 'FDR', 'Sig']].to_string(index=False))
else:
    print("CP was not found in any dataset.")

# --- 2. Proteome-Wide Multi-Run Overlap Analysis ---
print("\n" + "=" * 90)
print(" OVERLAPPING DIFFERENTIALLY PRESENT PROTEINS (>= 2/4 Subsets, FDR < 0.05)")
print(" Subsets (4 total): R4:Healthy, R5:Spouse, R6:Healthy, R6:Spouse (No LB)")
print("=" * 90)

for wf_name, directions in workflow_overlaps.items():
    print(f"\n--- Workflow: {wf_name} ---")

    for direction, title in [("Enriched_GALD", "Higher presence in GALD"), 
                             ("Enriched_Control", "Higher presence in Control")]:
        overlapping = {g: subs for g, subs in directions[direction].items() if len(subs) >= 2}

        print(f"  {title} (in >= 2/4 subsets):")
        if overlapping:
            sorted_genes = sorted(overlapping.items(), key=lambda x: (-len(x[1]), x[0]))
            for gene, subs in sorted_genes:
                subs_str = ", ".join(subs)
                print(f"    - {gene:<12} ({len(subs)}/4 subsets) -> [{subs_str}]")
        else:
            print("    None")

print("\n" + "=" * 90)
