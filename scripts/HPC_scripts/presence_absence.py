# 09/11-2026
# Compact CP Presence/Absence & Fisher's Exact Test Script

import os
import pandas as pd
from scipy.stats import fisher_exact
from helper_functions.manage_datasets import get_all_fragpipe_datasets

# 1. Load all 12 filtered datasets (creates cache on first run; loads in <1s on subsequent runs)
# Set use_cache=False or force_reload=True anytime you want to re-read the raw TSVs
datasets = get_all_fragpipe_datasets(use_cache=True, force_reload=False, load_peptides=False)

EXPECTED_TOTALS = {
    'GALD': 14,
    'Healthy donor': 15,
    'GALD_spouse': 12
}

print("=" * 60)
print("CERULOPLASMIN (CP) PRESENCE/ABSENCE & FISHER'S EXACT TEST")
print("=" * 60)

summary_records = []

for key, data in datasets.items():
    label       = data["label"]
    prot_df     = data["prot"]
    comparisons = data["comparisons"]

    print(f"\n--- {label} ---")

    # Extract CP rows
    cp_df = prot_df[prot_df.get('Gene', pd.Series(dtype=str)) == 'CP'].copy()

    # Denominator: Total unique samples present in the run per group
    total_samples_in_run = prot_df.groupby('Group')['Sample'].nunique().to_dict()

    # Numerator: Unique samples where CP was detected per group
    cp_present_in_run = cp_df.groupby('Group')['Sample'].nunique().to_dict()

    for group1, group2 in comparisons:
        tot_g1 = total_samples_in_run.get(group1, EXPECTED_TOTALS.get(group1, 0))
        tot_g2 = total_samples_in_run.get(group2, EXPECTED_TOTALS.get(group2, 0))

        pres_g1 = cp_present_in_run.get(group1, 0)
        abs_g1  = tot_g1 - pres_g1

        pres_g2 = cp_present_in_run.get(group2, 0)
        abs_g2  = tot_g2 - pres_g2

        odds_ratio, p_val = fisher_exact([[pres_g1, abs_g1], [pres_g2, abs_g2]], alternative='two-sided')

        print(f"Comparison: {group1} vs. {group2}")
        print(f"  • {group1:<14}: {pres_g1:>2}/{tot_g1:<2} present ({abs_g1} absent)")
        print(f"  • {group2:<14}: {pres_g2:>2}/{tot_g2:<2} present ({abs_g2} absent)")
        print(f"  • Fisher's Exact Test P-value: {p_val:.4e} (Odds Ratio: {odds_ratio:.2f})")

        summary_records.append({
            'Dataset': label,
            'Comparison': f"{group1} vs {group2}",
            'G1_Present': pres_g1,
            'G1_Total': tot_g1,
            'G2_Present': pres_g2,
            'G2_Total': tot_g2,
            'Odds_Ratio': odds_ratio,
            'Fisher_P_value': p_val
        })

summary_df = pd.DataFrame(summary_records)
print("\n" + "=" * 60)
print("SUMMARY TABLE OF ALL COMPARISONS")
print("=" * 60)
print(summary_df[['Dataset', 'Comparison', 'G1_Present', 'G1_Total', 'G2_Present', 'G2_Total', 'Fisher_P_value']].to_string(index=False))
