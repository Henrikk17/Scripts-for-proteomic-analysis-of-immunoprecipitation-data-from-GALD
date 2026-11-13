# 13/11-2026
# Broad Proteome-Wide Two-Way ANOVA & Paired Subtraction: Liver Lysate vs. Lysate Buffer (Run 6)

import os
import warnings
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind, ttest_rel
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from matplotlib.backends.backend_pdf import PdfPages

from helper_functions.manage_datasets import get_all_fragpipe_datasets
from helper_functions.load_fragPipe_proteomic_data import load_fragPipe_proteomic_data

os.makedirs("../plots/anova_run6", exist_ok=True)

# Top RF & candidate proteins to specifically highlight in the console report
KEY_CANDIDATES = [
    'UGP2', 'TAGLN2', 'TFRC', 'TKFC', 'TLN1', 'PKM', 'FKBP11', 'ACTR3', 'RSU1',
    'MTTP', 'DHRS7', 'RPL21', 'CP', 'HP', 'C4BPA', 'PROS1', 'VASP'
]


def load_run6_with_buffer():
    """
    Loads Run 6 LFQ-Comparable data WITH Lysate Buffer samples (Samples 1-84, excluding 14 & 56).
    """
    datasets = get_all_fragpipe_datasets(use_cache=True, load_peptides=False)
    if "comp_run6_with_LB" in datasets:
        prot_df = datasets["comp_run6_with_LB"]["prot"].copy()
        # Verify Buffer samples weren't stripped out by manage_datasets.py
        if (prot_df['Sample_type'].astype(str).str.strip() != 'Liver Lysate').any():
            print("⚡ Loaded Run 6 (With LB) from cache.")
            return prot_df

    print("📂 Loading raw Run 6 (With LB) directly from disk...")
    root_r6 = "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run6_results_with_LB"
    annot_r6 = "../experiment_annotation/experiment_annotation_run6_R1344.tsv"
    prot_df, _ = load_fragPipe_proteomic_data(root_r6, annot_r6)
    prot_df = prot_df[~prot_df['Sample'].isin(['sample_14', 'sample_56'])].copy()
    return prot_df


def run_proteome_wide_anova(prot_df):
    """
    Performs Perseus imputation on the 82 paired Run 6 samples, then runs:
      1. Two-Way ANOVA: Intensity ~ C(Matrix) * C(Cohort)
      2. Paired Difference Analysis: Delta_j = Liver_j - Buffer_j (GALD vs Control)
    """
    quant_col = 'Razor Intensity' if 'Razor Intensity' in prot_df.columns else (
        'MaxLFQ Intensity' if 'MaxLFQ Intensity' in prot_df.columns else 'Intensity'
    )

    df = prot_df.dropna(subset=['Gene']).copy()
    df['Sample_Num'] = df['Sample'].str.replace('sample_', '', regex=False).astype(int)

    # Map Subject_ID so Sample k (1..42, Liver) pairs with Sample k+42 (43..84, Buffer)
    df['Subject_ID'] = df['Sample_Num'].apply(lambda n: n if n <= 42 else n - 42)
    df['Matrix'] = df['Sample_Num'].apply(lambda n: 'Liver' if n <= 42 else 'Buffer')

    # Clean Cohort naming (GALD, GALD_spouse, Healthy donor)
    sample_meta = df[['Sample', 'Subject_ID', 'Matrix', 'Group']].drop_duplicates().set_index('Sample')

    # Pivot to [Gene x Sample]
    wide_df = df.pivot_table(index='Gene', columns='Sample', values=quant_col, aggfunc='sum')
    wide_df.replace(0, np.nan, inplace=True)

    # Keep proteins detected in at least 5 samples overall
    wide_df = wide_df.dropna(thresh=5, axis=0)
    log2_df = np.log2(wide_df)

    # Record raw detection counts in Liver vs Buffer BEFORE imputation
    liver_cols = [c for c in log2_df.columns if sample_meta.loc[c, 'Matrix'] == 'Liver']
    buffer_cols = [c for c in log2_df.columns if sample_meta.loc[c, 'Matrix'] == 'Buffer']

    n_det_liver = log2_df[liver_cols].notna().sum(axis=1)
    n_det_buffer = log2_df[buffer_cols].notna().sum(axis=1)
    raw_mean_liver = log2_df[liver_cols].mean(axis=1)
    raw_mean_buffer = log2_df[buffer_cols].mean(axis=1)

    # Deterministic Perseus Imputation per sample
    np.random.seed(42)
    imputed_df = log2_df.copy()
    for col in imputed_df.columns:
        valid = imputed_df[col].dropna()
        if len(valid) > 2:
            m, s = valid.mean(), valid.std()
            na_mask = imputed_df[col].isna()
            imputed_df.loc[na_mask, col] = np.random.normal(loc=m - 1.8 * s, scale=s * 0.3, size=na_mask.sum())

    print(f"🔬 Running Two-Way ANOVA & Paired Subtraction across {len(imputed_df)} proteins...")

    records = []
    warnings.filterwarnings('ignore')

    for gene, row in imputed_df.iterrows():
        sub_df = pd.DataFrame({
            'Intensity': row.values,
            'Matrix': [sample_meta.loc[c, 'Matrix'] for c in imputed_df.columns],
            'Cohort': [str(sample_meta.loc[c, 'Group']).strip() for c in imputed_df.columns],
            'Subject_ID': [sample_meta.loc[c, 'Subject_ID'] for c in imputed_df.columns]
        })

        # 1. Two-Way ANOVA: Intensity ~ Matrix + Cohort + Matrix:Cohort
        try:
            model = ols('Intensity ~ C(Matrix) * C(Cohort)', data=sub_df).fit()
            anova_tbl = sm.stats.anova_lm(model, typ=2)
            p_matrix = anova_tbl.loc['C(Matrix)', 'PR(>F)']
            p_cohort = anova_tbl.loc['C(Cohort)', 'PR(>F)']
            p_inter = anova_tbl.loc['C(Matrix):C(Cohort)', 'PR(>F)']
        except Exception:
            p_matrix, p_cohort, p_inter = np.nan, np.nan, np.nan

        # 2. Paired Delta Analysis: Delta_j = Intensity(Liver, Subject_j) - Intensity(Buffer, Subject_j)
        piv_subj = sub_df.pivot(index='Subject_ID', columns='Matrix', values='Intensity')
        subj_cohort = sub_df.drop_duplicates('Subject_ID').set_index('Subject_ID')['Cohort']
        piv_subj['Cohort'] = subj_cohort
        piv_subj['Delta_Liver_minus_Buffer'] = piv_subj['Liver'] - piv_subj['Buffer']

        mean_l = sub_df[sub_df['Matrix'] == 'Liver']['Intensity'].mean()
        mean_b = sub_df[sub_df['Matrix'] == 'Buffer']['Intensity'].mean()
        log2fc_matrix = mean_l - mean_b

        gald_delta = piv_subj[piv_subj['Cohort'] == 'GALD']['Delta_Liver_minus_Buffer'].dropna()
        ctrl_delta = piv_subj[piv_subj['Cohort'] != 'GALD']['Delta_Liver_minus_Buffer'].dropna()

        delta_diff_gald_vs_ctrl = gald_delta.mean() - ctrl_delta.mean()
        _, p_delta_gald = ttest_ind(gald_delta, ctrl_delta, equal_var=False)

        # Also check GALD vs Control in Liver alone vs Buffer alone
        gald_l = piv_subj[piv_subj['Cohort'] == 'GALD']['Liver'].dropna()
        ctrl_l = piv_subj[piv_subj['Cohort'] != 'GALD']['Liver'].dropna()
        fc_gald_liver = gald_l.mean() - ctrl_l.mean()

        gald_b = piv_subj[piv_subj['Cohort'] == 'GALD']['Buffer'].dropna()
        ctrl_b = piv_subj[piv_subj['Cohort'] != 'GALD']['Buffer'].dropna()
        fc_gald_buffer = gald_b.mean() - ctrl_b.mean()

        records.append({
            'Gene': gene,
            'Det_Liver (max 41)': int(n_det_liver[gene]),
            'Det_Buffer (max 41)': int(n_det_buffer[gene]),
            'Raw_Mean_Liver': raw_mean_liver[gene],
            'Raw_Mean_Buffer': raw_mean_buffer[gene],
            'Log2FC_Liver_vs_Buffer': log2fc_matrix,
            'FC_GALD_vs_Ctrl (Liver)': fc_gald_liver,
            'FC_GALD_vs_Ctrl (Buffer)': fc_gald_buffer,
            'Paired_Delta_GALD_vs_Ctrl': delta_diff_gald_vs_ctrl,
            'P_Matrix': p_matrix,
            'P_Cohort': p_cohort,
            'P_Interaction': p_inter,
            'P_Paired_Delta': p_delta_gald
        })

    res_df = pd.DataFrame(records).dropna(subset=['P_Matrix', 'P_Cohort', 'P_Interaction'])

    # Multiple testing correction (Benjamini-Hochberg FDR)
    res_df['FDR_Matrix'] = multipletests(res_df['P_Matrix'], method='fdr_bh')[1]
    res_df['FDR_Cohort'] = multipletests(res_df['P_Cohort'], method='fdr_bh')[1]
    res_df['FDR_Interaction'] = multipletests(res_df['P_Interaction'], method='fdr_bh')[1]
    res_df['FDR_Paired_Delta'] = multipletests(res_df['P_Paired_Delta'], method='fdr_bh')[1]

    # Classify every protein into biological origin buckets
    def classify_protein(r):
        # 1. True Fetal Liver + GALD Enriched (Significant liver enrichment AND GALD pulls down more than Control after buffer subtraction)
        if (r['FDR_Matrix'] < 0.05 and r['Log2FC_Liver_vs_Buffer'] > 0.5) and (
            r['P_Paired_Delta'] < 0.05 and r['Paired_Delta_GALD_vs_Ctrl'] > 0.5
        ):
            return "1. True Fetal Liver Target (GALD-Enriched)"

        # 2. Maternal Plasma Carryover (Identical/Similar in Buffer & Liver, or higher in Buffer)
        if r['FDR_Matrix'] >= 0.05 or abs(r['Log2FC_Liver_vs_Buffer']) <= 0.5 or r['Log2FC_Liver_vs_Buffer'] < -0.5:
            if r['FDR_Cohort'] < 0.05:
                return "2A. Maternal Plasma Carryover (Cohort-Specific)"
            return "2B. Maternal Plasma / Bead Background (Identical Across All)"

        # 3. Non-Specific Fetal Liver Background (Strongly elevated in Liver, but identical between GALD & Controls)
        return "3. Non-Specific Fetal Liver Background"

    res_df['Classification'] = res_df.apply(classify_protein, axis=1)
    return res_df


def generate_plots_and_report(res_df):
    csv_out = "../plots/anova_run6/Run6_ANOVA_Liver_vs_Buffer_Full.csv"
    res_df.sort_values(by=['Classification', 'P_Paired_Delta']).to_csv(csv_out, index=False)

    print("\n" + "="*85)
    print("📊 PROTEOME-WIDE CLASSIFICATION SUMMARY (RUN 6: LIVER LYSATE vs. LYSATE BUFFER)")
    print("="*85)
    counts = res_df['Classification'].value_counts()
    for cat, cnt in counts.items():
        print(f"  • {cat:<58}: {cnt:4d} proteins ({cnt/len(res_df)*100:5.1f}%)")

    print("\n" + "="*85)
    print("🎯 DIAGNOSTIC TABLE FOR TOP RANDOM FOREST & CANDIDATE PROTEINS")
    print("="*85)
    sub_cand = res_df[res_df['Gene'].isin(KEY_CANDIDATES)].copy()
    sub_cand = sub_cand.sort_values(by='Log2FC_Liver_vs_Buffer', ascending=False)

    cols_to_show = [
        'Gene', 'Det_Liver (max 41)', 'Det_Buffer (max 41)',
        'Log2FC_Liver_vs_Buffer', 'FC_GALD_vs_Ctrl (Liver)',
        'FC_GALD_vs_Ctrl (Buffer)', 'Paired_Delta_GALD_vs_Ctrl',
        'P_Paired_Delta', 'Classification'
    ]
    print(sub_cand[cols_to_show].to_string(index=False))

    # Top 15 True Fetal Liver Targets (Sorted by Paired Delta P-value)
    true_fetal = res_df[res_df['Classification'] == "1. True Fetal Liver Target (GALD-Enriched)"].sort_values(
        by='P_Paired_Delta', ascending=True
    ).head(15)

    print("\n" + "="*85)
    print("🏆 TOP 15 TRUE FETAL LIVER TARGETS (SIGNIFICANT AFTERPAIRED BUFFER SUBTRACTION)")
    print("="*85)
    if not true_fetal.empty:
        print(true_fetal[cols_to_show].to_string(index=False))
    else:
        print("  None crossed strict thresholds.")

    # Save Diagnostic Plots to PDF
    pdf_path = "../plots/anova_run6/Run6_ANOVA_Liver_vs_Buffer_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Plot 1: Liver vs Buffer Fold Change vs Paired GALD Delta
        fig, ax = plt.subplots(figsize=(11, 8))
        palette = {
            "1. True Fetal Liver Target (GALD-Enriched)": "firebrick",
            "2A. Maternal Plasma Carryover (Cohort-Specific)": "darkorange",
            "2B. Maternal Plasma / Bead Background (Identical Across All)": "grey",
            "3. Non-Specific Fetal Liver Background": "steelblue"
        }

        sns.scatterplot(
            data=res_df,
            x='Log2FC_Liver_vs_Buffer',
            y='Paired_Delta_GALD_vs_Ctrl',
            hue='Classification',
            palette=palette,
            alpha=0.65,
            s=40,
            ax=ax
        )

        # Annotate Key Candidates
        for _, r in sub_cand.iterrows():
            ax.scatter(r['Log2FC_Liver_vs_Buffer'], r['Paired_Delta_GALD_vs_Ctrl'],
                       color='gold', edgecolor='black', s=140, zorder=5)
            ax.annotate(r['Gene'], (r['Log2FC_Liver_vs_Buffer'], r['Paired_Delta_GALD_vs_Ctrl']),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold', fontsize=9)

        ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Liver Enrichment Cutoff (+0.5)')
        ax.axvline(-0.5, color='black', linestyle='--', alpha=0.5)
        ax.axhline(0.5, color='firebrick', linestyle='--', alpha=0.5, label='GALD Paired Delta Cutoff (+0.5)')
        ax.axhline(0.0, color='grey', linestyle=':', alpha=0.5)

        ax.set_title("Run 6 Two-Way Dissection: Fetal Liver Origin vs. GALD-Specific Capture", fontweight='bold')
        ax.set_xlabel("Matrix Effect: Log2FC (Liver Lysate - Lysate Buffer)\n[< 0.5 = Maternal Plasma Carryover | > 0.5 = Derived from Fetal Liver]")
        ax.set_ylabel("Paired Background-Subtracted GALD Effect: Δ(Liver - Buffer) [GALD vs Control]\n[> 0.5 = GALD Captures More Fetal Protein Than Control]")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Plot 2: Direct Comparison of GALD Fold Change in Liver vs. GALD Fold Change in Buffer
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            data=res_df,
            x='FC_GALD_vs_Ctrl (Buffer)',
            y='FC_GALD_vs_Ctrl (Liver)',
            hue='Classification',
            palette=palette,
            alpha=0.65,
            s=40,
            ax=ax
        )
        # Identity line y = x (If a protein lies on this line, its GALD vs Control difference in Liver is 100% driven by Buffer!)
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.6, label='Identity Line (Liver FC == Buffer FC)')

        for _, r in sub_cand.iterrows():
            ax.scatter(r['FC_GALD_vs_Ctrl (Buffer)'], r['FC_GALD_vs_Ctrl (Liver)'],
                       color='gold', edgecolor='black', s=140, zorder=5)
            ax.annotate(r['Gene'], (r['FC_GALD_vs_Ctrl (Buffer)'], r['FC_GALD_vs_Ctrl (Liver)']),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold', fontsize=9)

        ax.set_title("Carryover Diagnostic: GALD vs. Control Fold-Change in Buffer vs. Liver", fontweight='bold')
        ax.set_xlabel("Log2FC (GALD vs. Control) in LYSATE BUFFER ONLY")
        ax.set_ylabel("Log2FC (GALD vs. Control) in LIVER LYSATE")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print("\n" + "="*85)
    print(f"✅ Full ANOVA table exported to: {csv_out}")
    print(f"✅ Diagnostic plots compiled to: {pdf_path}")
    print("="*85)


if __name__ == "__main__":
    prot_r6 = load_run6_with_buffer()
    anova_results = run_proteome_wide_anova(prot_r6)
    generate_plots_and_report(anova_results)