import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from helper_functions.load_maxQuant_proteomic_data import load_maxQuant_proteomic_data, find_maxquant_txt_dir

# Global tracking for peptide counts
global_pep_counts_mq = {}
global_pep_counts_fp = {}

def extract_peptide_counts(melted_df, engine, run_label, gene_col='Gene'):
    """
    Extracts the highest unique/razor peptide count for each gene per run from the melted dataframe.
    """
    target_dict = global_pep_counts_mq if engine == 'mq' else global_pep_counts_fp
    
    # Hunt for the peptide count column preserved by the loader
    pep_cols = [c for c in melted_df.columns if 'peptide' in c.lower() and ('count' in c.lower() or 'razor' in c.lower() or 'total' in c.lower())]
    
    if not pep_cols:
        print(f"  [⚠️ Warning] No peptide count column found in {engine.upper()} ({run_label}).")
        return 
        
    pep_col = pep_cols[0] 
    
    # Since it's melted, multiple rows exist for the same gene. Group by Gene and get the max.
    grouped = melted_df.dropna(subset=[gene_col, pep_col]).groupby(gene_col)[pep_col].max()
    
    for gene, max_count in grouped.items():
        clean_gene = str(gene).split(';')[0].strip()
        if clean_gene not in target_dict: 
            target_dict[clean_gene] = {}
        target_dict[clean_gene][run_label] = int(max_count)



def get_maxquant_datasets(cache_dir, base_dir="~/DTU-GALD"):
    """Loads, filters, and caches MaxQuant datasets."""
    cache_file = cache_dir / "mq_master_data.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f: return pickle.load(f)

    run_specs = {
        "run4": {"req_id": "0876", "annot": "../experiment_annotation/experiment_annotation_run4_R876.tsv"},
        "run5": {"req_id": "1101", "annot": "../experiment_annotation/experiment_annotation_run5_R1101.tsv"},
        "run6": {"req_id": "1344", "annot": "../experiment_annotation/experiment_annotation_run6_R1344.tsv"}
    }
    
    mq_data = {}
    for key, spec in run_specs.items():
        txt_dir = find_maxquant_txt_dir(spec["req_id"], base_dir=base_dir)
        prot_df, _ = load_maxQuant_proteomic_data(ROOT=txt_dir, annotation_file=spec["annot"], load_peptides=False)
        extract_peptide_counts(prot_df, 'mq', key) 
        
        prot_df = prot_df[~prot_df['Sample'].isin(['sample_14', 'sample_56'])].copy()
        mq_data[f"{key}_with_LB"] = prot_df.copy()
        if 'Sample_type' in prot_df.columns:
            mq_data[f"{key}_no_LB"] = prot_df[prot_df['Sample_type'].astype(str).str.strip() == 'Liver Lysate'].copy()
        else:
            mq_data[f"{key}_no_LB"] = prot_df.copy()

    with open(cache_file, "wb") as f: pickle.dump(mq_data, f)
    return mq_data

def build_four_subsets(prot_r4, prot_r5, prot_r6):
    return {
        "R4_GALD_vs_Healthy": (prot_r4, "GALD", "Healthy donor"),
        "R5_GALD_vs_Spouse":  (prot_r5, "GALD", "GALD_spouse"),
        "R6_GALD_vs_Healthy": (prot_r6, "GALD", "Healthy donor"),
        "R6_GALD_vs_Spouse":  (prot_r6, "GALD", "GALD_spouse")
    }

def prep_wide_df(df, group_case, group_ctrl):
    quant_col = 'Razor Intensity' if 'Razor Intensity' in df.columns else ('MaxLFQ Intensity' if 'MaxLFQ Intensity' in df.columns else 'Intensity')
    df_clean = df.dropna(subset=['Gene']).copy()
    df_clean['Gene'] = df_clean['Gene'].astype(str).str.split(';').str[0].str.strip()
    
    s_to_g = df_clean.drop_duplicates('Sample').set_index('Sample')['Group'].astype(str).str.strip().to_dict()
    c_samps = [s for s, g in s_to_g.items() if g == group_case]
    k_samps = [s for s, g in s_to_g.items() if g == group_ctrl]
    
    wide_df = df_clean.pivot_table(index='Gene', columns='Sample', values=quant_col, aggfunc='sum')
    wide_df.replace(0, np.nan, inplace=True)
    target_samples = [s for s in (c_samps + k_samps) if s in wide_df.columns]
    
    return wide_df[target_samples].dropna(thresh=3, axis=0), [s for s in c_samps if s in target_samples], [s for s in k_samps if s in target_samples], s_to_g

def impute_perseus(wide_df):
    log2_df = np.log2(wide_df)
    np.random.seed(42)
    imputed = log2_df.copy()
    for col in imputed.columns:
        val = imputed[col].dropna()
        if len(val) > 2:
            m, s = val.mean(), val.std()
            mask = imputed[col].isna()
            imputed.loc[mask, col] = np.random.normal(loc=m - 1.8 * s, scale=s * 0.3, size=mask.sum())
    return imputed
