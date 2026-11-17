import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from helper_functions.load_maxQuant_proteomic_data import load_maxQuant_proteomic_data, find_maxquant_txt_dir
from helper_functions.load_fragPipe_proteomic_data import load_fragPipe_proteomic_data

def get_fragpipe_imms_datasets(cache_dir, base_dir="~/results/LFQ_MBR_IM-MS"):
    """Loads, filters, and caches ONLY the FragPipe IM-MS datasets, deriving no_LB from with_LB."""
    cache_file = cache_dir / "fp_imms_master_data.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f: return pickle.load(f)

    base_path = Path(base_dir).expanduser()
    
    # We completely drop "run6_no_LB" from the disk load
    run_specs = {
        "run4": {"path": base_path / "run4_results", "annot": "../experiment_annotation/experiment_annotation_run4_R876.tsv"},
        "run5": {"path": base_path / "run5_results", "annot": "../experiment_annotation/experiment_annotation_run5_R1101.tsv"},
        "run6": {"path": base_path / "run6_results_with_LB", "annot": "../experiment_annotation/experiment_annotation_run6_R1344.tsv"}
    }

    fp_data = {}
    for key, spec in run_specs.items():
        print(f"Loading FragPipe IM-MS: {key}...")
        prot_df, pep_df = load_fragPipe_proteomic_data(ROOT=spec["path"], annotation_file=spec["annot"])
        
        # Remove QC samples
        prot_df = prot_df[~prot_df['Sample'].isin(['sample_14', 'sample_56'])].copy()
        if pep_df is not None:
            pep_df = pep_df[~pep_df['Sample'].isin(['sample_14', 'sample_56'])].copy()
            
        if key == "run6":
            # Save the full run (With Buffer)
            fp_data["run6_with_LB"] = {"prot": prot_df.copy(), "pep": pep_df.copy() if pep_df is not None else None}
            
            # Dynamically derive the strictly filtered run (Liver Lysate only)
            if 'Sample_type' in prot_df.columns:
                fp_data["run6_no_LB"] = {
                    "prot": prot_df[prot_df['Sample_type'].astype(str).str.strip() == 'Liver Lysate'].copy(),
                    "pep": pep_df[pep_df['Sample_type'].astype(str).str.strip() == 'Liver Lysate'].copy() if pep_df is not None else None
                }
            else:
                fp_data["run6_no_LB"] = {"prot": prot_df.copy(), "pep": pep_df.copy() if pep_df is not None else None}
        else:
            fp_data[key] = {"prot": prot_df, "pep": pep_df}

    with open(cache_file, "wb") as f: pickle.dump(fp_data, f)
    return fp_data

def get_maxquant_datasets(cache_dir, base_dir="~/DTU-GALD"):
    """Loads, filters, and caches MaxQuant datasets, INCLUDING peptide data."""
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
        print(f"Loading MaxQuant: {key}...")
        txt_dir = find_maxquant_txt_dir(spec["req_id"], base_dir=base_dir)
        prot_df, pep_df = load_maxQuant_proteomic_data(ROOT=txt_dir, annotation_file=spec["annot"], load_peptides=True)
        
        prot_df = prot_df[~prot_df['Sample'].isin(['sample_14', 'sample_56'])].copy()
        if pep_df is not None:
            pep_df = pep_df[~pep_df['Sample'].isin(['sample_14', 'sample_56'])].copy()
            
        mq_data[f"{key}_with_LB"] = {"prot": prot_df.copy(), "pep": pep_df.copy() if pep_df is not None else None}
        
        if 'Sample_type' in prot_df.columns:
            mq_data[f"{key}_no_LB"] = {
                "prot": prot_df[prot_df['Sample_type'].astype(str).str.strip() == 'Liver Lysate'].copy(),
                "pep": pep_df[pep_df['Sample_type'].astype(str).str.strip() == 'Liver Lysate'].copy() if pep_df is not None else None
            }
        else:
            mq_data[f"{key}_no_LB"] = {"prot": prot_df.copy(), "pep": pep_df.copy() if pep_df is not None else None}

    with open(cache_file, "wb") as f: pickle.dump(mq_data, f)
    return mq_data

def get_gald_peptide_interval(pep_df, gene):
    """Calculates the min-max interval of unique peptides detected in GALD mothers."""
    if pep_df is None or pep_df.empty: return "-"
    
    subset = pep_df[(pep_df['Gene'] == gene) & (pep_df['Group'] == 'GALD')]
    if subset.empty: return "0"
    
    counts = subset.groupby('Sample')['Peptide Sequence'].nunique()
    if counts.empty: return "0"
    
    min_c, max_c = counts.min(), counts.max()
    return str(min_c) if min_c == max_c else f"{min_c}-{max_c}"

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