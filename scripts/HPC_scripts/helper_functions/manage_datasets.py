# helper_functions/manage_datasets.py

import os
import pandas as pd
from helper_functions.load_fragPipe_proteomic_data import load_fragPipe_proteomic_data #Note: We still neeed to use the relative path from the script running the code.

# Central configuration of all 12 datasets and their corrected paths
DATASET_CONFIG = {
    # --- A. Glyco-N Workflow ---
    "glyco_run4": {
        "label": "Glyco-N | Run 4",
        "root": "/home/local/s215065/results/glyco_N_processed/run4_results_no_GALD14",
        "annot": "../experiment_annotation/experiment_annotation_run4_R876.tsv",
        "comparisons": [('GALD', 'Healthy donor')]
    },
    "glyco_run5": {
        "label": "Glyco-N | Run 5",
        "root": "/home/local/s215065/results/glyco_N_processed/run5_results_no_GALD14",
        "annot": "../experiment_annotation/experiment_annotation_run5_R1101.tsv",
        "comparisons": [('GALD', 'GALD_spouse')]
    },
    "glyco_run6_no_LB": {
        "label": "Glyco-N | Run 6 (No LB)",
        "root": "/home/local/s215065/results/glyco_N_processed/run6_results_no_GALD14_no_lysate_buffer",
        "annot": "../experiment_annotation/experiment_annotation_run6_R1344.tsv",
        "comparisons": [('GALD', 'Healthy donor'), ('GALD', 'GALD_spouse')]
    },
    "glyco_run6_with_LB": {
        "label": "Glyco-N | Run 6 (With LB)",
        "root": "/home/local/s215065/results/glyco_N_processed/run6_results_no_GALD14_including_lysate_buffer",
        "annot": "../experiment_annotation/experiment_annotation_run6_R1344.tsv",
        "comparisons": [('GALD', 'Healthy donor'), ('GALD', 'GALD_spouse')]
    },

    # --- B. LFQ-MBR Comparable Workflow ---
    "comp_run4": {
        "label": "LFQ-Comparable | Run 4",
        "root": "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run4_results",
        "annot": "../experiment_annotation/experiment_annotation_run4_R876.tsv",
        "comparisons": [('GALD', 'Healthy donor')]
    },
    "comp_run5": {
        "label": "LFQ-Comparable | Run 5",
        "root": "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run5_results",
        "annot": "../experiment_annotation/experiment_annotation_run5_R1101.tsv",
        "comparisons": [('GALD', 'GALD_spouse')]
    },
    "comp_run6_no_LB": {
        "label": "LFQ-Comparable | Run 6 (No LB)",
        "root": "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run6_results_no_LB",
        "annot": "../experiment_annotation/experiment_annotation_run6_R1344.tsv",
        "comparisons": [('GALD', 'Healthy donor'), ('GALD', 'GALD_spouse')]
    },
    "comp_run6_with_LB": {
        "label": "LFQ-Comparable | Run 6 (With LB)",
        "root": "/home/local/s215065/results/LFQ_MBR_comparable_with_glyco_N/run6_results_with_LB",
        "annot": "../experiment_annotation/experiment_annotation_run6_R1344.tsv",
        "comparisons": [('GALD', 'Healthy donor'), ('GALD', 'GALD_spouse')]
    },

    # --- C. LFQ-MBR IM-MS Workflow ---
    "imms_run4": {
        "label": "LFQ-IM-MS | Run 4",
        "root": "/home/local/s215065/results/LFQ_MBR_IM-MS/run4_results",
        "annot": "../experiment_annotation/experiment_annotation_run4_R876.tsv",
        "comparisons": [('GALD', 'Healthy donor')]
    },
    "imms_run5": {
        "label": "LFQ-IM-MS | Run 5",
        "root": "/home/local/s215065/results/LFQ_MBR_IM-MS/run5_results",
        "annot": "../experiment_annotation/experiment_annotation_run5_R1101.tsv",
        "comparisons": [('GALD', 'GALD_spouse')]
    },
    "imms_run6_no_LB": {
        "label": "LFQ-IM-MS | Run 6 (No LB)",
        "root": "/home/local/s215065/results/LFQ_MBR_IM-MS/run6_results_no_LB",
        "annot": "../experiment_annotation/experiment_annotation_run6_R1344.tsv",
        "comparisons": [('GALD', 'Healthy donor'), ('GALD', 'GALD_spouse')]
    },
    "imms_run6_with_LB": {
        "label": "LFQ-IM-MS | Run 6 (With LB)",
        "root": "/home/local/s215065/results/LFQ_MBR_IM-MS/run6_results_with_LB",
        "annot": "../experiment_annotation/experiment_annotation_run6_R1344.tsv",
        "comparisons": [('GALD', 'Healthy donor'), ('GALD', 'GALD_spouse')]
    },
}


def check_and_filter_df(df, name, df_type="Protein"):
    """
    Checks for unexpected samples (sample_14, sample_56, Lysate buffer),
    alerts the user if found, and returns a clean filtered DataFrame.
    """
    unexpected_samples = {'sample_14', 'sample_56'}
    found_unexpected = set(df['Sample'].unique()).intersection(unexpected_samples)
    if found_unexpected:
        print(f"  (!) WARNING [{name} - {df_type}]: Found {found_unexpected}! Filtering out...")

    if 'Sample_type' in df.columns:
        non_liver = df[df['Sample_type'] != 'Liver Lysate']['Sample_type'].unique()
        if len(non_liver) > 0:
            print(f"  ℹ️  INFO [{name} - {df_type}]: Filtering out non-Liver Lysate {non_liver}...")

    cond = (~df['Sample'].isin(unexpected_samples))
    if 'Sample_type' in df.columns:
        cond = cond & (df['Sample_type'] == 'Liver Lysate')

    return df[cond].copy()


def get_all_fragpipe_datasets(use_cache=True, force_reload=False, load_peptides=False, cache_dir="../cached_dataframes"):
    """
    Loads and filters all 12 FragPipe datasets.

    Parameters:
    - use_cache (bool): If True, loads from/saves to fast .pkl files in cache_dir.
    - force_reload (bool): If True, ignores existing cache, reloads raw TSVs, and overwrites cache.
    - load_peptides (bool): If False, skips loading large peptide tables when only proteins are needed.
    - cache_dir (str): Directory to store intermediate pickle files.

    Returns:
    - datasets (dict): Dictionary mapping dataset key -> {'label', 'prot', 'pep', 'comparisons'}
    """
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)

    datasets = {}

    for key, cfg in DATASET_CONFIG.items():
        label = cfg["label"]
        prot_cache_path = os.path.join(cache_dir, f"{key}_prot.pkl")
        pep_cache_path  = os.path.join(cache_dir, f"{key}_pep.pkl")

        # Check if valid cache exists
        cache_exists = os.path.exists(prot_cache_path) and (not load_peptides or os.path.exists(pep_cache_path))

        if use_cache and cache_exists and not force_reload:
            print(f"⚡ [CACHE] Loading {label}...")
            prot_df = pd.read_pickle(prot_cache_path)
            pep_df  = pd.read_pickle(pep_cache_path) if load_peptides else None
        else:
            print(f"📂 [RAW TSV] Loading & filtering {label}...")
            raw_prot_df, raw_pep_df = load_fragPipe_proteomic_data(cfg["root"], cfg["annot"])

            prot_df = check_and_filter_df(raw_prot_df, label, "Protein")
            pep_df  = check_and_filter_df(raw_pep_df, label, "Peptide")

            if use_cache:
                prot_df.to_pickle(prot_cache_path)
                pep_df.to_pickle(pep_cache_path)

            if not load_peptides:
                pep_df = None

        datasets[key] = {
            "label": label,
            "prot": prot_df,
            "pep": pep_df,
            "comparisons": cfg["comparisons"]
        }

    print("✅ All datasets loaded and ready.\n")
    return datasets
