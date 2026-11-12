# 12/11-2026
# HPC-Ready Function for loading MaxQuant output data

import re
import warnings
from pathlib import Path
import pandas as pd
import numpy as np


def find_maxquant_txt_dir(run_id_num, base_dir="~/DTU-GALD"):
    """
    Automatically locates the MaxQuant 'txt' directory containing proteinGroups.txt
    for a given request number (e.g., '0876' or '876', '1101', '1344') under ~/DTU-GALD.
    """
    base_path = Path(base_dir).expanduser().resolve()
    padded_id = f"R{int(str(run_id_num).lstrip('R')):04d}"   # e.g. R0876, R1101, R1344
    unpadded_id = f"R{int(str(run_id_num).lstrip('R'))}"     # e.g. R876

    # Check candidate root folders (e.g., unzipped_R0876, unzipped_R876, R0876)
    candidates = (
        list(base_path.glob(f"*{padded_id}*")) +
        list(base_path.glob(f"*{unpadded_id}*"))
    )

    for cand in candidates:
        if cand.is_dir():
            matches = list(cand.rglob("proteinGroups.txt"))
            if matches:
                return matches[0].parent

    raise FileNotFoundError(
        f"❌ Could not find 'proteinGroups.txt' for {padded_id} inside {base_path}"
    )


def _extract_sample_id(raw_col_series):
    """
    Robustly extracts 'sample_<N>' from MaxQuant column headers like:
      'LFQ intensity 1_S3-A1_1_9338' -> 'sample_1'
      'LFQ intensity S1_...'         -> 'sample_1'
      'LFQ intensity Sample_14_...'  -> 'sample_14'
    """
    cleaned = (
        raw_col_series
        .str.replace('LFQ intensity ', '', regex=False)
        .str.replace('Intensity ', '', regex=False)
        .str.strip()
    )
    first_token = cleaned.str.split('_', n=1).str[0]
    sample_nums = first_token.str.extract(r'(\d+)', expand=False)
    return 'sample_' + sample_nums.astype(int).astype(str)


def load_maxQuant_proteomic_data(ROOT, annotation_file, load_peptides=True):
    """
    Loads MaxQuant proteinGroups.txt (and optionally peptides.txt) on the HPC.
    ROOT can be the path to the 'txt' folder (supports '~') or auto-discovered.
    """
    annot_path = Path(annotation_file).expanduser().resolve()
    annot_df = pd.read_csv(annot_path, sep='\t')

    sample_to_group = dict(zip(annot_df['sample'].astype(str).str.strip(), annot_df['condition'].astype(str).str.strip()))
    sample_to_type = dict(zip(annot_df['sample'].astype(str).str.strip(), annot_df['sample_type'].astype(str).str.strip()))

    print(f"Loaded annotations: Found {len(sample_to_group)} total samples in {annot_path.name}.")

    # ==========================================
    # 1. SETUP PATHS
    # ==========================================
    root_path = Path(ROOT).expanduser().resolve()
    if root_path.name == "proteinGroups.txt":
        root_path = root_path.parent
    elif not (root_path / "proteinGroups.txt").exists():
        # If user passed a parent directory (e.g., ~/DTU-GALD/unzipped_R0876), search inside it
        found = list(root_path.rglob("proteinGroups.txt"))
        if found:
            root_path = found[0].parent
        else:
            raise FileNotFoundError(f"❌ 'proteinGroups.txt' not found under {root_path}")

    prot_file = root_path / "proteinGroups.txt"
    pep_file = root_path / "peptides.txt"

    # ==========================================
    # 2. LOAD & MELT PROTEIN DATA
    # ==========================================
    print(f"Processing {prot_file}...")

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
        prot_df = pd.read_csv(prot_file, sep='\t', low_memory=False)

    # --- Contaminant & Decoy Filtering ---
    if 'Potential contaminant' in prot_df.columns:
        prot_df = prot_df[prot_df['Potential contaminant'] != '+']
    if 'Reverse' in prot_df.columns:
        prot_df = prot_df[prot_df['Reverse'] != '+']
    if 'Only identified by site' in prot_df.columns:
        prot_df = prot_df[prot_df['Only identified by site'] != '+']

    # --- Rename columns ---
    prot_df.rename(columns={
        'Majority protein IDs': 'Protein',
        'Protein IDs': 'Protein Group',
        'Fasta headers': 'Description',
        'Gene names': 'Gene',
        'Protein names': 'Protein Name'
    }, inplace=True, errors='ignore')

    # Clean multi-mapped MaxQuant gene names (e.g., "TAGLN2;TAGLN" -> "TAGLN2")
    if 'Gene' in prot_df.columns:
        prot_df['Gene'] = (
            prot_df['Gene']
            .astype(str)
            .str.split(';')
            .str[0]
            .str.strip()
            .replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        )

    # --- Dynamically grab MaxQuant LFQ Columns ---
    lfq_cols_prot = [c for c in prot_df.columns if c.startswith('LFQ intensity ')]

    if not lfq_cols_prot:
        print("🚨 Warning: 'LFQ intensity' columns not found! Falling back to standard 'Intensity'.")
        lfq_cols_prot = [c for c in prot_df.columns if c.startswith('Intensity ') and not c.endswith('not normalized')]

    prot_id_vars_target = ['Protein', 'Protein Group', 'Gene', 'Description', 'Protein Name', 'Score', 'Q-value']
    prot_id_vars = [col for col in prot_id_vars_target if col in prot_df.columns]

    melted_prot = prot_df.melt(
        id_vars=prot_id_vars,
        value_vars=lfq_cols_prot,
        var_name='Sample_Col',
        value_name='Razor Intensity'
    )

    melted_prot.rename(columns={'Description': 'Protein Description'}, inplace=True)
    melted_prot['Sample'] = _extract_sample_id(melted_prot['Sample_Col'])
    melted_prot['Group'] = melted_prot['Sample'].map(sample_to_group)
    melted_prot['Sample_type'] = melted_prot['Sample'].map(sample_to_type)

    # --- SANITY CHECK ---
    unique_mapped_samples_prot = melted_prot['Sample'].unique()
    unique_annot_samples = annot_df['sample'].astype(str).str.strip().unique()
    missing_from_annot = set(unique_mapped_samples_prot) - set(unique_annot_samples)

    if len(missing_from_annot) > 0:
        raise ValueError(f"❌ Mapping Error in proteinGroups.txt! Missing annotations for: {missing_from_annot}")

    melted_prot['Razor Intensity'] = pd.to_numeric(melted_prot['Razor Intensity'], errors='coerce')
    master_prot_df = melted_prot[(melted_prot['Razor Intensity'].notna()) &
                                 (melted_prot['Razor Intensity'] > 0)].copy()

    print(f"-> Loaded {len(master_prot_df)} positive LFQ protein detections.")

    # ==========================================
    # 3. LOAD & MELT PEPTIDE DATA (OPTIONAL)
    # ==========================================
    if not load_peptides:
        return master_prot_df, None

    print(f"Processing {pep_file}...")
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
        pep_df = pd.read_csv(pep_file, sep='\t', low_memory=False)

    if 'Potential contaminant' in pep_df.columns:
        pep_df = pep_df[pep_df['Potential contaminant'] != '+']
    if 'Reverse' in pep_df.columns:
        pep_df = pep_df[pep_df['Reverse'] != '+']

    pep_df.rename(columns={
        'Sequence': 'Peptide Sequence',
        'Leading razor protein': 'Protein',
        'Proteins': 'Protein Group',
        'Gene names': 'Gene',
        'Protein names': 'Protein Name'
    }, inplace=True, errors='ignore')

    if 'Gene' in pep_df.columns:
        pep_df['Gene'] = (
            pep_df['Gene']
            .astype(str)
            .str.split(';')
            .str[0]
            .str.strip()
            .replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        )

    int_cols_pep = [c for c in pep_df.columns if c.startswith('LFQ intensity ')]
    if not int_cols_pep:
        int_cols_pep = [c for c in pep_df.columns if c.startswith('Intensity ')]

    pep_id_vars_target = ['Peptide Sequence', 'Protein', 'Protein Group', 'Gene', 'Protein Name', 'Score', 'PEP']
    pep_id_vars = [col for col in pep_id_vars_target if col in pep_df.columns]

    melted_pep = pep_df.melt(
        id_vars=pep_id_vars,
        value_vars=int_cols_pep,
        var_name='Sample_Col',
        value_name='Razor Intensity'
    )

    melted_pep['Sample'] = _extract_sample_id(melted_pep['Sample_Col'])
    melted_pep['Group'] = melted_pep['Sample'].map(sample_to_group)
    melted_pep['Sample_type'] = melted_pep['Sample'].map(sample_to_type)

    melted_pep['Razor Intensity'] = pd.to_numeric(melted_pep['Razor Intensity'], errors='coerce')
    master_pep_df = melted_pep[(melted_pep['Razor Intensity'].notna()) &
                               (melted_pep['Razor Intensity'] > 0)].copy()

    if 'Description' in prot_df.columns:
        desc_mapping = dict(zip(prot_df['Protein'], prot_df['Description']))
        master_pep_df['Protein Description'] = master_pep_df['Protein'].map(desc_mapping)

    print(f"-> Loaded {len(master_pep_df)} positive peptide detections.")
    return master_prot_df, master_pep_df
