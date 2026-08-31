#25/05-2026

#Function for loading FragPipe output data (MaxQuant-Compatible & Auto-Extracting Sample IDs): 

####
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
####

def load_fragPipe_proteomic_data(ROOT, annotation_file):
    """
    Loads and processes proteomic files from the specified FragPipe root path.
    Standardizes column names to perfectly match MaxQuant outputs and extracts clean sample IDs.
    """
    print("--- Loading FragPipe proteomic output data ---")
    
    # ==========================================
    # 1. LOAD ANNOTATIONS 
    # ==========================================
    annot_df = pd.read_csv(annotation_file, sep='\t')
    
    # Create dictionaries, ensuring no sneaky spaces exist
    sample_to_group = dict(zip(annot_df['sample'].astype(str).str.strip(), annot_df['condition']))
    sample_to_type = dict(zip(annot_df['sample'].astype(str).str.strip(), annot_df['sample_type']))

    print(f"Loaded annotations: Found {len(sample_to_group)} total samples in annotation file.")

    root_path = Path(ROOT) 
    prot_file = root_path / "combined_protein.tsv"
    pep_file = root_path / "combined_peptide.tsv"

    # ==========================================
    # 2. LOAD & MELT PROTEIN DATA
    # ==========================================
    print("\nProcessing combined_protein.tsv...")
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
        prot_df = pd.read_csv(prot_file, sep='\t')

    # --- Contaminant Filtering ---
    if 'Is Contaminant' in prot_df.columns:
        prot_df = prot_df[prot_df['Is Contaminant'] == False]
    else:
        print("(!) Warning: 'Is Contaminant' column not found in protein data.")
        prot_df = prot_df[~prot_df['Protein'].str.contains("contam", na=False)]

    # --- Rename & Standardize Columns to Match MaxQuant ---
    if 'Description' in prot_df.columns and 'Protein Description' not in prot_df.columns:
        prot_df.rename(columns={'Description': 'Protein Description'}, inplace=True)
        
    if 'Protein Group' not in prot_df.columns:
        prot_df['Protein Group'] = prot_df['Protein'] 
        
    if 'Protein Name' not in prot_df.columns:
        prot_df['Protein Name'] = prot_df['Protein'] 

    # --- Dynamically grab FragPipe LFQ Columns ---
    lfq_cols_prot = [c for c in prot_df.columns if c.endswith('MaxLFQ Intensity')]
    suffix_prot = ' MaxLFQ Intensity'

    if not lfq_cols_prot:
        lfq_cols_prot = [c for c in prot_df.columns if c.endswith('Razor Intensity')]
        suffix_prot = ' Razor Intensity'

    if not lfq_cols_prot:
        lfq_cols_prot = [c for c in prot_df.columns if c.endswith('Intensity') and not c.endswith('MaxLFQ Intensity')]
        suffix_prot = ' Intensity'

    # --- Setup Safe ID Variables ---
    prot_id_vars_target = ['Protein', 'Protein Group', 'Gene', 'Protein Description', 'Protein Name', 'Protein Probability']
    prot_id_vars = [col for col in prot_id_vars_target if col in prot_df.columns]

    # Melt into long format
    melted_prot = prot_df.melt(
        id_vars=prot_id_vars, 
        value_vars=lfq_cols_prot, 
        var_name='Sample_Col', 
        value_name='Razor Intensity' 
    )

    # --- Clean & Extract Sample IDs ---
    raw_sample_names_prot = melted_prot['Sample_Col'].str.replace(suffix_prot, '', regex=False).str.strip()
    
    # Use a case-insensitive regex (?i) and make the trailing underscore optional
    extracted_ids_prot = raw_sample_names_prot.str.extract(r'(?i)_s(\d+)', expand=False)
    
    if extracted_ids_prot.isna().any():
        failed_raw = raw_sample_names_prot[extracted_ids_prot.isna()].unique()
        raise ValueError(f"❌ Regex extraction failed! Could not find '_s##' in these protein files:\n{failed_raw}")

    # Convert to integer to strip leading zeros, then back to string!
    melted_prot['Sample'] = 'sample_' + extracted_ids_prot.astype(int).astype(str)
    
    melted_prot['Group'] = melted_prot['Sample'].map(sample_to_group)
    melted_prot['Sample_type'] = melted_prot['Sample'].map(sample_to_type)

    # --- SANITY CHECK ---
    unique_mapped_samples_prot = melted_prot['Sample'].unique()
    unique_annot_samples = annot_df['sample'].astype(str).str.strip().unique()
    missing_from_annot = set(unique_mapped_samples_prot) - set(unique_annot_samples)
    
    if len(missing_from_annot) > 0:
        raise ValueError(f"❌ Mapping Error in combined_protein.tsv! Missing annotations for extracted IDs: {missing_from_annot}")

    # Drop zeroes and NaNs
    master_prot_df = melted_prot[(melted_prot['Razor Intensity'].notna()) & 
                                 (melted_prot['Razor Intensity'] > 0)].copy()

    # Generate short ID
    master_prot_df['Protein_short'] = master_prot_df['Protein'].apply(
        lambda x: str(x).split('|')[1] if pd.notna(x) and '|' in str(x) else x
    )

    print(f"-> Loaded {len(master_prot_df)} positive LFQ protein detections.")

    # ==========================================
    # 3. LOAD & MELT PEPTIDE DATA
    # ==========================================
    print("\nProcessing combined_peptide.tsv...")
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
        pep_df = pd.read_csv(pep_file, sep='\t')

    # --- Contaminant Filtering ---
    if 'Is Contaminant' in pep_df.columns:
        pep_df = pep_df[pep_df['Is Contaminant'] == False]
    else:
        print("(!) Warning: 'Is Contaminant' column not found in peptide data.")
        pep_df = pep_df[~pep_df['Protein'].str.contains("contam", na=False)]

    # --- Rename & Standardize Columns to Match MaxQuant ---
    if 'Peptide' in pep_df.columns:
        pep_df.rename(columns={'Peptide': 'Peptide Sequence'}, inplace=True)
    elif 'Sequence' in pep_df.columns and 'Peptide Sequence' not in pep_df.columns:
        pep_df.rename(columns={'Sequence': 'Peptide Sequence'}, inplace=True)

    if 'Protein Group' not in pep_df.columns:
        pep_df['Protein Group'] = pep_df['Protein']
        
    if 'Protein Name' not in pep_df.columns:
        pep_df['Protein Name'] = pep_df['Protein']

    # --- Grab Peptide Intensity Columns ---
    lfq_cols_pep = [c for c in pep_df.columns if c.endswith('MaxLFQ Intensity')]
    suffix_pep = ' MaxLFQ Intensity'

    if not lfq_cols_pep:
        lfq_cols_pep = [c for c in pep_df.columns if c.endswith('Razor Intensity')]
        suffix_pep = ' Razor Intensity'

    if not lfq_cols_pep:
        lfq_cols_pep = [c for c in pep_df.columns if c.endswith('Intensity')]
        suffix_pep = ' Intensity'

    # --- Setup Safe ID Variables ---
    pep_id_vars_target = ['Peptide Sequence', 'Protein', 'Protein Group', 'Gene', 'Protein Name', 'Peptide Probability']
    pep_id_vars = [col for col in pep_id_vars_target if col in pep_df.columns]

    # --- Melt into long format ---
    melted_pep = pep_df.melt(
        id_vars=pep_id_vars, 
        value_vars=lfq_cols_pep, 
        var_name='Sample_Col', 
        value_name='Razor Intensity'
    )

    # --- Clean & Extract Sample IDs ---
    raw_sample_names_pep = melted_pep['Sample_Col'].str.replace(suffix_pep, '', regex=False).str.strip()
    
    extracted_ids_pep = raw_sample_names_pep.str.extract(r'(?i)_s(\d+)', expand=False)
    
    if extracted_ids_pep.isna().any():
        failed_raw = raw_sample_names_pep[extracted_ids_pep.isna()].unique()
        raise ValueError(f"❌ Regex extraction failed! Could not find '_s##' in these peptide files:\n{failed_raw}")

    # Convert to integer to strip leading zeros, then back to string!
    melted_pep['Sample'] = 'sample_' + extracted_ids_pep.astype(int).astype(str)

    melted_pep['Group'] = melted_pep['Sample'].map(sample_to_group)
    melted_pep['Sample_type'] = melted_pep['Sample'].map(sample_to_type)

    # --- SANITY CHECK ---
    unique_mapped_samples_pep = melted_pep['Sample'].unique()
    missing_from_annot_pep = set(unique_mapped_samples_pep) - set(unique_annot_samples)
    
    if len(missing_from_annot_pep) > 0:
        raise ValueError(f"❌ Mapping Error in combined_peptide.tsv! Missing annotations for extracted IDs: {missing_from_annot_pep}")

    # Drop zeroes and NaNs
    master_pep_df = melted_pep[(melted_pep['Razor Intensity'].notna()) & 
                               (melted_pep['Razor Intensity'] > 0)].copy()

    # Add 'Protein Description' by mapping it from the protein file
    desc_mapping = dict(zip(prot_df['Protein'], prot_df['Protein Description']))
    master_pep_df['Protein Description'] = master_pep_df['Protein'].map(desc_mapping)

    # Generate short ID
    master_pep_df['Protein_short'] = master_pep_df['Protein'].apply(
        lambda x: str(x).split('|')[1] if pd.notna(x) and '|' in str(x) else x
    )

    print(f"-> Loaded {len(master_pep_df)} positive peptide detections.")
    print("\n✅ Data loading complete! FragPipe 'master_prot_df' and 'master_pep_df' are ready and structurally match MaxQuant output.")

    return master_prot_df, master_pep_df