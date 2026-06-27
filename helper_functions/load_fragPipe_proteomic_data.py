#25/05-2026

#Function for loading FragPipe output data: 

import pandas as pd
import numpy as np
from pathlib import Path


def load_fragPipe_proteomic_data(ROOT, annotation_file):
    """
    Loads and processes proteomic files from the specified root path.
    """
    print("--- Loading proteomic output data ---")
    
    # ==========================================
    # 1. LOAD ANNOTATIONS (Bulletproofed with .strip())
    # ==========================================
    annot_df = pd.read_csv(annotation_file, sep='\t')
    
    # Create dictionaries, ensuring no sneaky spaces exist in the TSV file itself
    sample_to_group = dict(zip(annot_df['sample'].astype(str).str.strip(), annot_df['condition']))
    sample_to_type = dict(zip(annot_df['sample'].astype(str).str.strip(), annot_df['sample_type']))

    print(f"Loaded annotations: Found {len(sample_to_group)} total samples.")

    root_path = Path(ROOT) #The ROOT must be the FragPipe output folder! 
    prot_file = root_path / "combined_protein.tsv"
    pep_file = root_path / "combined_peptide.tsv"

    # ==========================================
    # 2. LOAD & MELT PROTEIN DATA
    # ==========================================
    print("\nProcessing combined_protein.tsv...")
    prot_df = pd.read_csv(prot_file, sep='\t')

    # Filter contaminants
    if 'Is Contaminant' in prot_df.columns:
        prot_df = prot_df[prot_df['Is Contaminant'] == False]
    else:
        print("(!) Warning: 'Is Contaminant' column not found in protein data.")
        prot_df = prot_df[~prot_df['Protein'].str.contains("contam", na=False)]

    # Dynamically grab ONLY the Intensity columns
    lfq_cols_prot = [c for c in prot_df.columns if c.endswith('MaxLFQ Intensity')]
    suffix_prot = 'MaxLFQ Intensity'

    if len(lfq_cols_prot) == 0:
        lfq_cols_prot = [c for c in prot_df.columns if c.endswith('Razor Intensity')]
        suffix_prot = 'Razor Intensity'

    if len(lfq_cols_prot) == 0:
        lfq_cols_prot = [c for c in prot_df.columns if c.endswith('Intensity')]
        suffix_prot = 'Intensity'

    # Use the correct column name for the description
    desc_col = 'Protein Description' if 'Protein Description' in prot_df.columns else 'Description'

    # Melt into long format
    melted_prot = prot_df.melt(
        id_vars=['Protein', 'Gene', desc_col], 
        value_vars=lfq_cols_prot, 
        var_name='Sample_Col', 
        value_name='Sample_Intensity' 
    )

    # Rename columns back to standard
    melted_prot.rename(columns={
        'Sample_Intensity': 'Razor Intensity', 
        desc_col: 'Protein Description'
    }, inplace=True)

    # Strip the suffix AND any leftover white spaces so mapping works!
    melted_prot['Sample'] = melted_prot['Sample_Col'].str.replace(suffix_prot, '', regex=False).str.strip()
    
    # Map the group and type
    melted_prot['Group'] = melted_prot['Sample'].map(sample_to_group)
    melted_prot['Sample_type'] = melted_prot['Sample'].map(sample_to_type)
    
    print("Melt df shape before removing zeroes/NaN intensity values:", melted_prot.shape)

    # Drop zeroes and NaNs
    master_prot_df = melted_prot[melted_prot['Razor Intensity'].notna() & (melted_prot['Razor Intensity'] > 0)].copy()
    print(f"-> Loaded {len(master_prot_df)} positive protein detections.")
    
    # ==========================================
    # 3. LOAD & MELT PEPTIDE DATA
    # ==========================================
    print("\nProcessing combined_peptide.tsv...")
    pep_df = pd.read_csv(pep_file, sep='\t')

    # Filter contaminants
    if 'Is Contaminant' in pep_df.columns:
        pep_df = pep_df[pep_df['Is Contaminant'] == False]
    else:
        print("(!) Warning: 'Is Contaminant' column not found in peptide data.")
        pep_df = pep_df[~pep_df['Protein'].str.contains("contam", na=False)]

    # Dynamically grab Intensity columns for peptides
    lfq_cols_pep = [c for c in pep_df.columns if c.endswith('MaxLFQ Intensity')]
    suffix_pep = 'MaxLFQ Intensity'

    if len(lfq_cols_pep) == 0:
        lfq_cols_pep = [c for c in pep_df.columns if c.endswith('Razor Intensity')]
        suffix_pep = 'Razor Intensity'

    if len(lfq_cols_pep) == 0:
        lfq_cols_pep = [c for c in pep_df.columns if c.endswith('Intensity')]
        suffix_pep = 'Intensity'

    # Handle sequence column naming
    if 'Peptide' in pep_df.columns:
        seq_col = 'Peptide'
    elif 'Peptide Sequence' in pep_df.columns:
        seq_col = 'Peptide Sequence'
    else:
        seq_col = 'Sequence'

    # Melt safely
    melted_pep = pep_df.melt(
        id_vars=[seq_col, 'Protein', 'Gene'], 
        value_vars=lfq_cols_pep, 
        var_name='Sample_Col', 
        value_name='Razor Intensity' 
    )

    melted_pep.rename(columns={seq_col: 'Peptide Sequence'}, inplace=True)

    # Strip suffix AND trailing spaces!
    melted_pep['Sample'] = melted_pep['Sample_Col'].str.replace(suffix_pep, '', regex=False).str.strip()
    melted_pep['Group'] = melted_pep['Sample'].map(sample_to_group)
    melted_pep['Sample_type'] = melted_pep['Sample'].map(sample_to_type)

    # Drop zeroes and NaNs
    master_pep_df = melted_pep[melted_pep['Razor Intensity'].notna() & (melted_pep['Razor Intensity'] > 0)].copy()

    # Fixed: Safely map description using the dynamic desc_col variable
    desc_mapping = dict(zip(prot_df['Protein'], prot_df[desc_col]))
    master_pep_df['Protein Description'] = master_pep_df['Protein'].map(desc_mapping)

    print(f"-> Loaded {len(master_pep_df)} positive peptide detections.")

    # ==========================================
    # 4. GENERATE SHORT PROTEIN IDs
    # ==========================================
    print("\nGenerating 'Protein_short' IDs...")

    #sp|P50914|RL14_HUMAN --> P50914

    master_prot_df['Protein_short'] = master_prot_df['Protein'].apply(
        lambda x: str(x).split('|')[1] if pd.notna(x) and '|' in str(x) else x
    )

    master_pep_df['Protein_short'] = master_pep_df['Protein'].apply(
        lambda x: str(x).split('|')[1] if pd.notna(x) and '|' in str(x) else x
    )

    print("\n✅ Data loading complete! 'master_prot_df' and 'master_pep_df' are ready.")

    return master_prot_df, master_pep_df
