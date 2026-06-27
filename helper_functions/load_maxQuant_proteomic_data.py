#25/05-2026

#Function for loading MaxQuant output data: 

####
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
####


def load_maxQuant_proteomic_data(ROOT, annotation_file):

    # ROOT must be the path to the "txt" folder in the MaxQuant output. 

    # Making annotation dict: 
    annot_df = pd.read_csv(annotation_file, sep='\t')
    
    # Create dictionaries: 
    sample_to_group = dict(zip(annot_df['sample'].astype(str).str.strip(), annot_df['condition']))
    sample_to_type = dict(zip(annot_df['sample'].astype(str).str.strip(), annot_df['sample_type']))

    print(f"Loaded annotations: Found {len(sample_to_group)} total samples in annotation file.")
    
    # ==========================================
    # 1. SETUP PATHS
    # ==========================================
    root_path = Path(ROOT) # Must be the 'txt' folder. 

    prot_file = root_path / "proteinGroups.txt"
    pep_file = root_path / "peptides.txt"
    
    # ==========================================
    # 2. LOAD & MELT PROTEIN DATA
    # ==========================================
    print("\nProcessing proteinGroups.txt...")

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
        prot_df = pd.read_csv(prot_file, sep='\t')

    # --- Contaminant & Decoy Filtering ---
    if 'Potential contaminant' in prot_df.columns:
        prot_df = prot_df[prot_df['Potential contaminant'] != '+']
        print("Filtered out contaminants based on 'Potential contaminant' column.")
    if 'Reverse' in prot_df.columns:
        prot_df = prot_df[prot_df['Reverse'] != '+']
        print("Filtered out reverse hits.")
    if 'Only identified by site' in prot_df.columns:
        prot_df = prot_df[prot_df['Only identified by site'] != '+']
        print("Filtered out 'Only identified by site' hits.")

    # --- Rename columns ---
    prot_df.rename(columns={
        'Majority protein IDs': 'Protein', 
        'Protein IDs': 'Protein Group',      # <--- ADDED PROTEIN GROUP
        'Fasta headers': 'Description', 
        'Gene names': 'Gene', 
        'Protein names': 'Protein Name' 
    }, inplace=True, errors='ignore')

    # --- Dynamically grab MaxQuant LFQ Columns ---
    lfq_cols_prot = [c for c in prot_df.columns if c.startswith('LFQ intensity')]

    if not lfq_cols_prot:
        print("🚨 Warning: 'LFQ intensity' columns not found! Falling back to standard 'Intensity'.")
        lfq_cols_prot = [c for c in prot_df.columns if c.startswith('Intensity') and not c.endswith('not normalized')]

    # --- Setup Safe ID Variables ---
    # Added 'Protein Group' so it survives the melt process
    prot_id_vars_target = ['Protein', 'Protein Group', 'Gene', 'Description', 'Protein Name', 'Score', 'Q-value']
    
    # Safety catch: Only use columns that actually exist in the dataframe to prevent Melt errors
    prot_id_vars = [col for col in prot_id_vars_target if col in prot_df.columns]

    # Melt into long format
    melted_prot = prot_df.melt(
        id_vars=prot_id_vars, 
        value_vars=lfq_cols_prot, 
        var_name='Sample_Col', 
        value_name='Razor Intensity' 
    )

    melted_prot.rename(columns={'Description': 'Protein Description'}, inplace=True)

    # --- Clean sample names ---
    melted_prot['Sample'] = melted_prot['Sample_Col'].str.replace('LFQ intensity ', '', regex=False).str.replace('Intensity ', '', regex=False)
    melted_prot['Sample'] = 'sample_' + melted_prot['Sample'].str.split('_', n=1).str[0].astype(int).astype(str)

    melted_prot['Group'] = melted_prot['Sample'].map(sample_to_group)
    melted_prot['Sample_type'] = melted_prot['Sample'].map(sample_to_type)

    # --- SANITY CHECK ---
    unique_mapped_samples_prot = melted_prot['Sample'].unique()
    unique_annot_samples = annot_df['sample'].astype(str).str.strip().unique()
    missing_from_annot = set(unique_mapped_samples_prot) - set(unique_annot_samples)
    
    if len(missing_from_annot) > 0:
        raise ValueError(f"❌ Mapping Error in proteinGroups.txt! Missing annotations for: {missing_from_annot}")

    # Drop zeroes and NaNs
    master_prot_df = melted_prot[(melted_prot['Razor Intensity'].notna()) & 
                                 (melted_prot['Razor Intensity'] > 0)].copy()

    print(f"-> Loaded {len(master_prot_df)} positive LFQ protein detections.")

    # ==========================================
    # 3. LOAD & MELT PEPTIDE DATA
    # ==========================================
    print("\nProcessing peptides.txt...")
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
        pep_df = pd.read_csv(pep_file, sep='\t')

    # --- Contaminant & Decoy Filtering ---
    if 'Potential contaminant' in pep_df.columns:
        pep_df = pep_df[pep_df['Potential contaminant'] != '+']
        print("Filtered out contaminants based on 'Potential contaminant' column.")
    if 'Reverse' in pep_df.columns:
        pep_df = pep_df[pep_df['Reverse'] != '+']
        print("Filtered out reverse hits.")
        
    # --- Rename columns ---
    pep_df.rename(columns={
        'Sequence': 'Peptide Sequence',
        'Leading razor protein': 'Protein',
        'Proteins': 'Protein Group',     
        'Gene names': 'Gene',
        'Protein names': 'Protein Name'  
    }, inplace=True, errors='ignore')

    # --- Grab Peptide Intensity Columns ---
    int_cols_pep = [c for c in pep_df.columns if c.startswith('LFQ intensity ')]
    if not int_cols_pep:
        print("ℹ️ Note: Peptide-level LFQ not found. Using standard 'Intensity' for peptides.")
        int_cols_pep = [c for c in pep_df.columns if c.startswith('Intensity ')]

    # --- Setup Safe ID Variables ---
    pep_id_vars_target = ['Peptide Sequence', 'Protein', 'Protein Group', 'Gene', 'Protein Name', 'Score', 'PEP']
    pep_id_vars = [col for col in pep_id_vars_target if col in pep_df.columns]

    # --- Melt into long format ---
    melted_pep = pep_df.melt(
        id_vars=pep_id_vars, 
        value_vars=int_cols_pep, 
        var_name='Sample_Col', 
        value_name='Razor Intensity'
    )

    # --- Clean sample names ---
    melted_pep['Sample'] = melted_pep['Sample_Col'].str.replace('LFQ intensity ', '', regex=False).str.replace('Intensity ', '', regex=False)
    melted_pep['Sample'] = 'sample_' + melted_pep['Sample'].str.split('_', n=1).str[0].astype(int).astype(str) 

    melted_pep['Group'] = melted_pep['Sample'].map(sample_to_group)
    melted_pep['Sample_type'] = melted_pep['Sample'].map(sample_to_type)

    # --- SANITY CHECK ---
    unique_mapped_samples_pep = melted_pep['Sample'].unique()
    missing_from_annot_pep = set(unique_mapped_samples_pep) - set(unique_annot_samples)
    
    if len(missing_from_annot_pep) > 0:
        raise ValueError(f"❌ Mapping Error in peptides.txt! Missing annotations for: {missing_from_annot_pep}")

    # Drop zeroes and NaNs
    master_pep_df = melted_pep[(melted_pep['Razor Intensity'].notna()) & 
                            (melted_pep['Razor Intensity'] > 0)].copy()

    # Add 'Protein Description' by mapping it from the protein file
    desc_mapping = dict(zip(prot_df['Protein'], prot_df['Description']))
    master_pep_df['Protein Description'] = master_pep_df['Protein'].map(desc_mapping)

    print(f"-> Loaded {len(master_pep_df)} positive peptide detections.")
    print("\n✅ Data loading complete! 'master_prot_df' and 'master_pep_df' for MaxQuant output are ready.")

    return master_prot_df, master_pep_df
