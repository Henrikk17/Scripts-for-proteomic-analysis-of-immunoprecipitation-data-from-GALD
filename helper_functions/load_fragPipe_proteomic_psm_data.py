#25/05-2026

#Function for loading PTM output data (N-linked glycosylation information): 
import pandas as pd
from pathlib import Path

def load_fragPipe_proteomic_psm_data(ROOT, annotation_file, reduced_columns=True):
    """
    Loads and processes PSM (PTM/Glycan) files from FragPipe output directories,
    mapping sample metadata via an annotation TSV file.
    """
    root_path = Path(ROOT)
    dataFrame_result_list = []

    print(f"Scanning directory: {root_path}")

    # ==========================================
    # 1. LOAD ANNOTATIONS 
    # ==========================================
    # We load the annotation file exactly like the master script
    annot_df = pd.read_csv(annotation_file, sep='\t')
    
    # Create dictionaries, ensuring no sneaky spaces exist in the TSV file itself
    sample_to_group = dict(zip(annot_df['sample'].astype(str).str.strip(), annot_df['condition']))
    sample_to_type = dict(zip(annot_df['sample'].astype(str).str.strip(), annot_df['sample_type']))

    print(f"Loaded annotations: Found {len(sample_to_group)} total samples mapped.")

    # ==========================================
    # 2. DYNAMIC FOLDER SCANNING
    # ==========================================
    # iterdir() looks at every file and folder inside the ROOT path
    for folder_path in root_path.iterdir():
        
        # We only care about DIRECTORIES, and we explicitly skip 'ptm-shepherd-output'
        if folder_path.is_dir() and folder_path.name != "ptm-shepherd-output": 
            
            # The sample name is literally just the folder's name!
            sample_number = folder_path.name 
            clean_sample_name = sample_number.strip() # Strip spaces just in case
            psm_file = folder_path / "psm.tsv"

            if psm_file.exists():
                print(f"Loading {clean_sample_name}...")
                
                # Load the raw file
                psm_df = pd.read_csv(psm_file, sep='\t')

                # Fetch the metadata from your dictionaries
                group = sample_to_group.get(clean_sample_name, "Unknown")
                sample_type = sample_to_type.get(clean_sample_name, "Unknown")
                
                if group == "Unknown":
                    print(f"  -> (!) Note: '{clean_sample_name}' was not found in sample_to_group dictionary. Labeled as Unknown.")

                # Add our custom metadata
                psm_df["Sample"] = clean_sample_name
                psm_df["Group"] = group 
                psm_df["Sample_type"] = sample_type # Added to match your peptide/protein scripts
                
                # Append the ENTIRE dataframe to the list
                dataFrame_result_list.append(psm_df)
            else:
                print(f"(!!!) Warning: psm.tsv not found for {clean_sample_name} (!!!)")
                # Opted for a warning rather than raising an error, in case one folder legitimately failed 
                # but you still want to load the rest. If you prefer a hard crash, you can uncomment below:
                # raise FileNotFoundError(f"psm.tsv not found in {folder_path}.")

    # Check if we actually found anything before trying to concatenate
    if not dataFrame_result_list:
        raise ValueError("No psm.tsv files were found in any valid subdirectories!")

    # ==========================================
    # 3. CREATE MASTER DATAFRAME & CLEANUP
    # ==========================================
    master_psm_df = pd.concat(dataFrame_result_list, ignore_index=True)

    # RAM SAVER: Filter down to just the columns you actually care about.
    if reduced_columns: 
        important_cols = [
            'Sample', 'Group', 'Sample_type', 'Peptide', 'Modified Peptide', 'Gene', 
            'Protein', 'Protein Description', 'Total Glycan Composition', 
            'Assigned Modifications', 'Intensity', 'Probability'
        ]

        # This clever list comprehension ensures Python doesn't crash if a column name slightly changed
        cols_to_keep = [col for col in important_cols if col in master_psm_df.columns]
        master_psm_df = master_psm_df[cols_to_keep]

    print(f"\n✅ Master PSM DataFrame created! Total rows: {len(master_psm_df)}")

    return master_psm_df
