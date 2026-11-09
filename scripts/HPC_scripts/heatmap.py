# 09/11-2026
# Peptide-Level Heatmap Generator for FragPipe Workflows

import os
import re
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from helper_functions.manage_datasets import get_all_fragpipe_datasets

# Ensure output directory exists
os.makedirs("../plots", exist_ok=True)

# Helper function for "natural" alphanumeric sorting
def alphanum_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', str(s))]

def generate_peptide_heatmaps(target_gene_list, datasets, group_palette):
    """
    Generates peptide-level heatmaps for target genes across all FragPipe datasets,
    saving one multi-page PDF per gene.
    """
    # Extract sequence column name safely (FragPipe sometimes uses 'Peptide' instead of 'Peptide Sequence')
    for gene in target_gene_list:
        print(f"\n{'='*80}")
        print(f"🎯 GENERATING HEATMAPS FOR TARGET: {gene}")
        print(f"{'='*80}")

        pdf_path = f"../plots/Peptide_Heatmap_{gene}.pdf"
        
        with PdfPages(pdf_path) as pdf:
            plots_generated = 0
            
            for key, data in datasets.items():
                run_name = data["label"]
                pep_df = data["pep"]
                
                # Skip redundant 'With LB' runs to keep the PDF clean
                if "With LB" in run_name or pep_df is None or pep_df.empty:
                    continue

                # Auto-detect column names
                pep_col = 'Peptide Sequence' if 'Peptide Sequence' in pep_df.columns else 'Peptide'
                
                if 'Razor Intensity' in pep_df.columns:
                    quant_col = 'Razor Intensity'
                elif 'MaxLFQ Intensity' in pep_df.columns:
                    quant_col = 'MaxLFQ Intensity'
                else:
                    quant_col = 'Intensity'

                # ==========================================
                # 1. Filter for Target Peptides
                # ==========================================
                df_filtered = pep_df[pep_df.get("Gene", pd.Series(dtype=str)).astype(str).str.strip() == gene].copy()

                if df_filtered.empty:
                    print(f"  ⏩ No target peptides found for {gene} in {run_name}. Skipping.")
                    continue
                
                # ==========================================
                # 2. Extract and Sort Samples
                # ==========================================
                df_filtered[quant_col] = pd.to_numeric(df_filtered[quant_col], errors='coerce')
                df_filtered['Group'] = df_filtered['Group'].astype(str).str.strip()
                df_filtered['Sample_Label'] = df_filtered['Sample'].astype(str) + '\n' + df_filtered['Group']
                
                all_samples_df = pep_df[['Sample', 'Group']].drop_duplicates().copy()
                all_samples_df['Group'] = all_samples_df['Group'].astype(str).str.strip()
                all_samples_df['Sample_Label'] = all_samples_df['Sample'].astype(str) + '\n' + all_samples_df['Group']
                
                sample_group_map = all_samples_df.set_index('Sample_Label')['Group'].to_dict()
                
                # Sort logic: Group GALD first, then Healthy, then Spouse, then alphanumeric
                def sort_key(sample_label):
                    group = sample_group_map.get(sample_label, "")
                    group_order = 0 if group == 'GALD' else (1 if group == 'Healthy donor' else 2)
                    return (group_order, alphanum_key(sample_label))
                
                sorted_samples = sorted(sample_group_map.keys(), key=sort_key)

                # ==========================================
                # 3. Pivot and Transform Data
                # ==========================================
                pivot_df = df_filtered.pivot_table(
                    index=pep_col, 
                    columns='Sample_Label', 
                    values=quant_col, 
                    aggfunc='sum'
                )

                # Ensure all samples are present (fill missing with NaN)
                for sample in sorted_samples:
                    if sample not in pivot_df.columns:
                        pivot_df[sample] = np.nan
                        
                pivot_df = pivot_df[sorted_samples]

                # Log2 transform and row-wise Z-score
                log_df = np.log2(pivot_df.replace(0, np.nan))
                with np.errstate(invalid='ignore'):
                    z_df = log_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

                z_df = z_df.dropna(how='all')
                
                if z_df.empty:
                    print(f"  ⏩ No valid numeric data left for {gene} in {run_name}. Skipping.")
                    continue
                    
                # Sort rows by least missing values, then highest mean intensity
                z_df['na_count'] = z_df.isna().sum(axis=1)
                z_df['mean_intensity'] = log_df.mean(axis=1)
                z_df = z_df.sort_values(by=['na_count', 'mean_intensity'], ascending=[True, False])
                z_df = z_df.drop(columns=['na_count', 'mean_intensity'])

                # ==========================================
                # 4. Draw Heatmap
                # ==========================================
                print(f"  [*] Plotting {run_name} ({len(z_df)} peptides)...")
                
                dynamic_width = max(12, len(sorted_samples) * 0.45)
                fig, ax = plt.subplots(figsize=(dynamic_width, max(4, len(z_df) * 0.35))) 
                ax.set_facecolor('#404040') 

                sns.heatmap(
                    z_df,
                    cmap='Oranges',
                    center=0,
                    vmin=-2.5, vmax=2.5,
                    ax=ax,
                    cbar_kws={'label': 'log2 Z-Score (Peptide abundance)', 'shrink': 0.8, 'pad': 0.05},
                    linewidths=0.5,
                    linecolor='white'
                )

                ax.set_title(f"{gene} Peptide-level Abundance\n{run_name}", fontsize=14, weight='bold', pad=40)
                ax.set_xlabel("Samples", fontsize=12, labelpad=10)
                ax.set_ylabel("Peptide Sequence", fontsize=12, labelpad=10)
                
                # Format X-axis labels with cohort colors
                labels = ax.get_xticklabels()
                for tick_label in labels:
                    full_label = tick_label.get_text()
                    group_name = sample_group_map.get(full_label)
                    sample_name_only = full_label.split('\n')[0]
                    tick_label.set_text(sample_name_only)
                    
                    if group_name in group_palette:
                        tick_label.set_color("black")
                        tick_label.set_fontweight('bold')
                        tick_label.set_bbox(dict(facecolor=group_palette[group_name], edgecolor='none', alpha=0.3, boxstyle='round,pad=0.3'))

                ax.set_xticklabels(labels)
                plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
                plt.yticks(fontsize=9)

                # Add custom legend
                legend_patches = [mpatches.Patch(color=color, alpha=0.6, label=group) 
                                  for group, color in group_palette.items() if group in sample_group_map.values()]
                ax.legend(handles=legend_patches, title='Cohorts', loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                plots_generated += 1
                
        if plots_generated > 0:
            print(f"  ✅ Saved {plots_generated} heatmaps to {pdf_path}")
        else:
            print(f"  ❌ No plots were generated for {gene} (deleted empty PDF).")
            os.remove(pdf_path)


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    print("Loading cached datasets (including peptides)...")
    # MUST set load_peptides=True to read the peptide dataframes
    datasets = get_all_fragpipe_datasets(use_cache=True, load_peptides=True)

    # Targets identified from the previous version of the Volcano Overlap analysis (before it got updated)
    target_gene_list = ["CP", "HP", "C4BPA", "VASP", "OLA1"] 

    group_palette = {
        'GALD': 'firebrick', 
        'Healthy donor': 'steelblue', 
        'GALD_spouse': 'forestgreen'
    }

    generate_peptide_heatmaps(target_gene_list, datasets, group_palette)
    