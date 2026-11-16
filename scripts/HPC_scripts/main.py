# 16/11-2026
# Master Proteomic Pipeline: MaxQuant vs. FragPipe 
# Includes: Volcano, Fisher, PCA, Random Forest, and Run 6 Contaminant ANOVA

import os
import pickle
import warnings
from pathlib import Path
import matplotlib
# Ensure plots can be generated on headless HPC nodes without X11 forwarding errors
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, fisher_exact
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from helper_functions.manage_datasets import get_all_fragpipe_datasets
from helper_functions.load_maxQuant_proteomic_data import (
    load_maxQuant_proteomic_data,
    find_maxquant_txt_dir
)
from helper_functions.load_fragPipe_proteomic_data import load_fragPipe_proteomic_data

warnings.filterwarnings('ignore')

# ==========================================
# 🎛️ EXECUTION TOGGLES (FOR DEBUGGING/SPEED)
# ==========================================
RUN_VOLCANO = True
RUN_FISHER  = True
RUN_PCA     = True
RUN_RF      = True
RUN_ANOVA   = True
RUN_SUMMARY = True  

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
FRAGPIPE_WORKFLOW = "imms"
HIGHLIGHT_GENES = ["CP", "TFRC", "TFR2", "TAGLN2", "TLN1"]

USE_FDR = True       
P_CUTOFF = 0.05      
LOG2FC_CUTOFF = 1.0  

OUT_DIR = Path("../plots/mq_vs_fragpipe_comparison")
CACHE_DIR = Path("../cache_master_pipeline")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. DATA LOADING & PEPTIDE TRACKING
# ==========================================
global_pep_counts_mq = {}
global_pep_counts_fp = {}

def extract_peptide_counts(raw_df, engine, run_label, gene_col='Gene'):
    target_dict = global_pep_counts_mq if engine == 'mq' else global_pep_counts_fp
    
    pep_cols = [c for c in raw_df.columns if 'peptide' in c.lower() and ('count' in c.lower() or 'razor' in c.lower() or 'total' in c.lower())]
    if not pep_cols:
        return 
        
    pep_col = pep_cols[0] 
    
    for _, row in raw_df.dropna(subset=[gene_col]).iterrows():
        gene = str(row[gene_col]).split(';')[0].strip()
        try:
            count = float(row[pep_col])
            if not np.isnan(count):
                if gene not in target_dict:
                    target_dict[gene] = {}
                target_dict[gene][run_label] = max(target_dict[gene].get(run_label, 0), int(count))
        except (ValueError, TypeError):
            pass

def get_maxquant_datasets(base_dir="~/DTU-GALD"):
    cache_file = CACHE_DIR / "mq_master_data.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

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

# ==========================================
# 2. RUN 6 ANOVA (CONTAMINANT CLASSIFICATION)
# ==========================================
def run_run6_anova(prot_df_with_lb, engine_name):
    print(f"\n🔬 Running ANOVA Contaminant Classification for {engine_name}...")
    
    quant_col = 'Razor Intensity' if 'Razor Intensity' in prot_df_with_lb.columns else (
        'MaxLFQ Intensity' if 'MaxLFQ Intensity' in prot_df_with_lb.columns else 'Intensity'
    )

    df = prot_df_with_lb.dropna(subset=['Gene']).copy()
    df['Gene'] = df['Gene'].astype(str).str.split(';').str[0].str.strip()
    
    # Robustly handle Sample strings (e.g. 'Sample_1' vs 'sample_1')
    df['Sample_Num'] = df['Sample'].astype(str).str.lower().str.replace('sample_', '', regex=False).astype(int)

    df['Subject_ID'] = df['Sample_Num'].apply(lambda n: n if n <= 42 else n - 42)
    
    if 'Sample_type' in df.columns:
        df['Matrix'] = df['Sample_type'].apply(lambda x: 'Buffer' if 'buffer' in str(x).lower() else 'Liver')
    else:
        df['Matrix'] = df['Sample_Num'].apply(lambda n: 'Liver' if n <= 42 else 'Buffer')
        
    num_buffer = (df['Matrix'] == 'Buffer').sum()
    print(f"  [*] Detected {num_buffer} raw Lysate Buffer data points in dataframe.")
    
    if 'Buffer' not in df['Matrix'].values:
        print(f"  ⚠️ WARNING: No 'Lysate buffer' samples detected in {engine_name} data! Skipping ANOVA.")
        return {}

    sample_meta = df[['Sample', 'Subject_ID', 'Matrix', 'Group']].drop_duplicates().set_index('Sample')

    wide_df = df.pivot_table(index='Gene', columns='Sample', values=quant_col, aggfunc='sum')
    wide_df.replace(0, np.nan, inplace=True)
    wide_df = wide_df.dropna(thresh=5, axis=0) 
    
    log2_df = np.log2(wide_df)

    np.random.seed(42)
    imputed_df = log2_df.copy()
    for col in imputed_df.columns:
        valid = imputed_df[col].dropna()
        if len(valid) > 2:
            m, s = valid.mean(), valid.std()
            imputed_df.loc[imputed_df[col].isna(), col] = np.random.normal(loc=m - 1.8 * s, scale=s * 0.3, size=imputed_df[col].isna().sum())

    records = []
    for gene, row in imputed_df.iterrows():
        sub = pd.DataFrame({
            'Intensity': row.values,
            'Matrix': [sample_meta.loc[c, 'Matrix'] for c in imputed_df.columns],
            'Subject_ID': [sample_meta.loc[c, 'Subject_ID'] for c in imputed_df.columns],
            'Cohort': [str(sample_meta.loc[c, 'Group']).strip() for c in imputed_df.columns]
        })
        
        piv = sub.pivot(index='Subject_ID', columns='Matrix', values='Intensity')
        if 'Liver' not in piv.columns or 'Buffer' not in piv.columns: continue
            
        piv['Delta'] = piv['Liver'] - piv['Buffer']
        piv['Cohort'] = sub.drop_duplicates('Subject_ID').set_index('Subject_ID')['Cohort']
        
        mean_l = sub[sub['Matrix'] == 'Liver']['Intensity'].mean()
        mean_b = sub[sub['Matrix'] == 'Buffer']['Intensity'].mean()
        log2fc_matrix = mean_l - mean_b
        
        gald_d = piv[piv['Cohort'] == 'GALD']['Delta'].dropna()
        ctrl_d = piv[piv['Cohort'] != 'GALD']['Delta'].dropna()
        
        if len(gald_d) > 1 and len(ctrl_d) > 1:
            _, p_delta = ttest_ind(gald_d, ctrl_d, equal_var=False)
        else:
            p_delta = np.nan
            
        records.append({'Gene': gene, 'Log2FC_Liver_vs_Buffer': log2fc_matrix, 'P_Paired_Delta': p_delta})

    res = pd.DataFrame(records).dropna()
    res['FDR_Paired_Delta'] = multipletests(res['P_Paired_Delta'], method='fdr_bh')[1]
    
    def classify(r):
        if r['Log2FC_Liver_vs_Buffer'] <= 0.5:
            return "Contaminant (Maternal/Buffer Carryover)"
        elif r['P_Paired_Delta'] < 0.05:
            return "True Fetal Liver Target (GALD Enriched)"
        else:
            return "Non-Specific Fetal Liver Background"
            
    res['ANOVA_Class'] = res.apply(classify, axis=1)
    
    out_file = OUT_DIR / f"Run6_ANOVA_Classification_{engine_name}.txt"
    res.to_csv(out_file, sep='\t', index=False)
    print(f"  ✅ Saved ANOVA classifications to {out_file.name}")
    
    return res.set_index('Gene')['ANOVA_Class'].to_dict()


# ==========================================
# 3. CORE SUBSET ANALYSES PREPARATION
# ==========================================
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
    wide_filtered = wide_df[target_samples].dropna(thresh=3, axis=0)
    
    c_samps_valid = [s for s in c_samps if s in target_samples]
    k_samps_valid = [s for s in k_samps if s in target_samples]
    return wide_filtered, c_samps_valid, k_samps_valid, s_to_g

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

# ==========================================
# 4. MASTER ORCHESTRATOR
# ==========================================
def run_engine_pipeline(subsets_dict, prot_with_lb, engine_name):
    print(f"\n{'='*70}\n🚀 RUNNING PIPELINE: {engine_name}\n{'='*70}")
    
    anova_classes = {}
    if RUN_ANOVA:
        anova_classes = run_run6_anova(prot_with_lb, engine_name)
    
    pdf_volcano = PdfPages(OUT_DIR / f"Volcano_{engine_name}.pdf") if RUN_VOLCANO else None
    pdf_pca = PdfPages(OUT_DIR / f"PCA_{engine_name}.pdf") if RUN_PCA else None
    
    res_volcano, res_fisher, res_rf = {}, {}, {}
    
    for subset_name, (df_prot, g_case, g_ctrl) in subsets_dict.items():
        print(f"  • Processing subset: {subset_name}")
        wide_df, c_samps, k_samps, s_to_g = prep_wide_df(df_prot, g_case, g_ctrl)
        imputed_df = impute_perseus(wide_df)
        
        if RUN_VOLCANO:
            records = []
            for gene, row in imputed_df.iterrows():
                c_vals, k_vals = row[c_samps].values, row[k_samps].values
                log2fc = c_vals.mean() - k_vals.mean()
                _, p = ttest_ind(c_vals, k_vals, equal_var=False)
                records.append({'Gene': gene, 'Log2FC': log2fc, 'p_value': p})
            
            v_df = pd.DataFrame(records).dropna()
            v_df['FDR'] = multipletests(v_df['p_value'], method='fdr_bh')[1]
            stat = 'FDR' if USE_FDR else 'p_value'
            
            sig = v_df[(v_df[stat] < P_CUTOFF) & (v_df['Log2FC'].abs() >= LOG2FC_CUTOFF)]
            for _, r in sig.iterrows():
                res_volcano.setdefault(r['Gene'], []).append(f"{subset_name}({'Up' if r['Log2FC']>0 else 'Down'})")

            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(v_df['Log2FC'], -np.log10(v_df['p_value']), alpha=0.5, color='grey', s=10)
            ax.scatter(sig['Log2FC'], -np.log10(sig['p_value']), alpha=0.8, color='firebrick', s=20)
            ax.set_title(f"{engine_name} Volcano: {subset_name}")
            pdf_volcano.savefig(fig, bbox_inches='tight'); plt.close()

        if RUN_FISHER:
            for gene, row in wide_df.iterrows():
                c_pres, k_pres = row[c_samps].notna().sum(), row[k_samps].notna().sum()
                if c_pres > k_pres: 
                    _, p = fisher_exact([[c_pres, len(c_samps)-c_pres], [k_pres, len(k_samps)-k_pres]])
                    if p < P_CUTOFF:
                        res_fisher.setdefault(gene, []).append(f"{subset_name}({c_pres}/{len(c_samps)} vs {k_pres}/{len(k_samps)})")

        if RUN_PCA:
            X_sc = StandardScaler().fit_transform(imputed_df.T)
            y = np.array([s_to_g[s] for s in imputed_df.columns])
            pcs = PCA(n_components=2, random_state=42).fit_transform(X_sc)
            
            fig, ax = plt.subplots()
            sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=y, ax=ax)
            ax.set_title(f"{engine_name} PCA: {subset_name}")
            pdf_pca.savefig(fig, bbox_inches='tight'); plt.close()

        if RUN_RF:
            X_sc = StandardScaler().fit_transform(imputed_df.T)
            y_bin = np.array([1 if s_to_g[s] == 'GALD' else 0 for s in imputed_df.columns])
            rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
            rf.fit(X_sc, y_bin)
            
            imp = pd.DataFrame({'Gene': imputed_df.index, 'Imp': rf.feature_importances_}).sort_values('Imp', ascending=False).head(20)
            for i, r in enumerate(imp.iterrows()):
                res_rf.setdefault(r[1]['Gene'], []).append(f"{subset_name}(#{i+1})")

    if RUN_VOLCANO: pdf_volcano.close()
    if RUN_PCA: pdf_pca.close()

    engine_results = {
        'anova_class': anova_classes,
        'volcano_sig': {g: hits for g, hits in res_volcano.items() if len(hits) >= 2},
        'fisher_sig': {g: hits for g, hits in res_fisher.items() if len(hits) >= 2},
        'rf_sig': {g: hits for g, hits in res_rf.items() if len(hits) >= 2},
    }
    return engine_results


# ==========================================
# 5. FINAL CROSS-ENGINE & CROSS-ANALYSIS SUMMARY
# ==========================================
def compile_master_summary(mq_res, fp_res):
    print("\n" + "="*70)
    print("🏆 COMPILING MASTER CROSS-ENGINE SUMMARY")
    print("="*70)
    
    all_genes = set(mq_res['volcano_sig'].keys()) | set(mq_res['fisher_sig'].keys()) | set(mq_res['rf_sig'].keys()) | \
                set(fp_res['volcano_sig'].keys()) | set(fp_res['fisher_sig'].keys()) | set(fp_res['rf_sig'].keys())
                
    summary_rows = []
    for g in all_genes:
        mq_counts = global_pep_counts_mq.get(g, {})
        fp_counts = global_pep_counts_fp.get(g, {})
        
        mq_parts = [f"MQ {run}: {cnt}" for run, cnt in sorted(mq_counts.items())]
        fp_parts = [f"FP {run}: {cnt}" for run, cnt in sorted(fp_counts.items())]
        all_parts = mq_parts + fp_parts
        
        pep_string = "[" + ", ".join(all_parts) + "]" if all_parts else "[No peptide data]"
        
        in_mq_volc = "Yes" if g in mq_res['volcano_sig'] else "-"
        in_fp_volc = "Yes" if g in fp_res['volcano_sig'] else "-"
        in_mq_fish = "Yes" if g in mq_res['fisher_sig'] else "-"
        in_fp_fish = "Yes" if g in fp_res['fisher_sig'] else "-"
        in_mq_rf = "Yes" if g in mq_res['rf_sig'] else "-"
        in_fp_rf = "Yes" if g in fp_res['rf_sig'] else "-"
        
        mq_any = (in_mq_volc == "Yes" or in_mq_fish == "Yes" or in_mq_rf == "Yes")
        fp_any = (in_fp_volc == "Yes" or in_fp_fish == "Yes" or in_fp_rf == "Yes")
        
        if not (mq_any and fp_any): 
            continue 

        anova_mq = mq_res['anova_class'].get(g, "Not Evaluated")
        anova_fp = fp_res['anova_class'].get(g, "Not Evaluated")
        
        summary_rows.append({
            'Gene': g,
            'Detailed_Peptide_Counts': pep_string,
            'Volcano (MQ|FP)': f"{in_mq_volc} | {in_fp_volc}",
            'Fisher (MQ|FP)': f"{in_mq_fish} | {in_fp_fish}",
            'RF Top20 (MQ|FP)': f"{in_mq_rf} | {in_fp_rf}",
            'ANOVA Contaminant Check (MQ)': anova_mq,
            'ANOVA Contaminant Check (FP)': anova_fp
        })

    master_df = pd.DataFrame(summary_rows)
    out_file = OUT_DIR / "MASTER_CrossEngine_Relevant_Proteins.tsv"
    
    if not master_df.empty:
        master_df = master_df.sort_values(by='ANOVA Contaminant Check (MQ)')
        master_df.to_csv(out_file, sep='\t', index=False)
        print(master_df.to_string(index=False))
    else:
        print("No proteins passed the strict >=2 subsets criteria across BOTH engines.")
        with open(out_file, 'w') as f: f.write("No overlaps found.")
        
    print(f"\n✅ Master summary saved to: {out_file.name}")


if __name__ == "__main__":
    print("Loading data...")
    fp_all = get_all_fragpipe_datasets(use_cache=True, load_peptides=False)
    
    # Explicitly load FragPipe Run 6 with LB from disk to bypass stripped cache
    print(f"📂 Loading raw FragPipe {FRAGPIPE_WORKFLOW.upper()} Run 6 (With LB) directly from disk...")
    
    # Support both "IM-MS" and "imms" folder naming conventions
    if FRAGPIPE_WORKFLOW.lower() == 'imms':
        fp_lb_path = os.path.expanduser("~/results/LFQ_MBR_IM-MS/run6_results_with_LB/")
    else:
        fp_lb_path = os.path.expanduser(f"~/results/LFQ_MBR_{FRAGPIPE_WORKFLOW.upper()}/run6_results_with_LB/")
        
    fp_lb_annot = "../experiment_annotation/experiment_annotation_run6_R1344.tsv"
    
    try:
        fp_with_lb, _ = load_fragPipe_proteomic_data(fp_lb_path, fp_lb_annot)
        fp_with_lb = fp_with_lb[~fp_with_lb['Sample'].isin(['sample_14', 'sample_56'])].copy()
        extract_peptide_counts(fp_with_lb, 'fp', 'run6_with_LB')
    except Exception as e:
        print(f"  ⚠️ Failed to load FP Run 6 with LB from disk ({e}). Falling back to cached version.")
        fp_with_lb = fp_all.get(f"{FRAGPIPE_WORKFLOW}_run6_with_LB", {}).get("prot", pd.DataFrame())
    
    for k, d in fp_all.items(): 
        clean_run = k.replace(f"{FRAGPIPE_WORKFLOW}_", "").replace("_no_LB", "").replace("_with_LB", "")
        extract_peptide_counts(d["prot"], 'fp', clean_run)
    
    mq_all = get_maxquant_datasets()
    
    fp_sub = build_four_subsets(fp_all[f"{FRAGPIPE_WORKFLOW}_run4"]["prot"], fp_all[f"{FRAGPIPE_WORKFLOW}_run5"]["prot"], fp_all[f"{FRAGPIPE_WORKFLOW}_run6_no_LB"]["prot"])
    mq_sub = build_four_subsets(mq_all["run4_no_LB"], mq_all["run5_no_LB"], mq_all["run6_no_LB"])

    mq_results = run_engine_pipeline(mq_sub, mq_all["run6_with_LB"], "MaxQuant")
    fp_results = run_engine_pipeline(fp_sub, fp_with_lb, f"FragPipe_{FRAGPIPE_WORKFLOW.upper()}")

    if RUN_SUMMARY:
        compile_master_summary(mq_results, fp_results)
