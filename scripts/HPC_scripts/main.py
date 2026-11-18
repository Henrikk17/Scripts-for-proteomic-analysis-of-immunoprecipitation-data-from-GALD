# 16/11-2026
# Master Proteomic Orchestrator: MaxQuant vs. FragPipe 

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Import from our compartmentalized modules
from helper_functions.master_data_handlers import (
    get_maxquant_datasets, get_fragpipe_imms_datasets, build_four_subsets, 
    prep_wide_df, impute_perseus, get_gald_peptide_interval
)
from helper_functions.master_analytical_modules import (
    analyze_volcano, analyze_fisher, analyze_pca, analyze_rf, analyze_background_ttest
)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

# ==========================================
# CONFIGURATION
# ==========================================
# Renamed RUN_ANOVA to RUN_BACKGROUND_TTEST
RUN_VOLCANO, RUN_FISHER, RUN_PCA, RUN_RF, RUN_BACKGROUND_TTEST, RUN_SUMMARY = True, True, True, True, True, True

USE_FDR = True       
P_CUTOFF = 0.05      
LOG2FC_CUTOFF = 1.0  
GLOBAL_SEED = 42

HIGHLIGHT_GENES = ["CP", "HP", "TFRC", "TFR2"]

OUT_DIR = Path("../plots/mq_vs_fragpipe_comparison")
CACHE_DIR = Path("../cache_master_pipeline")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# PIPELINE EXECUTION
# ==========================================
def run_engine_pipeline(subsets_dict, prot_with_lb, engine_name):
    print(f"\n{'='*70}\n🚀 RUNNING PIPELINE: {engine_name}\n{'='*70}")
    
    bg_classes = analyze_background_ttest(prot_with_lb, engine_name, OUT_DIR, GLOBAL_SEED, HIGHLIGHT_GENES) if RUN_BACKGROUND_TTEST else {}
    pdf_volcano = PdfPages(OUT_DIR / f"Volcano_{engine_name}.pdf") if RUN_VOLCANO else None
    pdf_pca = PdfPages(OUT_DIR / f"PCA_{engine_name}.pdf") if RUN_PCA else None
    
    res_volcano, res_fisher, res_rf = {}, {}, {}
    engine_enriched_records = []
    
    for subset_name, (df_prot, g_case, g_ctrl) in subsets_dict.items():
        print(f"\n  ➤ Processing subset: {subset_name}")
        wide_df, c_samps, k_samps, s_to_g = prep_wide_df(df_prot, g_case, g_ctrl)
        imputed_df = impute_perseus(wide_df, GLOBAL_SEED)
        
        if RUN_VOLCANO:
            det, sig_df, tops, hl_hits = analyze_volcano(imputed_df, c_samps, k_samps, subset_name, engine_name, pdf_volcano, USE_FDR, P_CUTOFF, LOG2FC_CUTOFF, HIGHLIGHT_GENES)
            genes = sig_df['Gene'].tolist()
            for g, d in zip(genes, det): res_volcano.setdefault(g, []).append(d)
            if tops: print(f"    [{engine_name} Volcano Top 10] -> {', '.join(tops)}")
            if hl_hits: print(f"      * Candidate Hit(s): {', '.join(hl_hits)} (Volcano)")
            
            if not sig_df.empty:
                v_out = sig_df.copy()
                v_out['Subset'] = subset_name
                v_out['Analysis'] = 'Volcano'
                engine_enriched_records.append(v_out)

        if RUN_FISHER:
            det, sig_df, tops, hl_hits = analyze_fisher(wide_df, c_samps, k_samps, subset_name, P_CUTOFF, HIGHLIGHT_GENES)
            genes = sig_df['Gene'].tolist()
            for g, d in zip(genes, det): res_fisher.setdefault(g, []).append(d)
            if tops: print(f"    [{engine_name} Fisher Top 10]  -> {', '.join(tops)}")
            if hl_hits: print(f"      * Candidate Hit(s): {', '.join(hl_hits)} (Fisher)")
            
            if not sig_df.empty:
                f_out = sig_df.copy()
                f_out['Subset'] = subset_name
                f_out['Analysis'] = 'Fisher'
                engine_enriched_records.append(f_out)

        if RUN_PCA: 
            analyze_pca(imputed_df, s_to_g, subset_name, engine_name, pdf_pca, GLOBAL_SEED)
        
        if RUN_RF:
            det, genes, hl_hits = analyze_rf(imputed_df, s_to_g, subset_name, HIGHLIGHT_GENES, GLOBAL_SEED)
            for g, d in zip(genes, det): res_rf.setdefault(g, []).append(d)
            if genes: print(f"    [{engine_name} RF Top 20]      -> {', '.join(genes)}")
            if hl_hits: print(f"      * Candidate Hit(s): {', '.join(hl_hits)} (Random Forest)")

    if RUN_VOLCANO: pdf_volcano.close()
    if RUN_PCA: pdf_pca.close()

    # Compile and export all enriched subsets for this engine
    if engine_enriched_records:
        df_out = pd.concat(engine_enriched_records, ignore_index=True)
        col_order = ['Subset', 'Analysis', 'Gene', 'Log2FC', 'p_value', 'FDR', 'c_pres', 'k_pres']
        ordered_cols = [c for c in col_order if c in df_out.columns]
        
        # Sort by Subset -> Analysis -> Most Significant p_value
        df_out = df_out[ordered_cols].sort_values(by=['Subset', 'Analysis', 'p_value'])
        df_out.to_csv(OUT_DIR / f"{engine_name}_Enriched_Proteins.txt", sep='\t', index=False)

    return {
        'bg_class': bg_classes,
        'volcano_hits': res_volcano,
        'fisher_hits': res_fisher,
        'rf_hits': res_rf,
    }

# ==========================================
# MASTER SUMMARY
# ==========================================
def compile_master_summary(mq_res, fp_res, mq_all, fp_all):
    print("\n" + "="*70 + "\n🏆 COMPILING MASTER CROSS-ENGINE SUMMARY (Proteins passing >=2/4 subsets in BOTH Engines)\n" + "="*70)
    
    def get_passed_genes(res):
        passed = set()
        all_g = set(res['volcano_hits'].keys()) | set(res['fisher_hits'].keys()) | set(res['rf_hits'].keys())
        for g in all_g:
            if len(res['volcano_hits'].get(g, [])) >= 2 or len(res['fisher_hits'].get(g, [])) >= 2 or len(res['rf_hits'].get(g, [])) >= 2:
                passed.add(g)
        return passed
        
    mq_passed = get_passed_genes(mq_res)
    fp_passed = get_passed_genes(fp_res)
    valid_genes = mq_passed.intersection(fp_passed)
                
    summary_rows = []
    for g in valid_genes:
        v_mq = len(mq_res['volcano_hits'].get(g, []))
        v_fp = len(fp_res['volcano_hits'].get(g, []))
        f_mq = len(mq_res['fisher_hits'].get(g, []))
        f_fp = len(fp_res['fisher_hits'].get(g, []))
        r_mq = len(mq_res['rf_hits'].get(g, []))
        r_fp = len(fp_res['rf_hits'].get(g, []))
        
        fmt = lambda x: str(x) if x > 0 else "-"

        r4_mq = get_gald_peptide_interval(mq_all['run4_no_LB'].get('pep'), g)
        r4_fp = get_gald_peptide_interval(fp_all['run4'].get('pep'), g)
        
        r5_mq = get_gald_peptide_interval(mq_all['run5_no_LB'].get('pep'), g)
        r5_fp = get_gald_peptide_interval(fp_all['run5'].get('pep'), g)
        
        r6_mq = get_gald_peptide_interval(mq_all['run6_with_LB'].get('pep'), g)
        r6_fp = get_gald_peptide_interval(fp_all['run6_with_LB'].get('pep'), g)
        
        pep_string = f"[MQ/FP run 4: {r4_mq}/{r4_fp}, MQ/FP run 5: {r5_mq}/{r5_fp}, run 6: {r6_mq}/{r6_fp}]"
        
        summary_rows.append({
            'Gene': g,
            'GALD Peptide Interval': pep_string,
            'Volcano(MQ|FP)': f"{fmt(v_mq)}|{fmt(v_fp)}",
            'Fisher(MQ|FP)': f"{fmt(f_mq)}|{fmt(f_fp)}",
            'RF_Top20(MQ|FP)': f"{fmt(r_mq)}|{fmt(r_fp)}",
            'Background_TTest(MQ)': mq_res['bg_class'].get(g, "Not Evaluated"),
            'Background_TTest(FP)': fp_res['bg_class'].get(g, "Not Evaluated")
        })

    master_df = pd.DataFrame(summary_rows)
    out_file = OUT_DIR / "MASTER_CrossEngine_Relevant_Proteins.tsv"
    
    if not master_df.empty:
        # Sort by the newly named background class column
        master_df = master_df.sort_values(by='Background_TTest(MQ)')
        master_df.to_csv(out_file, sep='\t', index=False)
        print(master_df.to_string(index=False))
    else:
        print("No proteins passed the strict >=2 subsets criteria across BOTH engines.")

if __name__ == "__main__":
    print("Loading ONLY IM-MS FragPipe and MaxQuant data...")
    
    fp_all = get_fragpipe_imms_datasets(CACHE_DIR)
    mq_all = get_maxquant_datasets(CACHE_DIR)
    
    fp_sub = build_four_subsets(fp_all["run4"]["prot"], fp_all["run5"]["prot"], fp_all["run6_no_LB"]["prot"])
    mq_sub = build_four_subsets(mq_all["run4_no_LB"]["prot"], mq_all["run5_no_LB"]["prot"], mq_all["run6_no_LB"]["prot"])

    mq_results = run_engine_pipeline(mq_sub, mq_all["run6_with_LB"]["prot"], "MaxQuant")
    fp_results = run_engine_pipeline(fp_sub, fp_all["run6_with_LB"]["prot"], "FragPipe_IMMS")

    if RUN_SUMMARY: 
        compile_master_summary(mq_results, fp_results, mq_all, fp_all)
        
    print(f"\n📁 All plots successfully generated and saved to: {OUT_DIR.resolve()}")