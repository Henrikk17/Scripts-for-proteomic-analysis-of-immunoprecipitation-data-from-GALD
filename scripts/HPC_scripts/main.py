# 16/11-2026
# Master Proteomic Orchestrator: MaxQuant vs. FragPipe

import os
from pathlib import Path
import matplotlib
if os.environ.get('DISPLAY', '') == '': matplotlib.use('Agg')
else: matplotlib.use('TkAgg')

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Import from our compartmentalized modules
from helper_functions.manage_datasets import get_all_fragpipe_datasets
from helper_functions.load_fragPipe_proteomic_data import load_fragPipe_proteomic_data
from helper_functions.master_data_handlers import (
    global_pep_counts_mq, global_pep_counts_fp, extract_peptide_counts,
    get_maxquant_datasets, build_four_subsets, prep_wide_df, impute_perseus
)
from helper_functions.master_analytical_modules import (
    analyze_volcano, analyze_fisher, analyze_pca, analyze_rf, analyze_anova
)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

# ==========================================
# CONFIGURATION
# ==========================================
RUN_VOLCANO, RUN_FISHER, RUN_PCA, RUN_RF, RUN_ANOVA, RUN_SUMMARY = True, True, True, True, True, True

FRAGPIPE_WORKFLOW = "imms"
USE_FDR = True
P_CUTOFF = 0.05
LOG2FC_CUTOFF = 1.0

# Ensure your list of promising candidates is fully populated here
HIGHLIGHT_GENES = ["CP", "TFRC", "TFR2"]

OUT_DIR = Path("../plots/mq_vs_fragpipe_comparison")
CACHE_DIR = Path("../cache_master_pipeline")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# PIPELINE EXECUTION
# ==========================================
def run_engine_pipeline(subsets_dict, prot_with_lb, engine_name):
    print(f"\n{'='*70}\n🚀 RUNNING PIPELINE: {engine_name}\n{'='*70}")

    anova_classes = analyze_anova(prot_with_lb, engine_name, OUT_DIR) if RUN_ANOVA else {}
    pdf_volcano = PdfPages(OUT_DIR / f"Volcano_{engine_name}.pdf") if RUN_VOLCANO else None
    pdf_pca = PdfPages(OUT_DIR / f"PCA_{engine_name}.pdf") if RUN_PCA else None

    res_volcano, res_fisher, res_rf = {}, {}, {}

    for subset_name, (df_prot, g_case, g_ctrl) in subsets_dict.items():
        print(f"\n  ➤ Processing subset: {subset_name}")
        wide_df, c_samps, k_samps, s_to_g = prep_wide_df(df_prot, g_case, g_ctrl)
        imputed_df = impute_perseus(wide_df)

        if RUN_VOLCANO:
            det, genes, tops, hl_hits = analyze_volcano(imputed_df, c_samps, k_samps, subset_name, engine_name, pdf_volcano, USE_FDR, P_CUTOFF, LOG2FC_CUTOFF, HIGHLIGHT_GENES)
            for g, d in zip(genes, det): res_volcano.setdefault(g, []).append(d)
            if tops: print(f"    [Volcano Top 10] -> {', '.join(tops)}")
            if hl_hits: print(f"    [Candidate Hit!] -> {', '.join(hl_hits)} (Significant in Volcano)")

        if RUN_FISHER:
            det, genes, tops, hl_hits = analyze_fisher(wide_df, c_samps, k_samps, subset_name, P_CUTOFF, HIGHLIGHT_GENES)
            for g, d in zip(genes, det): res_fisher.setdefault(g, []).append(d)
            if tops: print(f"    [Fisher Top 10]  -> {', '.join(tops)}")
            if hl_hits: print(f"    [Candidate Hit!] -> {', '.join(hl_hits)} (Significant in Fisher)")

        if RUN_PCA:
            analyze_pca(imputed_df, s_to_g, subset_name, engine_name, pdf_pca)

        if RUN_RF:
            det, genes, hl_hits = analyze_rf(imputed_df, s_to_g, subset_name, HIGHLIGHT_GENES)
            for g, d in zip(genes, det): res_rf.setdefault(g, []).append(d)
            if hl_hits: print(f"    [Candidate Hit!] -> {', '.join(hl_hits)} (In Random Forest Top 20)")

    if RUN_VOLCANO: pdf_volcano.close()
    if RUN_PCA: pdf_pca.close()

    return {
        'anova_class': anova_classes,
        'volcano_sig': {g: hits for g, hits in res_volcano.items() if len(hits) >= 2},
        'fisher_sig': {g: hits for g, hits in res_fisher.items() if len(hits) >= 2},
        'rf_sig': {g: hits for g, hits in res_rf.items() if len(hits) >= 2},
    }

# ==========================================
# MASTER SUMMARY
# ==========================================
def compile_master_summary(mq_res, fp_res):
    print("\n" + "="*70 + "\n🏆 COMPILING MASTER CROSS-ENGINE SUMMARY\n" + "="*70)
    all_genes = set(mq_res['volcano_sig'].keys()) | set(mq_res['fisher_sig'].keys()) | set(mq_res['rf_sig'].keys()) | \
                set(fp_res['volcano_sig'].keys()) | set(fp_res['fisher_sig'].keys()) | set(fp_res['rf_sig'].keys())

    summary_rows = []
    for g in all_genes:
        mq_any = (g in mq_res['volcano_sig'] or g in mq_res['fisher_sig'] or g in mq_res['rf_sig'])
        fp_any = (g in fp_res['volcano_sig'] or g in fp_res['fisher_sig'] or g in fp_res['rf_sig'])
        if not (mq_any and fp_any): continue

        mq_p = [f"MQ {r}:{c}" for r, c in sorted(global_pep_counts_mq.get(g, {}).items())]
        fp_p = [f"FP {r}:{c}" for r, c in sorted(global_pep_counts_fp.get(g, {}).items())]

        summary_rows.append({
            'Gene': g,
            'Peptides': "[" + ", ".join(mq_p + fp_p) + "]",
            'Volcano(MQ|FP)': f"{'Yes' if g in mq_res['volcano_sig'] else '-'}|{'Yes' if g in fp_res['volcano_sig'] else '-'}",
            'Fisher(MQ|FP)': f"{'Yes' if g in mq_res['fisher_sig'] else '-'}|{'Yes' if g in fp_res['fisher_sig'] else '-'}",
            'RF_Top20(MQ|FP)': f"{'Yes' if g in mq_res['rf_sig'] else '-'}|{'Yes' if g in fp_res['rf_sig'] else '-'}",
            'ANOVA(MQ)': mq_res['anova_class'].get(g, "Not Evaluated"),
            'ANOVA(FP)': fp_res['anova_class'].get(g, "Not Evaluated")
        })

    master_df = pd.DataFrame(summary_rows)
    out_file = OUT_DIR / "MASTER_CrossEngine_Relevant_Proteins.tsv"

    if not master_df.empty:
        master_df = master_df.sort_values(by='ANOVA(MQ)')
        master_df.to_csv(out_file, sep='\t', index=False)
        print(master_df.to_string(index=False))
    else:
        print("No proteins passed the strict >=2 subsets criteria across BOTH engines.")

if __name__ == "__main__":
    print("Loading data...")
    fp_all = get_all_fragpipe_datasets(use_cache=True, load_peptides=False)

    print(f"📂 Loading raw FragPipe {FRAGPIPE_WORKFLOW.upper()} Run 6 (With LB) directly from disk...")
    fp_lb_path = os.path.expanduser("~/results/LFQ_MBR_IM-MS/run6_results_with_LB/") if FRAGPIPE_WORKFLOW.lower() == 'imms' else os.path.expanduser(f"~/results/LFQ_MBR_{FRAGPIPE_WORKFLOW.upper()}/run6_results_with_LB/")
    fp_lb_annot = "../experiment_annotation/experiment_annotation_run6_R1344.tsv"

    try:
        fp_with_lb, _ = load_fragPipe_proteomic_data(fp_lb_path, fp_lb_annot)
        fp_with_lb = fp_with_lb[~fp_with_lb['Sample'].isin(['sample_14', 'sample_56'])].copy()
        extract_peptide_counts(fp_with_lb, 'fp', 'run6_with_LB')
    except Exception:
        fp_with_lb = fp_all.get(f"{FRAGPIPE_WORKFLOW}_run6_with_LB", {}).get("prot", pd.DataFrame())


    for k, d in fp_all.items(): 
        # STRICT FILTER: Only extract peptides for the active workflow
        if k.startswith(FRAGPIPE_WORKFLOW):
            clean_run = k.replace(f"{FRAGPIPE_WORKFLOW}_", "").replace("_no_LB", "").replace("_with_LB", "")
            extract_peptide_counts(d["prot"], 'fp', clean_run)

    mq_all = get_maxquant_datasets(CACHE_DIR)

    fp_sub = build_four_subsets(fp_all[f"{FRAGPIPE_WORKFLOW}_run4"]["prot"], fp_all[f"{FRAGPIPE_WORKFLOW}_run5"]["prot"], fp_all[f"{FRAGPIPE_WORKFLOW}_run6_no_LB"]["prot"])
    mq_sub = build_four_subsets(mq_all["run4_no_LB"], mq_all["run5_no_LB"], mq_all["run6_no_LB"])

    mq_results = run_engine_pipeline(mq_sub, mq_all["run6_with_LB"], "MaxQuant")
    fp_results = run_engine_pipeline(fp_sub, fp_with_lb, f"FragPipe_{FRAGPIPE_WORKFLOW.upper()}")

    if RUN_SUMMARY: compile_master_summary(mq_results, fp_results)
