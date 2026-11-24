# 16/11-2026
# Master Proteomic Orchestrator: MaxQuant vs. FragPipe

from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Strictly force headless mode to prevent Tkinter core dumps

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Import from our compartmentalized modules
from helper_functions.master_data_handlers import (
    get_maxquant_datasets, get_fragpipe_imms_datasets, build_four_subsets,
    prep_wide_df, impute_perseus, get_gald_peptide_interval
)
from helper_functions.master_analytical_modules import (
    analyze_volcano, analyze_fisher,
    analyze_pca, analyze_rf, analyze_background_ttest
)
from helper_functions.load_fragPipe_proteomic_data import load_fragPipe_proteomic_data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

# ==========================================
# CONFIGURATION
# ==========================================
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

# Dedicated no-MBR FragPipe run 6 dataset (used ONLY for LB vs Liver background check)
FP_NO_MBR_R6_DIR = Path("/home/local/s215065/results/LFQ_no_MBR_IM-MS/run6_with_LB")
RUN6_ANNOT_FILE = Path("/home/local/s215065/data_analysis/experiment_annotation/experiment_annotation_run6_R1344.tsv")

EXCLUDED_SAMPLES = ['sample_14', 'sample_56']


def _exclude_lb_samples(pep_df):
    """Return a copy of peptide detections excluding Lysate Buffer (LB) negative controls."""
    if pep_df is None or pep_df.empty or 'Sample' not in pep_df.columns:
        return pep_df

    df_filt = pep_df.copy()
    sample_str = df_filt['Sample'].astype(str).str.strip().str.lower()
    name_lb = sample_str.str.contains(r'(?:^|_)lb$|buffer', regex=True, na=False)

    s_num = pd.to_numeric(
        sample_str.str.replace('sample_', '', regex=False).str.replace('sample', '', regex=False).str.strip(),
        errors='coerce'
    )
    num_lb = s_num.between(43, 84, inclusive='both').fillna(False)

    if 'Sample_type' in df_filt.columns:
        stype = df_filt['Sample_type'].astype(str).str.strip().str.lower()
        type_lb = stype.str.contains(r'\blb\b|buffer', regex=True, na=False)
        lb_mask = type_lb | name_lb
    else:
        lb_mask = name_lb | num_lb

    return df_filt[~lb_mask].copy()


def _load_fp_no_mbr_r6_prot():
    """Load run 6 no-MBR FragPipe protein detections strictly for background testing."""
    if not FP_NO_MBR_R6_DIR.exists():
        print(f"⚠️ Warning: No-MBR run 6 directory not found: {FP_NO_MBR_R6_DIR}")
        return None
    if not RUN6_ANNOT_FILE.exists():
        print(f"⚠️ Warning: Run 6 annotation file not found: {RUN6_ANNOT_FILE}")
        return None

    prot_no_mbr, _ = load_fragPipe_proteomic_data(FP_NO_MBR_R6_DIR, RUN6_ANNOT_FILE)
    if prot_no_mbr is not None and 'Sample' in prot_no_mbr.columns:
        prot_no_mbr = prot_no_mbr[~prot_no_mbr['Sample'].isin(EXCLUDED_SAMPLES)].copy()
    return prot_no_mbr


# ==========================================
# PIPELINE EXECUTION
# ==========================================
def run_engine_pipeline(subsets_dict, prot_with_lb, engine_name, bg_prot_override=None, bg_label_override=None):
    print(f"\n{'='*70}\n🚀 RUNNING PIPELINE: {engine_name}\n{'='*70}")

    bg_source = bg_prot_override if bg_prot_override is not None else prot_with_lb
    bg_label = bg_label_override if bg_label_override is not None else engine_name
    bg_classes = (
        analyze_background_ttest(bg_source, bg_label, OUT_DIR, HIGHLIGHT_GENES)
        if RUN_BACKGROUND_TTEST else {}
    )
    pdf_volcano = PdfPages(OUT_DIR / f"Volcano_{engine_name}.pdf") if RUN_VOLCANO else None
    pdf_pca = PdfPages(OUT_DIR / f"PCA_{engine_name}.pdf") if RUN_PCA else None

    res_volcano, res_fisher, res_rf = {}, {}, {}
    res_fisher_above_avg, res_highint_exclusive = {}, {}
    engine_enriched_records = []
    engine_watchlist_records = []

    for subset_name, (df_prot, g_case, g_ctrl) in subsets_dict.items():
        print(f"\n  ➤ Processing subset: {subset_name}")
        wide_quant, wide_pres, c_samps, k_samps, s_to_g = prep_wide_df(df_prot, g_case, g_ctrl)
        imputed_df = impute_perseus(wide_quant, GLOBAL_SEED)
        volcano_tested_genes = set(wide_quant.index)

        volcano_filtered_fisher_callout = []

        if RUN_FISHER:
            det, sig_df, tops, hl_hits, wl_df = analyze_fisher(
                wide_pres, c_samps, k_samps, subset_name, P_CUTOFF, HIGHLIGHT_GENES,
                volcano_genes=volcano_tested_genes,
            )
            genes = sig_df['Gene'].tolist() if not sig_df.empty else []
            for g, d in zip(genes, det):
                res_fisher.setdefault(g, []).append(d)
            if tops:
                print(f"    [{engine_name} Fisher Top 10]  -> {', '.join(tops)}")
            if hl_hits:
                print(f"      * Candidate Hit(s): {', '.join(hl_hits)} (Fisher)")

            if not sig_df.empty:
                above_avg_df = sig_df[sig_df['Above_Avg_GALD_Int']]
                if not above_avg_df.empty:
                    subset_avg_val = above_avg_df['Subset_GALD_Avg_Log2Int'].iloc[0]
                    tags = []
                    for _, r in above_avg_df.iterrows():
                        g = r['Gene']
                        res_fisher_above_avg.setdefault(g, []).append(subset_name)
                        if r['Non-GALD_sample_presence'] <= 1:
                            res_highint_exclusive.setdefault(g, set()).add(subset_name)
                        if r['Volcano_Filtered_Out']:
                            volcano_filtered_fisher_callout.append(
                                f"{g} ({int(r['GALD_sample_presence'])}/{len(c_samps)} vs {int(r['Non-GALD_sample_presence'])}/{len(k_samps)})"
                            )
                        vf_flag = ", Volcano-Filtered" if r['Volcano_Filtered_Out'] else ""
                        tags.append(
                            f"{g} ({int(r['GALD_sample_presence'])}/{len(c_samps)} vs {int(r['Non-GALD_sample_presence'])}/{len(k_samps)}, "
                            f"log2={r['GALD_Mean_Log2Int']:.1f}{vf_flag})"
                        )
                    print(
                        f"      * Fisher Above-Avg GALD Intensity (>{subset_avg_val:.1f} log2, n={len(above_avg_df)}): "
                        + ", ".join(tags[:10])
                        + (f" (+{len(tags) - 10} more)" if len(tags) > 10 else "")
                    )

                f_out = sig_df.copy()
                f_out['Subset'] = subset_name
                f_out['Analysis'] = 'Fisher'
                f_out['Log2FC'] = pd.Series(dtype='float64', index=f_out.index)
                engine_enriched_records.append(f_out)

            if not wl_df.empty:
                wl_tags = [
                    f"{r['Gene']} ({int(r['GALD_sample_presence'])}/{len(c_samps)} vs {int(r['Non-GALD_sample_presence'])}/{len(k_samps)}, "
                    f"p={r['p_value']:.3f}, log2={r['GALD_Mean_Log2Int']:.1f})"
                    for _, r in wl_df.iterrows()
                ]
                print(
                    f"      * [Watchlist] Volcano-filtered, <=1 Ctrl, Above-Avg GALD Int (0.05<=p<0.10): "
                    + ", ".join(wl_tags[:10])
                    + (f" (+{len(wl_tags) - 10} more)" if len(wl_tags) > 10 else "")
                )
                wl_out = wl_df.copy()
                wl_out['Subset'] = subset_name
                engine_watchlist_records.append(wl_out)

        if RUN_VOLCANO:
            det, sig_df, tops, hl_hits = analyze_volcano(
                imputed_df, c_samps, k_samps, subset_name, engine_name,
                pdf_volcano, USE_FDR, P_CUTOFF, LOG2FC_CUTOFF, HIGHLIGHT_GENES,
                wide_pres=wide_pres,
                volcano_filtered_fisher_hits=volcano_filtered_fisher_callout,
            )
            genes = sig_df['Gene'].tolist() if not sig_df.empty else []
            for g, d in zip(genes, det):
                res_volcano.setdefault(g, []).append(d)
            if tops:
                print(f"    [{engine_name} Volcano Top 10] -> {', '.join(tops)}")
            if hl_hits:
                print(f"      * Candidate Hit(s): {', '.join(hl_hits)} (Volcano)")

            if not sig_df.empty:
                if 'Non-GALD_sample_presence' in sig_df.columns and 'Above_Avg_GALD_Int' in sig_df.columns:
                    v_excl = sig_df[
                        (sig_df['Non-GALD_sample_presence'] <= 1) & (sig_df['Above_Avg_GALD_Int'].fillna(False))
                    ]
                    for g in v_excl['Gene']:
                        res_highint_exclusive.setdefault(g, set()).add(subset_name)

                v_out = sig_df.copy()
                v_out['Subset'] = subset_name
                v_out['Analysis'] = 'Volcano'
                engine_enriched_records.append(v_out)

        if RUN_PCA:
            analyze_pca(imputed_df, s_to_g, subset_name, engine_name, pdf_pca, GLOBAL_SEED)

        if RUN_RF:
            det, genes, hl_hits = analyze_rf(imputed_df, s_to_g, subset_name, HIGHLIGHT_GENES, GLOBAL_SEED)
            for g, d in zip(genes, det):
                res_rf.setdefault(g, []).append(d)
            if genes:
                print(f"    [{engine_name} RF Top 20]      -> {', '.join(genes)}")
            if hl_hits:
                print(f"      * Candidate Hit(s): {', '.join(hl_hits)} (Random Forest)")

    if RUN_VOLCANO:
        pdf_volcano.close()
    if RUN_PCA:
        pdf_pca.close()

    # Compile and export all enriched subsets for this engine
    non_empty_records = [df_rec for df_rec in engine_enriched_records if not df_rec.empty]
    if non_empty_records:
        df_out = pd.concat(non_empty_records, ignore_index=True)
        col_order = [
            'Subset', 'Analysis', 'Gene', 'Log2FC', 'p_value', 'FDR',
            'GALD_sample_presence', 'Non-GALD_sample_presence',
            'GALD_Mean_Log2Int', 'Subset_GALD_Avg_Log2Int', 'Above_Avg_GALD_Int',
            'Exclusivity', 'Volcano_Filtered_Out',
        ]
        ordered_cols = [c for c in col_order if c in df_out.columns]

        for num_col in ['Log2FC', 'p_value', 'FDR', 'GALD_Mean_Log2Int', 'Subset_GALD_Avg_Log2Int']:
            if num_col in df_out.columns:
                df_out[num_col] = pd.to_numeric(df_out[num_col], errors='coerce')

        for pres_col in ['GALD_sample_presence', 'Non-GALD_sample_presence']:
            if pres_col in df_out.columns:
                df_out[pres_col] = pd.to_numeric(df_out[pres_col], errors='coerce').astype('Int64')

        sort_cols = [c for c in ['Subset', 'Analysis', 'p_value', 'GALD_Mean_Log2Int'] if c in df_out.columns]
        sort_asc = [False if c == 'GALD_Mean_Log2Int' else True for c in sort_cols]
        df_out = df_out[ordered_cols].sort_values(by=sort_cols, ascending=sort_asc, na_position='last')
        df_out.to_csv(OUT_DIR / f"{engine_name}_Enriched_Proteins.txt", sep='\t', index=False)

    # Export Volcano-filtered borderline exclusive watchlist (safeguard)
    non_empty_wl = [df_wl for df_wl in engine_watchlist_records if not df_wl.empty]
    if non_empty_wl:
        wl_all = pd.concat(non_empty_wl, ignore_index=True)
        wl_cols = [
            'Subset', 'Gene', 'p_value', 'GALD_sample_presence', 'Non-GALD_sample_presence',
            'GALD_Mean_Log2Int', 'Subset_GALD_Avg_Log2Int', 'Above_Avg_GALD_Int',
            'Exclusivity', 'Volcano_Filtered_Out',
        ]
        wl_all = wl_all[[c for c in wl_cols if c in wl_all.columns]].sort_values(
            by=['Subset', 'p_value', 'GALD_Mean_Log2Int'], ascending=[True, True, False], na_position='last'
        )
        wl_all.to_csv(OUT_DIR / f"{engine_name}_VolcanoFiltered_Exclusive_Watchlist.tsv", sep='\t', index=False)

    return {
        'bg_class': bg_classes,
        'volcano_hits': res_volcano,
        'fisher_hits': res_fisher,
        'fisher_above_avg': res_fisher_above_avg,
        'highint_exclusive': {g: sorted(subs) for g, subs in res_highint_exclusive.items()},
        'rf_hits': res_rf,
    }


# ==========================================
# MASTER SUMMARY
# ==========================================
def compile_master_summary(mq_res, fp_res, mq_all, fp_all):
    print(
        "\n" + "=" * 70
        + "\n🏆 COMPILING MASTER CROSS-ENGINE SUMMARY (Significant in >=2 independent runs in BOTH Engines: Volcano BH-FDR or Fisher Raw-p)\n"
        + "=" * 70
    )

    def get_sig_runs(res, gene):
        hits = res['volcano_hits'].get(gene, []) + res['fisher_hits'].get(gene, [])
        return sorted({h.split('_')[0] for h in hits if '_' in h})

    def get_passed_genes(res):
        all_g = set(res['volcano_hits'].keys()) | set(res['fisher_hits'].keys())
        return {g for g in all_g if len(get_sig_runs(res, g)) >= 2}

    mq_passed = get_passed_genes(mq_res)
    fp_passed = get_passed_genes(fp_res)
    valid_genes = sorted(mq_passed.intersection(fp_passed))

    r6_mq_pep = _exclude_lb_samples(mq_all['run6_with_LB'].get('pep'))
    r6_fp_pep = _exclude_lb_samples(fp_all['run6_with_LB'].get('pep'))

    summary_rows = []
    for g in valid_genes:
        v_mq = len(mq_res['volcano_hits'].get(g, []))
        v_fp = len(fp_res['volcano_hits'].get(g, []))
        f_mq = len(mq_res['fisher_hits'].get(g, []))
        f_fp = len(fp_res['fisher_hits'].get(g, []))
        fa_mq = len(mq_res.get('fisher_above_avg', {}).get(g, []))
        fa_fp = len(fp_res.get('fisher_above_avg', {}).get(g, []))
        r_mq = len(mq_res['rf_hits'].get(g, []))
        r_fp = len(fp_res['rf_hits'].get(g, []))

        mq_runs = ",".join(get_sig_runs(mq_res, g))
        fp_runs = ",".join(get_sig_runs(fp_res, g))

        excl_mq = ",".join(sorted({s.split('_')[0] for s in mq_res.get('highint_exclusive', {}).get(g, [])})) or "-"
        excl_fp = ",".join(sorted({s.split('_')[0] for s in fp_res.get('highint_exclusive', {}).get(g, [])})) or "-"

        fmt = lambda x: str(x) if x > 0 else "-"

        r4_mq = get_gald_peptide_interval(mq_all['run4_no_LB'].get('pep'), g)
        r4_fp = get_gald_peptide_interval(fp_all['run4'].get('pep'), g)

        r5_mq = get_gald_peptide_interval(mq_all['run5_no_LB'].get('pep'), g)
        r5_fp = get_gald_peptide_interval(fp_all['run5'].get('pep'), g)

        r6_mq = get_gald_peptide_interval(r6_mq_pep, g)
        r6_fp = get_gald_peptide_interval(r6_fp_pep, g)

        pep_string = f"[MQ/FP run 4: {r4_mq}/{r4_fp}, MQ/FP run 5: {r5_mq}/{r5_fp}, MQ/FP run 6: {r6_mq}/{r6_fp}]"

        summary_rows.append({
            'Gene': g,
            'Independent_Runs(MQ|FP)': f"{mq_runs}|{fp_runs}",
            'GALD Peptide Interval': pep_string,
            'Volcano_FDR(MQ|FP)': f"{fmt(v_mq)}|{fmt(v_fp)}",
            'Fisher_RawP(MQ|FP)': f"{fmt(f_mq)}|{fmt(f_fp)}",
            'Fisher_AboveAvgInt(MQ|FP)': f"{fmt(fa_mq)}|{fmt(fa_fp)}",
            'HighInt_GALD_Exclusive(MQ|FP)': f"{excl_mq}|{excl_fp}",
            'RF_Top20_Support(MQ|FP)': f"{fmt(r_mq)}|{fmt(r_fp)}",
            'Background_TTest(MQ)': mq_res['bg_class'].get(g, "Not Evaluated"),
            'Background_TTest(FP)': fp_res['bg_class'].get(g, "Not Evaluated"),
        })

    master_df = pd.DataFrame(summary_rows)
    out_file = OUT_DIR / "MASTER_CrossEngine_Relevant_Proteins.tsv"

    if not master_df.empty:
        master_df = master_df.sort_values(by=['Background_TTest(MQ)', 'Gene'])
        master_df.to_csv(out_file, sep='\t', index=False)
        print(master_df.to_string(index=False))
    else:
        print("No proteins passed significance across >=2 independent runs in BOTH engines.")


if __name__ == "__main__":
    print("Loading ONLY IM-MS FragPipe and MaxQuant data...")

    fp_all = get_fragpipe_imms_datasets(CACHE_DIR)
    mq_all = get_maxquant_datasets(CACHE_DIR)

    # Dedicated no-MBR run 6 data used strictly for the LB vs Liver background check
    fp_no_mbr_r6_prot = _load_fp_no_mbr_r6_prot()

    fp_sub = build_four_subsets(fp_all["run4"]["prot"], fp_all["run5"]["prot"], fp_all["run6_no_LB"]["prot"])
    mq_sub = build_four_subsets(mq_all["run4_no_LB"]["prot"], mq_all["run5_no_LB"]["prot"], mq_all["run6_no_LB"]["prot"])

    # Run sequentially to avoid Matplotlib & Scikit-Learn deadlocks
    mq_results = run_engine_pipeline(mq_sub, mq_all["run6_with_LB"]["prot"], "MaxQuant")
    fp_results = run_engine_pipeline(
        fp_sub,
        fp_all["run6_with_LB"]["prot"],
        "FragPipe_IMMS",
        bg_prot_override=fp_no_mbr_r6_prot,
        bg_label_override="FragPipe_IMMS_noMBR_R6"
    )

    if RUN_SUMMARY:
        compile_master_summary(mq_results, fp_results, mq_all, fp_all)

    print(f"\n📁 All plots successfully generated and saved to: {OUT_DIR.resolve()}")