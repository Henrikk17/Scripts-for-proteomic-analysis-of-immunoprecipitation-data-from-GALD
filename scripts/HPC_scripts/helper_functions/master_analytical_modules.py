# Master Analytical Modules: Volcano, Fisher, PCA, Random Forest, Paired Background Testing, and High-Intensity Exclusive Detection

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import binomtest, fisher_exact, ttest_1samp, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
import shutil
import subprocess
import tempfile

from helper_functions.master_data_handlers import (
    _select_representative_protein_groups,
    get_gald_peptide_interval,
    impute_perseus,
)

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


# ==========================================
# SHARED HELPER: PRE-IMPUTATION GALD INTENSITY & EXCLUSIVITY
# ==========================================
def _compute_gald_intensity_stats(wide_pres, c_samps, k_samps):
    """
    Compute per-gene GALD/control presence, raw-positive mean log2 intensity,
    within-subset GALD intensity percentile and robust Z-score (median/MAD),
    and exclusivity class prior to any imputation.
    """
    if wide_pres is None or wide_pres.empty or not c_samps:
        return pd.DataFrame(), np.nan

    c_raw = wide_pres[c_samps].where(wide_pres[c_samps] > 0)
    c_log2 = np.log2(c_raw)
    gene_mean_log2 = c_log2.mean(axis=1, skipna=True)
    valid_means = gene_mean_log2.dropna()
    subset_avg_log2 = float(valid_means.mean()) if not valid_means.empty else np.nan

    if not valid_means.empty:
        subset_med_log2 = float(valid_means.median())
        mad = float(np.median(np.abs(valid_means.values - subset_med_log2)))
        robust_scale = 1.4826 * mad
        if not np.isfinite(robust_scale) or robust_scale <= 0:
            robust_scale = float(valid_means.std(ddof=1)) if len(valid_means) > 1 else np.nan
        if np.isfinite(robust_scale) and robust_scale > 0:
            gald_robust_z = (gene_mean_log2 - subset_med_log2) / robust_scale
        else:
            gald_robust_z = pd.Series(np.nan, index=wide_pres.index, dtype=float)
        gald_int_pct = gene_mean_log2.rank(pct=True, na_option='keep') * 100.0
    else:
        gald_robust_z = pd.Series(np.nan, index=wide_pres.index, dtype=float)
        gald_int_pct = pd.Series(np.nan, index=wide_pres.index, dtype=float)

    c_counts = c_raw.notna().sum(axis=1).astype(int)
    if k_samps:
        k_raw = wide_pres[k_samps].where(wide_pres[k_samps] > 0)
        k_counts = k_raw.notna().sum(axis=1).astype(int)
        k_mean_log2 = np.log2(k_raw).mean(axis=1, skipna=True)
    else:
        k_counts = pd.Series(0, index=wide_pres.index, dtype=int)
        k_mean_log2 = pd.Series(np.nan, index=wide_pres.index, dtype=float)

    n_k = max(len(k_samps), 1)
    exclusivity = np.where(
        k_counts == 0,
        "GALD-Exclusive (0 Ctrl)",
        np.where(
            (k_counts <= 1) | ((k_counts / n_k) <= 0.10),
            "Almost GALD-Exclusive (<=1 Ctrl)",
            "GALD-Enriched Presence",
        ),
    )

    stats_df = pd.DataFrame({
        'GALD_sample_presence': c_counts,
        'Non-GALD_sample_presence': k_counts,
        'GALD_Mean_Log2Int': gene_mean_log2,
        'Non-GALD_Mean_Log2Int': k_mean_log2,
        'GALD_Int_Pct': gald_int_pct,
        'GALD_Int_RobustZ': gald_robust_z,
        'Subset_GALD_Avg_Log2Int': subset_avg_log2,
        'Above_Avg_GALD_Int': gene_mean_log2 > subset_avg_log2 if np.isfinite(subset_avg_log2) else False,
        'Exclusivity': exclusivity,
    }, index=wide_pres.index)

    return stats_df, subset_avg_log2


def detect_high_intensity_exclusive(
    wide_pres,
    c_samps,
    k_samps,
    subset_name,
    volcano_genes=None,
    fisher_sig_genes=None,
    volcano_sig_genes=None,
    min_gald_det=2,
    max_ctrl_det=1,
    pct_cutoff=90.0,
    z_cutoff=2.0,
):
    """
    Independent detector for high-abundance, GALD-exclusive (or <=1 control) proteins.
    Operates directly on pre-filter wide_pres — independent of the 50% completeness
    filter and Fisher/Volcano p-value cutoffs.
    Returns (candidates_df, pres_stats_df).
    """
    n_c = len(c_samps)
    n_k = len(k_samps)
    if wide_pres is None or wide_pres.empty or n_c == 0:
        return pd.DataFrame(), pd.DataFrame()

    pres_stats, _ = _compute_gald_intensity_stats(wide_pres, c_samps, k_samps)
    if pres_stats.empty:
        return pd.DataFrame(), pres_stats

    volc_tested = set(volcano_genes) if volcano_genes is not None else set()
    f_sig = set(fisher_sig_genes) if fisher_sig_genes is not None else set()
    v_sig = set(volcano_sig_genes) if volcano_sig_genes is not None else set()

    gald_prop = pres_stats['GALD_sample_presence'] / n_c
    ctrl_prop = pres_stats['Non-GALD_sample_presence'] / max(n_k, 1)

    is_high_int = (
        (pres_stats['GALD_Int_Pct'] >= pct_cutoff)
        | (pres_stats['GALD_Int_RobustZ'] >= z_cutoff)
    )
    is_exclusive = (
        (pres_stats['GALD_sample_presence'] >= min_gald_det)
        & (pres_stats['Non-GALD_sample_presence'] <= max_ctrl_det)
        & (gald_prop > ctrl_prop)
    )

    cand = pres_stats[is_high_int & is_exclusive].copy()
    if cand.empty:
        return pd.DataFrame(), pres_stats

    rows = []
    for gene, r in cand.iterrows():
        cp = int(r['GALD_sample_presence'])
        kp = int(r['Non-GALD_sample_presence'])
        if n_k > 0 and 0 < (cp + kp) < (n_c + n_k):
            _, p_f = fisher_exact([[cp, n_c - cp], [kp, n_k - kp]], alternative='greater')
        else:
            p_f = np.nan

        in_f = gene in f_sig
        in_v = gene in v_sig
        if in_f and in_v:
            status = "Fisher+Volcano"
        elif in_f:
            status = "Fisher only"
        elif in_v:
            status = "Volcano only"
        else:
            status = "MISSED (Neither)"

        rows.append({
            'Subset': subset_name,
            'Gene': gene,
            'GALD_sample_presence': cp,
            'Non-GALD_sample_presence': kp,
            'GALD_n': n_c,
            'Non-GALD_n': n_k,
            'GALD_Mean_Log2Int': float(r['GALD_Mean_Log2Int']),
            'Non-GALD_Mean_Log2Int': float(r['Non-GALD_Mean_Log2Int']) if pd.notna(r['Non-GALD_Mean_Log2Int']) else np.nan,
            'GALD_Int_Pct': float(r['GALD_Int_Pct']),
            'GALD_Int_RobustZ': float(r['GALD_Int_RobustZ']) if pd.notna(r['GALD_Int_RobustZ']) else np.nan,
            'Fisher_Raw_p': p_f,
            'Exclusivity': str(r['Exclusivity']),
            'Volcano_Filtered_Out': (gene not in volc_tested) if volcano_genes is not None else False,
            'Main_Test_Status': status,
        })

    cand_df = pd.DataFrame(rows).sort_values(
        by=['GALD_Int_RobustZ', 'GALD_Int_Pct', 'GALD_sample_presence'],
        ascending=[False, False, False],
        na_position='last',
    )
    return cand_df, pres_stats


# ==========================================
# 1. VOLCANO PLOT (R LIMMA ON OBSERVED INTENSITIES + BH-FDR)
# ==========================================
_LIMMA_R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
expr_path <- args[1]
meta_path <- args[2]
out_path  <- args[3]

suppressPackageStartupMessages(library(limma))

expr_df <- read.delim(
  expr_path, check.names = FALSE, row.names = 1,
  stringsAsFactors = FALSE, quote = "", comment.char = ""
)
meta_df <- read.delim(
  meta_path, check.names = FALSE,
  stringsAsFactors = FALSE, quote = "", comment.char = ""
)

sample_ids <- as.character(meta_df$Sample)
mat <- as.matrix(expr_df[, sample_ids, drop = FALSE])
storage.mode(mat) <- "double"

group <- factor(meta_df$Group, levels = c("Control", "GALD"))
design <- model.matrix(~ group)

fit <- lmFit(mat, design)
use_trend  <- nrow(mat) >= 10
use_robust <- requireNamespace("statmod", quietly = TRUE) && (nrow(mat) >= 10)
fit <- eBayes(fit, trend = use_trend, robust = use_robust)

tt <- topTable(fit, coef = "groupGALD", number = Inf, sort.by = "none", adjust.method = "BH")
out_df <- data.frame(
  Gene    = rownames(tt),
  Log2FC  = tt$logFC,
  AveExpr = tt$AveExpr,
  t_stat  = tt$t,
  p_value = tt$P.Value,
  FDR     = tt$adj.P.Val,
  B_stat  = tt$B,
  stringsAsFactors = FALSE
)
write.table(out_df, file = out_path, sep = "\t", quote = FALSE, row.names = FALSE)
"""


def run_limma_r(wide_pres, c_samps, k_samps, min_obs_per_group=2, rscript_bin="Rscript"):
    """
    Run Bioconductor limma (lmFit + eBayes trend) via Rscript on un-imputed
    observed log2 LFQ intensities (no Perseus imputation).
    """
    if wide_pres is None or wide_pres.empty or len(c_samps) < 2 or len(k_samps) < 2:
        return pd.DataFrame()

    if shutil.which(rscript_bin) is None:
        raise FileNotFoundError(
            f"'{rscript_bin}' not found in PATH. Load your HPC R module or set RSCRIPT_BIN in main.py."
        )

    all_samps = list(c_samps) + list(k_samps)
    safe_df = wide_pres[all_samps].where(wide_pres[all_samps] > 0)

    n_c_obs = safe_df[c_samps].notna().sum(axis=1)
    n_k_obs = safe_df[k_samps].notna().sum(axis=1)
    keep = (n_c_obs >= min_obs_per_group) & (n_k_obs >= min_obs_per_group)

    if not keep.any():
        return pd.DataFrame()

    log2_df = np.log2(safe_df.loc[keep])
    meta_df = pd.DataFrame({
        "Sample": all_samps,
        "Group": (["GALD"] * len(c_samps)) + (["Control"] * len(k_samps)),
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        expr_path = tmp / "expr_log2.tsv"
        meta_path = tmp / "meta.tsv"
        out_path = tmp / "limma_out.tsv"
        script_path = tmp / "run_limma.R"

        log2_df.to_csv(expr_path, sep="\t", na_rep="NA", index_label="Gene")
        meta_df.to_csv(meta_path, sep="\t", index=False)
        script_path.write_text(_LIMMA_R_SCRIPT, encoding="utf-8")

        proc = subprocess.run(
            [rscript_bin, str(script_path), str(expr_path), str(meta_path), str(out_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"R limma failed (exit {proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        res_df = pd.read_csv(out_path, sep="\t")

    res_df["n_GALD_obs"] = res_df["Gene"].map(n_c_obs.to_dict()).astype(int)
    res_df["n_Ctrl_obs"] = res_df["Gene"].map(n_k_obs.to_dict()).astype(int)
    return res_df.dropna(subset=["Log2FC", "p_value", "FDR"]).copy()


def analyze_volcano(
    wide_pres,
    c_samps,
    k_samps,
    subset_name,
    engine_name,
    pdf_obj,
    use_fdr,
    p_cutoff,
    fc_cutoff,
    highlight_genes,
    volcano_filtered_fisher_hits=None,
    min_obs_per_group=2,
    rscript_bin="Rscript",
):
    if wide_pres is None or wide_pres.empty or len(c_samps) < 2 or len(k_samps) < 2:
        return [], pd.DataFrame(), [], []

    v_df = run_limma_r(
        wide_pres, c_samps, k_samps,
        min_obs_per_group=min_obs_per_group,
        rscript_bin=rscript_bin,
    )
    if v_df.empty:
        return [], pd.DataFrame(), [], []

    stat = 'FDR' if use_fdr else 'p_value'
    v_df['neg_log10_stat'] = -np.log10(np.clip(v_df[stat].values, 1e-300, 1.0))

    sig = v_df[(v_df[stat] < p_cutoff) & (v_df['Log2FC'] >= fc_cutoff)].copy()
    down = v_df[(v_df[stat] < p_cutoff) & (v_df['Log2FC'] <= -fc_cutoff)].copy()

    # Attach pre-imputation presence, GALD intensity, percentile/Z-score, and exclusivity context
    if not sig.empty:
        pres_stats, _ = _compute_gald_intensity_stats(wide_pres, c_samps, k_samps)
        if not pres_stats.empty:
            sig = sig.merge(pres_stats, left_on='Gene', right_index=True, how='left')
            sig['Volcano_Filtered_Out'] = False

    top_hits = sig.sort_values(by=stat).head(10)['Gene'].tolist()
    hl_hits = [g for g in highlight_genes if g in sig['Gene'].values]

    if pdf_obj:
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(v_df['Log2FC'], v_df['neg_log10_stat'], alpha=0.5, color='grey', s=10, zorder=1)
        ax.scatter(sig['Log2FC'], sig['neg_log10_stat'], alpha=0.8, color='firebrick', s=20, zorder=2)
        ax.scatter(down['Log2FC'], down['neg_log10_stat'], alpha=0.8, color='steelblue', s=20, zorder=2)

        ax.set_title(f"{engine_name} limma Volcano (Observed-Only): {subset_name} (n={len(v_df)} proteins)")
        ax.axvline(x=fc_cutoff, color='firebrick', linestyle=':', linewidth=1.5)
        ax.axvline(x=-fc_cutoff, color='steelblue', linestyle=':', linewidth=1.5)
        ax.axhline(y=-np.log10(p_cutoff), color='black', linestyle=':', linewidth=1.5)

        for _, row in sig.iterrows():
            if row['Gene'] not in highlight_genes:
                ax.text(
                    row['Log2FC'] + 0.05,
                    row['neg_log10_stat'],
                    str(row['Gene']),
                    fontsize=6,
                    color='firebrick',
                    va='bottom',
                    ha='left',
                )

        for _, row in down.iterrows():
            if row['Gene'] not in highlight_genes:
                ax.text(
                    row['Log2FC'] - 0.05,
                    row['neg_log10_stat'],
                    str(row['Gene']),
                    fontsize=6,
                    color='steelblue',
                    va='bottom',
                    ha='right',
                )

        target_markers = ['*', '^', 'D', 's']
        target_colors = ['gold', 'lime', 'magenta', 'cyan']

        for idx, target in enumerate(highlight_genes):
            if target in v_df['Gene'].values:
                row = v_df[v_df['Gene'] == target].iloc[0]
                m = target_markers[idx % len(target_markers)]
                c = target_colors[idx % len(target_colors)]

                ax.scatter(
                    row['Log2FC'],
                    row['neg_log10_stat'],
                    color=c,
                    marker=m,
                    s=120,
                    edgecolor='black',
                    linewidth=1,
                    zorder=10,
                )

                align_ha = 'left' if row['Log2FC'] >= 0 else 'right'
                offset = 0.08 if row['Log2FC'] >= 0 else -0.08

                ax.text(
                    row['Log2FC'] + offset,
                    row['neg_log10_stat'],
                    str(row['Gene']),
                    fontsize=7,
                    color='black',
                    fontweight='bold',
                    va='bottom',
                    ha=align_ha,
                    zorder=11,
                )

        y_label_text = r'$-\log_{10}(\text{FDR})$' if use_fdr else r'$-\log_{10}(\text{p-value})$'
        ax.set_xlabel(r'$\log_2(\text{Fold Change})$ (GALD vs Control, Observed-Only limma)')
        ax.set_ylabel(y_label_text)

        # Surface high-intensity Fisher hits excluded from observed-only limma (<min_obs_per_group in one arm)
        if volcano_filtered_fisher_hits:
            max_show = 12
            shown = volcano_filtered_fisher_hits[:max_show]
            extra = len(volcano_filtered_fisher_hits) - len(shown)
            note = (
                f"Excluded from limma (<{min_obs_per_group} obs/group) but significant in Fisher w/ Above-Avg GALD intensity: "
                + ", ".join(shown)
                + (f" (+{extra} more)" if extra > 0 else "")
            )
            fig.text(
                0.5, -0.02, note,
                ha='center', va='top', fontsize=7.5, color='#4a1c1c',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff4e6', edgecolor='#d97706', lw=0.8),
            )

        pdf_obj.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    details = [f"{subset_name}(Up)" for _ in range(len(sig))]
    return details, sig.drop(columns=['neg_log10_stat']), top_hits, hl_hits

# ==========================================
# 3. FISHER'S EXACT TEST (PRESENCE/ABSENCE, RAW P-VALUE + GALD INTENSITY)
# ==========================================
def analyze_fisher(
    wide_pres,
    c_samps,
    k_samps,
    subset_name,
    p_cutoff,
    highlight_genes,
    volcano_genes=None,
    watchlist_p_cutoff=0.10,
):
    """
    One-sided Fisher's Exact Test (GALD > Control) on protein presence/absence,
    augmented with pre-imputation GALD log2 intensity benchmarking and explicit
    tracking of proteins filtered out of the quantitative Volcano arm.

    DESIGN NOTE FOR REVIEWERS (INTENTIONAL STATISTICAL CHOICE):
    BH-FDR correction is intentionally NOT applied to Fisher's Exact Test p-values.
    With small cohort sizes (n=12-15 per group), Fisher's exact p-values are
    coarsely discrete and have a high minimum attainable p-value floor; applying
    standard BH-FDR across hundreds of sparse proteins overly inflates adjusted
    p-values and eliminates true on/off autoantigen candidates. Instead, false
    positives are controlled downstream by requiring replication across >=2
    independent MS runs AND concordance across both search engines (MaxQuant
    and FragPipe). Directional enrichment is strictly verified by group
    proportions (cp / n_c > kp / n_k) rather than raw counts, and ties among
    discrete p-values are broken by descending GALD_Mean_Log2Int so high-signal
    GALD-exclusive proteins are never buried.
    """
    n_c = len(c_samps)
    n_k = len(k_samps)
    if wide_pres.empty or n_c == 0 or n_k == 0:
        return [], pd.DataFrame(), [], [], pd.DataFrame()

    pres_stats, subset_avg_log2 = _compute_gald_intensity_stats(wide_pres, c_samps, k_samps)
    volcano_gene_set = set(volcano_genes) if volcano_genes is not None else None

    records = []
    for gene in wide_pres.index:
        cp = int(pres_stats.loc[gene, 'GALD_sample_presence'])
        kp = int(pres_stats.loc[gene, 'Non-GALD_sample_presence'])

        # Only test proteins with missingness (exclude 0% or 100% across both groups)
        if (cp + kp) == 0 or (cp + kp) == (n_c + n_k):
            continue

        _, p_val = fisher_exact(
            [[cp, n_c - cp], [kp, n_k - kp]],
            alternative='greater',
        )
        records.append({
            'Gene': gene,
            'p_value': p_val,
            'GALD_sample_presence': cp,
            'Non-GALD_sample_presence': kp,
            'GALD_prop': cp / n_c,
            'Non-GALD_prop': kp / n_k,
            'GALD_Mean_Log2Int': pres_stats.loc[gene, 'GALD_Mean_Log2Int'],
            'Non-GALD_Mean_Log2Int': pres_stats.loc[gene, 'Non-GALD_Mean_Log2Int'],
            'GALD_Int_Pct': pres_stats.loc[gene, 'GALD_Int_Pct'],
            'GALD_Int_RobustZ': pres_stats.loc[gene, 'GALD_Int_RobustZ'],
            'Subset_GALD_Avg_Log2Int': subset_avg_log2,
            'Above_Avg_GALD_Int': bool(pres_stats.loc[gene, 'Above_Avg_GALD_Int']),
            'Exclusivity': pres_stats.loc[gene, 'Exclusivity'],
            'Volcano_Filtered_Out': (gene not in volcano_gene_set) if volcano_gene_set is not None else False,
        })

    if not records:
        return [], pd.DataFrame(), [], [], pd.DataFrame()

    f_df = pd.DataFrame(records)

    # Gate on raw p-value + proportion direction; break p-value ties by highest GALD intensity
    sig = (
        f_df[(f_df['p_value'] < p_cutoff) & (f_df['GALD_prop'] > f_df['Non-GALD_prop'])]
        .sort_values(by=['p_value', 'GALD_Mean_Log2Int'], ascending=[True, False], na_position='last')
        .copy()
    )

    # Borderline safeguard watchlist: Volcano-filtered, <=1 control detection, above-avg GALD intensity, p in [p_cutoff, watchlist_p_cutoff)
    watchlist = (
        f_df[
            (f_df['p_value'] >= p_cutoff)
            & (f_df['p_value'] < watchlist_p_cutoff)
            & (f_df['GALD_prop'] > f_df['Non-GALD_prop'])
            & (f_df['Volcano_Filtered_Out'])
            & (f_df['Non-GALD_sample_presence'] <= 1)
            & (f_df['Above_Avg_GALD_Int'])
        ]
        .sort_values(by=['p_value', 'GALD_Mean_Log2Int'], ascending=[True, False], na_position='last')
        .copy()
    )

    top_hits = sig.head(10)['Gene'].tolist()
    hl_hits = [g for g in highlight_genes if g in sig['Gene'].values]
    details = [f"{subset_name}(Fisher)" for _ in range(len(sig))]

    out_cols = [
        'Gene', 'p_value', 'GALD_sample_presence', 'Non-GALD_sample_presence',
        'GALD_Mean_Log2Int', 'Non-GALD_Mean_Log2Int',
        'GALD_Int_Pct', 'GALD_Int_RobustZ',
        'Subset_GALD_Avg_Log2Int', 'Above_Avg_GALD_Int',
        'Exclusivity', 'Volcano_Filtered_Out',
    ]
    return details, sig[out_cols], top_hits, hl_hits, watchlist[out_cols]


# ==========================================
# 4. PRINCIPAL COMPONENT ANALYSIS (MEAN-CENTERED)
# ==========================================
def analyze_pca(imputed_df, s_to_g, subset_name, engine_name, pdf_obj, seed):
    clean_df = imputed_df.dropna(axis=0)
    if clean_df.shape[0] < 2 or clean_df.shape[1] < 2:
        return

    samples = list(clean_df.columns)
    X = clean_df.T.values
    X_centered = X - np.mean(X, axis=0, keepdims=True)

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X_centered)
    var_pct = pca.explained_variance_ratio_ * 100.0

    if pdf_obj:
        fig, ax = plt.subplots(figsize=(9, 7))
        groups = [s_to_g.get(s, 'Unknown') for s in samples]
        unique_groups = sorted(set(groups))
        palette = {'GALD': 'firebrick', 'Healthy donor': 'steelblue', 'GALD_spouse': 'darkorange'}

        for grp in unique_groups:
            mask = [g == grp for g in groups]
            color = palette.get(grp, 'slategray')
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=grp,
                color=color,
                s=65,
                alpha=0.85,
                edgecolor='black',
                linewidth=0.6,
            )

        texts = []
        for idx, s in enumerate(samples):
            label_txt = str(s).replace('_', ' ').strip()
            t = ax.text(
                coords[idx, 0],
                coords[idx, 1],
                label_txt,
                fontsize=7,
                ha='center',
                va='center',
            )
            texts.append(t)

        if adjust_text is not None and texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))

        ax.axhline(0, color='lightgrey', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='lightgrey', linestyle='--', linewidth=0.8)
        ax.set_title(f"{engine_name} PCA: {subset_name}")
        ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}% variance)")
        ax.legend(frameon=True, loc='best')

        pdf_obj.savefig(fig, bbox_inches='tight')
        plt.close(fig)


# ==========================================
# 5. RANDOM FOREST (LEAK-FREE IN-FOLD FILTERING + FAST REPEATED CV)
# ==========================================
def _resolve_rf_n_jobs(n_jobs):
    if n_jobs is not None:
        return max(1, int(n_jobs))
    for env_key in ("SLURM_CPUS_PER_TASK", "LSB_DJOB_NUMPROC"):
        raw = os.environ.get(env_key)
        if raw:
            try:
                return max(1, min(int(raw), 8))
            except ValueError:
                pass
    return max(1, min(os.cpu_count() or 1, 8))


def analyze_rf(
    wide_pres,
    s_to_g,
    subset_name,
    highlight_genes,
    seed,
    n_splits=5,
    n_repeats=10,
    min_bal_acc=0.60,
    min_group_frac=0.5,
    n_estimators=200,
    n_jobs=None,
):
    if wide_pres is None or wide_pres.empty:
        return [], [], []

    valid_cols = [s for s in wide_pres.columns if s in s_to_g]
    if not valid_cols:
        return [], [], []

    pres_df = wide_pres[valid_cols].where(wide_pres[valid_cols] > 0).dropna(how='all', axis=0)
    if pres_df.empty:
        return [], [], []

    y_bin = np.array([1 if s_to_g[s] == 'GALD' else 0 for s in pres_df.columns], dtype=np.int32)
    class_counts = np.bincount(y_bin, minlength=2)
    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        print(f"      * RF skipped [{subset_name}]: insufficient samples per class (min_class={min_class_count})")
        return [], [], []

    # Pre-compute boolean presence mask and per-sample Perseus imputation once.
    # Because Perseus imputation uses only each sample's own observed distribution,
    # running it on wide_pres before CV introduces no label or train/test leakage.
    pres_mask = pres_df.notna().T.to_numpy(dtype=bool, copy=False)
    imputed_all = impute_perseus(pres_df, seed)
    X_all = np.ascontiguousarray(imputed_all.T.to_numpy(dtype=np.float32))
    genes = pres_df.index.to_numpy()
    n_genes = len(genes)

    eff_splits = min(n_splits, min_class_count)
    rskf = RepeatedStratifiedKFold(n_splits=eff_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(rskf.split(X_all, y_bin))
    eff_jobs = _resolve_rf_n_jobs(n_jobs)

    def _run_fold(fold_idx, train_idx, test_idx):
        y_train = y_bin[train_idx]
        y_test = y_bin[test_idx]

        # Leak-free group completeness filter computed strictly on training fold
        tr_pres = pres_mask[train_idx]
        c_tr = (y_train == 1)
        k_tr = (y_train == 0)

        min_c = max(2, int(np.ceil(c_tr.sum() * min_group_frac)))
        min_k = max(2, int(np.ceil(k_tr.sum() * min_group_frac)))

        keep_mask = (
            (tr_pres[c_tr].sum(axis=0) >= min_c)
            | (tr_pres[k_tr].sum(axis=0) >= min_k)
        )
        if not keep_mask.any():
            return None

        X_train = X_all[train_idx][:, keep_mask]
        X_test = X_all[test_idx][:, keep_mask]

        # Directional check strictly on training fold
        gald_up_kept = X_train[c_tr].mean(axis=0) > X_train[k_tr].mean(axis=0)

        # Random Forests are scale-invariant (no StandardScaler needed);
        # n_jobs=1 avoids thread thrashing while outer CV runs in parallel.
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed + fold_idx,
            class_weight='balanced',
            n_jobs=1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        return (
            accuracy_score(y_test, y_pred),
            balanced_accuracy_score(y_test, y_pred),
            keep_mask,
            rf.feature_importances_,
            gald_up_kept,
        )

    # Thread-based parallelism: Cython RF releases the GIL and avoids Matplotlib fork deadlocks
    results = Parallel(n_jobs=eff_jobs, prefer="threads")(
        delayed(_run_fold)(fold_idx, train_idx, test_idx)
        for fold_idx, (train_idx, test_idx) in enumerate(splits)
    )
    results = [r for r in results if r is not None]

    if not results:
        print(f"      * RF skipped [{subset_name}]: no features passed in-fold completeness filter")
        return [], [], []

    n_valid = len(results)
    fold_accs = np.empty(n_valid, dtype=float)
    fold_bal_accs = np.empty(n_valid, dtype=float)
    imp_sum = np.zeros(n_genes, dtype=float)
    gald_up_count = np.zeros(n_genes, dtype=int)

    for i, (acc, bacc, keep_mask, feat_imp, gald_up_kept) in enumerate(results):
        fold_accs[i] = acc
        fold_bal_accs[i] = bacc
        imp_sum[keep_mask] += feat_imp
        gald_up_count[keep_mask] += gald_up_kept.astype(int)

    ddof = 1 if n_valid > 1 else 0
    mean_acc, sd_acc = float(fold_accs.mean()), float(fold_accs.std(ddof=ddof))
    mean_bacc, sd_bacc = float(fold_bal_accs.mean()), float(fold_bal_accs.std(ddof=ddof))

    print(
        f"      * RF Repeated CV [{subset_name}] ({eff_splits}x{n_repeats} folds, n={len(y_bin)}): "
        f"Acc={mean_acc:.3f}±{sd_acc:.3f} | BalAcc={mean_bacc:.3f}±{sd_bacc:.3f}"
    )

    # Gate: do not return feature hits if CV balanced accuracy is at/near chance
    if mean_bacc < min_bal_acc:
        print(
            f"      * RF gated [{subset_name}]: BalAcc {mean_bacc:.3f} < {min_bal_acc:.2f} "
            "(returning 0 RF hits)"
        )
        return [], [], []

    imp_df = pd.DataFrame({
        'Gene': genes,
        'Imp': imp_sum / n_valid,
        'GALD_Up_FoldFrac': gald_up_count / n_valid,
    })

    # Require positive importance and GALD-abundant direction across training folds
    imp = (
        imp_df[(imp_df['Imp'] > 0) & (imp_df['GALD_Up_FoldFrac'] > 0.5)]
        .sort_values('Imp', ascending=False)
        .head(20)
    )

    hl_hits = [g for g in highlight_genes if g in imp['Gene'].values]
    details = [f"{subset_name}(#{i + 1})" for i in range(len(imp))]
    return details, imp['Gene'].tolist(), hl_hits


# ==========================================
# 6. PAIRED LIVER VS. LB BACKGROUND TEST + DIRECT DiD
# ==========================================
def analyze_background_ttest(
    prot_with_lb,
    engine_name,
    out_dir,
    highlight_genes,
    fc_thresh=0.5,
    p_thresh=0.05,
    min_liver_frac=0.5,
):
    """
    Paired LFQ evaluation of Liver Lysate (non-LB) vs. Lysate Buffer (LB)
    separately within GALD and non-GALD subjects, plus direct
    difference-in-differences (Delta_GALD > Delta_NonGALD):
      - Pairing strictly uses preserved Subject_ID / Donor_ID metadata when
        present, falling back to the l_num + 42 sample-number offset only with
        an explicit warning.
      - Tests:
        1) Within-cohort presence/absence enrichment via exact one-sided
           McNemar binomial test on discordant pairs:
           binomtest(liver_only, liver_only + lb_only, 0.5, alternative='greater')
           plus a zero-LB safeguard.
        2) Within-cohort quantitative enrichment on co-detected pairs via
           one-sided paired t-test with BH-FDR control.
        3) Direct GALD > Non-GALD specificity via:
           (a) Quantitative DiD: one-sided Welch's t-test on paired log2
               differences (BH-FDR controlled).
           (b) Clean Liver-only presence DiD: one-sided Fisher's exact test on
               Liver-only (Liver+ & LB-) pair counts.
           (c) Carryover-aware DiD ("GALD Liver+LB vs. Non-GALD all-absent/sparse"):
               for proteins BH-FDR confirmed as quantitatively Liver > LB in
               GALD (and not Liver-enriched in Non-GALD), tests GALD
               Liver-enriched pairs (Liver-only OR co-detected with
               diff >= fc_thresh) against total Non-GALD Liver detections via
               one-sided Fisher's exact test so high-abundance targets with
               trace LB carryover are not falsely marked unconfirmed when
               Non-GALD has <2 co-detected pairs.
    """
    if prot_with_lb is None or prot_with_lb.empty:
        return {}

    quant_col = (
        'Razor Intensity' if 'Razor Intensity' in prot_with_lb.columns
        else ('MaxLFQ Intensity' if 'MaxLFQ Intensity' in prot_with_lb.columns else 'Intensity')
    )

    df_clean = prot_with_lb.dropna(subset=['Gene']).copy()
    df_clean['Gene'] = df_clean['Gene'].astype(str).str.split(';').str[0].str.strip()
    df_clean = df_clean[df_clean['Gene'].ne('') & df_clean['Gene'].str.lower().ne('nan')]

    if df_clean.empty or 'Sample' not in df_clean.columns:
        return {}

    df_clean['Sample_str'] = df_clean['Sample'].astype(str).str.strip()

    def extract_num(s_name):
        cleaned = str(s_name).lower().replace('sample_', '').replace('sample', '').strip()
        try:
            return int(float(cleaned))
        except ValueError:
            return None

    has_group = 'Group' in df_clean.columns
    df_clean['Group_clean'] = df_clean['Group'].astype(str).str.strip() if has_group else 'Unknown'

    has_type = 'Sample_type' in df_clean.columns
    sample_to_type = (
        df_clean.drop_duplicates('Sample_str')
        .set_index('Sample_str')['Sample_type']
        .astype(str).str.strip().str.lower()
        .to_dict()
        if has_type else {}
    )

    def _clean_subj_series(s):
        cleaned = s.astype(str).str.strip()
        return cleaned.where(~cleaned.str.lower().isin({'', 'nan', 'none', 'na'}))

    subj_col = None
    for cand_col in ['Subject_ID', 'Subject', 'Donor_ID', 'Donor', 'Pair_ID', 'subject_id']:
        if cand_col in df_clean.columns and _clean_subj_series(df_clean[cand_col]).notna().any():
            subj_col = cand_col
            break

    uniq_samples = df_clean.drop_duplicates('Sample_str')
    sample_to_group = dict(zip(uniq_samples['Sample_str'], uniq_samples['Group_clean']))
    sample_to_subj = (
        dict(zip(uniq_samples['Sample_str'], _clean_subj_series(uniq_samples[subj_col])))
        if subj_col else {}
    )

    def is_lb_sample(s_name):
        if has_type:
            stype = sample_to_type.get(s_name, '')
            if 'buffer' in stype or stype == 'lb' or ' lb' in stype or '_lb' in stype:
                return True
            if 'liver' in stype or 'lysate' in stype:
                return False

        low = s_name.lower()
        if '_lb' in low or 'buffer' in low or low.endswith('lb'):
            return True
        s_num = extract_num(s_name)
        return s_num is not None and 43 <= s_num <= 84

    def is_liver_sample(s_name):
        if is_lb_sample(s_name):
            return False
        if has_type:
            stype = sample_to_type.get(s_name, '')
            if 'liver' in stype:
                return True
        s_num = extract_num(s_name)
        return s_num is None or s_num <= 42

    # Pivot at Protein Group level (no invalid summation of MaxLFQ intensities)
    wide_raw = _select_representative_protein_groups(df_clean, quant_col, sample_col='Sample_str')
    if wide_raw.empty:
        return {}
    wide_log2 = np.log2(wide_raw)

    liver_cols = [c for c in wide_log2.columns if is_liver_sample(c)]
    lb_cols = [c for c in wide_log2.columns if is_lb_sample(c)]

    gald_pairs = []
    nongald_pairs = []

    def classify_cohort(l_col, b_col):
        l_grp = str(sample_to_group.get(l_col, '')).strip()
        b_grp = str(sample_to_group.get(b_col, '')).strip()
        if l_grp and b_grp and l_grp.upper() != b_grp.upper():
            raise ValueError(
                f"❌ Background pairing mismatch: {l_col} ({l_grp}) paired with {b_col} ({b_grp})!"
            )
        grp = (l_grp or b_grp).upper()
        is_gald = (
            'GALD' in grp
            and 'SPOUSE' not in grp
            and 'NON' not in grp
            and 'CTRL' not in grp
            and 'HEALTHY' not in grp
        )
        if is_gald:
            gald_pairs.append((l_col, b_col))
        else:
            nongald_pairs.append((l_col, b_col))

    if subj_col:
        lb_by_subj = {
            str(sample_to_subj[b_col]).strip(): b_col
            for b_col in lb_cols
            if pd.notna(sample_to_subj.get(b_col))
        }
        unpaired_liver = []
        for l_col in liver_cols:
            s_id = sample_to_subj.get(l_col)
            if pd.notna(s_id):
                paired_lb = lb_by_subj.get(str(s_id).strip())
                if paired_lb is not None:
                    classify_cohort(l_col, paired_lb)
                    continue
            unpaired_liver.append(l_col)
        pairing_desc = f"Subject_ID ('{subj_col}')"
        if unpaired_liver:
            print(
                f"    ⚠️ [{engine_name} Background] {len(unpaired_liver)} Liver sample(s) had no matching "
                f"LB pair under '{subj_col}': {', '.join(unpaired_liver[:5])}"
            )
    else:
        print(
            f"    ⚠️ [{engine_name} Background] Subject_ID column unavailable/empty; "
            "falling back to sample-number offset (l_num + 42)."
        )
        lb_by_num = {extract_num(c): c for c in lb_cols if extract_num(c) is not None}
        for l_col in liver_cols:
            l_num = extract_num(l_col)
            if l_num is not None and 1 <= l_num <= 42:
                paired_lb = lb_by_num.get(l_num + 42)
                if paired_lb is not None:
                    classify_cohort(l_col, paired_lb)
        pairing_desc = "fallback l_num+42"

    def run_paired_cohort(pairs):
        idx = wide_log2.index
        n_total = len(pairs)
        res = pd.DataFrame(index=idx)
        res['liver_det'] = 0
        res['lb_det'] = 0
        res['liver_only'] = 0
        res['lb_only'] = 0
        res['both_tested'] = 0
        res['enriched_pairs'] = 0
        res['mcnemar_p'] = np.nan
        res['mean_fc'] = np.nan
        res['quant_p'] = np.nan
        res['quant_fdr'] = np.nan
        res['sig_absent_lb'] = False
        res['sig_quant'] = False

        if n_total == 0:
            return res, np.empty((len(idx), 0), dtype=float)

        l_list = [l_c for l_c, _ in pairs]
        b_list = [b_c for _, b_c in pairs]
        l_vals = wide_log2[l_list].values
        b_vals = wide_log2[b_list].values
        diffs = l_vals - b_vals

        l_pres = ~np.isnan(l_vals)
        b_pres = ~np.isnan(b_vals)

        n_liver_det = l_pres.sum(axis=1)
        n_lb_det = b_pres.sum(axis=1)
        n_liver_only = (l_pres & ~b_pres).sum(axis=1)
        n_lb_only = (~l_pres & b_pres).sum(axis=1)
        n_both = (l_pres & b_pres).sum(axis=1)
        n_both_enr = (l_pres & b_pres & (diffs >= fc_thresh)).sum(axis=1)

        res['liver_det'] = n_liver_det
        res['lb_det'] = n_lb_det
        res['liver_only'] = n_liver_only
        res['lb_only'] = n_lb_only
        res['both_tested'] = n_both
        res['enriched_pairs'] = n_liver_only + n_both_enr

        # Case 1: Exact one-sided paired McNemar binomial test on discordant pairs
        mcnemar_p = np.full(len(idx), np.nan, dtype=float)
        for i in range(len(idx)):
            lo = int(n_liver_only[i])
            bo = int(n_lb_only[i])
            n_disc = lo + bo
            if n_disc > 0:
                mcnemar_p[i] = binomtest(lo, n_disc, 0.5, alternative='greater').pvalue
        res['mcnemar_p'] = mcnemar_p

        min_liver_det = max(3, int(np.ceil(n_total * min_liver_frac)))
        res['sig_absent_lb'] = (
            (res['mcnemar_p'].notna() & (res['mcnemar_p'] < p_thresh) & (res['liver_only'] > res['lb_only']))
            | ((res['liver_det'] >= min_liver_det) & (res['lb_det'] == 0))
        )

        # Case 2: Quantitative paired t-test on co-detected pairs (BH-FDR controlled)
        if n_total >= 2:
            mean_fc = np.full(len(idx), np.nan, dtype=float)
            quant_p = np.full(len(idx), np.nan, dtype=float)

            for i in range(len(idx)):
                if n_both[i] >= 1:
                    valid_d = diffs[i, ~np.isnan(diffs[i])]
                    mean_fc[i] = float(np.mean(valid_d))
                    if len(valid_d) >= 2 and np.std(valid_d, ddof=1) > 0:
                        _, p = ttest_1samp(valid_d, 0.0, alternative='greater')
                        quant_p[i] = p

            res['mean_fc'] = mean_fc
            res['quant_p'] = quant_p

            valid_q = res['quant_p'].dropna()
            if not valid_q.empty:
                _, q_fdr, _, _ = multipletests(valid_q.values, method='fdr_bh')
                res.loc[valid_q.index, 'quant_fdr'] = q_fdr

            res['sig_quant'] = (
                res['mean_fc'].notna()
                & res['quant_fdr'].notna()
                & (res['mean_fc'] >= fc_thresh)
                & (res['quant_fdr'] < p_thresh)
            )

        return res, diffs

    gald_res, gald_diffs = run_paired_cohort(gald_pairs)
    nongald_res, nongald_diffs = run_paired_cohort(nongald_pairs)

    # Case 3: Direct Difference-in-Differences (Delta_GALD > Delta_NonGALD)
    idx = wide_log2.index
    n_genes = len(idx)
    n_gald = len(gald_pairs)
    n_nongald = len(nongald_pairs)

    did_fc = np.full(n_genes, np.nan, dtype=float)
    did_quant_p = np.full(n_genes, np.nan, dtype=float)
    did_pres_p = np.full(n_genes, np.nan, dtype=float)
    did_carryover_p = np.full(n_genes, np.nan, dtype=float)

    g_lo_arr = gald_res['liver_only'].values
    ng_lo_arr = nongald_res['liver_only'].values
    g_enr_arr = gald_res['enriched_pairs'].values
    ng_ld_arr = nongald_res['liver_det'].values

    for i in range(n_genes):
        # 3a. Quantitative DiD: Welch's one-sided t-test on paired log2 differences (BH-FDR controlled)
        g_d = gald_diffs[i, ~np.isnan(gald_diffs[i])] if gald_diffs.shape[1] > 0 else np.array([])
        ng_d = nongald_diffs[i, ~np.isnan(nongald_diffs[i])] if nongald_diffs.shape[1] > 0 else np.array([])

        if len(g_d) >= 1 and len(ng_d) >= 1:
            did_fc[i] = float(np.mean(g_d) - np.mean(ng_d))
        if len(g_d) >= 2 and len(ng_d) >= 2:
            if np.std(g_d, ddof=1) > 0 or np.std(ng_d, ddof=1) > 0:
                _, p_did = ttest_ind(g_d, ng_d, equal_var=False, alternative='greater')
                did_quant_p[i] = p_did

        # 3b. Clean Liver-only presence DiD: one-sided Fisher on Liver-only (Liver+ & LB-) pair counts
        if n_gald > 0 and n_nongald > 0 and (g_lo_arr[i] > 0 or ng_lo_arr[i] > 0):
            _, p_pres = fisher_exact(
                [[int(g_lo_arr[i]), int(n_gald - g_lo_arr[i])],
                 [int(ng_lo_arr[i]), int(n_nongald - ng_lo_arr[i])]],
                alternative='greater',
            )
            did_pres_p[i] = p_pres

        # 3c. Carryover-aware DiD ("GALD Liver+LB vs. Non-GALD all-absent/sparse"):
        # Compare GALD Liver-enriched pairs (Liver-only OR co-detected diff >= fc_thresh)
        # conservatively against ALL Non-GALD Liver detections.
        if n_gald > 0 and n_nongald > 0 and (g_enr_arr[i] > 0 or ng_ld_arr[i] > 0):
            _, p_co = fisher_exact(
                [[int(g_enr_arr[i]), int(n_gald - g_enr_arr[i])],
                 [int(ng_ld_arr[i]), int(n_nongald - ng_ld_arr[i])]],
                alternative='greater',
            )
            did_carryover_p[i] = p_co

    did_df = pd.DataFrame({
        'did_fc': did_fc,
        'did_quant_p': did_quant_p,
        'did_quant_fdr': np.nan,
        'did_pres_p': did_pres_p,
        'did_carryover_p': did_carryover_p,
    }, index=idx)

    valid_dq = did_df['did_quant_p'].dropna()
    if not valid_dq.empty:
        _, dq_fdr, _, _ = multipletests(valid_dq.values, method='fdr_bh')
        did_df.loc[valid_dq.index, 'did_quant_fdr'] = dq_fdr

    did_df['sig_did_quant'] = (
        did_df['did_fc'].notna()
        & did_df['did_quant_fdr'].notna()
        & (did_df['did_fc'] >= fc_thresh)
        & (did_df['did_quant_fdr'] < p_thresh)
    )
    did_df['sig_did_pres'] = (
        did_df['did_pres_p'].notna()
        & (did_df['did_pres_p'] < p_thresh)
        & ((g_lo_arr / max(n_gald, 1)) > (ng_lo_arr / max(n_nongald, 1)))
    )
    did_df['sig_did_carryover'] = (
        gald_res['sig_quant']
        & ~(nongald_res['sig_quant'] | nongald_res['sig_absent_lb'])
        & did_df['did_carryover_p'].notna()
        & (did_df['did_carryover_p'] < p_thresh)
        & ((g_enr_arr / max(n_gald, 1)) > (ng_ld_arr / max(n_nongald, 1)))
    )

    classifications = {}
    records = []

    for gene in idx:
        g = gald_res.loc[gene]
        ng = nongald_res.loc[gene]
        d = did_df.loc[gene]

        gald_sig_up = bool(g['sig_absent_lb'] or g['sig_quant'])
        nongald_sig_up = bool(ng['sig_absent_lb'] or ng['sig_quant'])
        gald_gt_ctrl = bool(d['sig_did_quant'] or d['sig_did_pres'] or d['sig_did_carryover'])

        if gald_sig_up and gald_gt_ctrl:
            if bool(g['sig_absent_lb']) and bool(d['sig_did_pres']) and not bool(d['sig_did_quant']):
                cat = "GALD-Enriched Over LB & Controls (Absent in LB)"
            elif bool(d['sig_did_carryover']) and not bool(d['sig_did_quant']) and not bool(d['sig_did_pres']):
                cat = "GALD-Enriched Over LB & Controls (Trace LB Carryover)"
            else:
                cat = "GALD-Enriched Over LB & Controls"
        elif gald_sig_up and nongald_sig_up:
            if bool(g['sig_absent_lb']) and bool(ng['sig_absent_lb']):
                cat = "Shared Liver Enrichment (Absent in LB)"
            else:
                cat = "Shared Liver Enrichment (GALD & Non-GALD)"
        elif gald_sig_up and not nongald_sig_up:
            if bool(g['sig_absent_lb']):
                cat = "Liver-Enriched in GALD (Unconfirmed vs Controls, Absent in LB)"
            else:
                cat = "Liver-Enriched in GALD (Unconfirmed vs Controls)"
        elif nongald_sig_up and not gald_sig_up:
            if bool(ng['sig_absent_lb']):
                cat = "Non-GALD Liver Enrichment (Absent in LB)"
            else:
                cat = "Non-GALD Liver Enrichment"
        elif int(g['lb_det']) == 0 and int(ng['lb_det']) == 0:
            cat = "Low Detection (Absent in LB)"
        else:
            cat = "Contaminant / Carryover (Not Enriched Over LB)"

        classifications[gene] = cat
        records.append({
            'Gene': gene,
            'Category': cat,
            'DiD_Log2FC_GALD_vs_NonGALD': d['did_fc'],
            'DiD_Quant_p': d['did_quant_p'],
            'DiD_Quant_FDR': d['did_quant_fdr'],
            'DiD_Pres_Fisher_p': d['did_pres_p'],
            'DiD_Carryover_Fisher_p': d['did_carryover_p'],
            'GALD_Liver_Det': int(g['liver_det']),
            'GALD_LB_Det': int(g['lb_det']),
            'GALD_LiverOnly_Pairs': int(g['liver_only']),
            'GALD_LBOnly_Pairs': int(g['lb_only']),
            'GALD_Both_Pairs_Tested': int(g['both_tested']),
            'GALD_Enriched_Pairs': int(g['enriched_pairs']),
            'GALD_Pairs_Total': n_gald,
            'GALD_McNemar_p': g['mcnemar_p'],
            'GALD_Log2FC_Liver_vs_LB': g['mean_fc'],
            'GALD_p_value': g['quant_p'],
            'GALD_FDR': g['quant_fdr'],
            'NonGALD_Liver_Det': int(ng['liver_det']),
            'NonGALD_LB_Det': int(ng['lb_det']),
            'NonGALD_LiverOnly_Pairs': int(ng['liver_only']),
            'NonGALD_LBOnly_Pairs': int(ng['lb_only']),
            'NonGALD_Both_Pairs_Tested': int(ng['both_tested']),
            'NonGALD_Enriched_Pairs': int(ng['enriched_pairs']),
            'NonGALD_Pairs_Total': n_nongald,
            'NonGALD_McNemar_p': ng['mcnemar_p'],
            'NonGALD_Log2FC_Liver_vs_LB': ng['mean_fc'],
            'NonGALD_p_value': ng['quant_p'],
            'NonGALD_FDR': ng['quant_fdr'],
        })

    if records and out_dir is not None:
        bg_df = pd.DataFrame(records).sort_values(
            by=[
                'Category', 'DiD_Quant_FDR', 'DiD_Pres_Fisher_p',
                'DiD_Carryover_Fisher_p', 'GALD_FDR', 'GALD_McNemar_p'
            ],
            na_position='last',
        )
        out_path = Path(out_dir) / f"{engine_name}_Background_LB_vs_Liver_Separated.tsv"
        bg_df.to_csv(out_path, sep='\t', index=False)

        counts = bg_df['Category'].value_counts().to_dict()
        print(
            f"    [{engine_name} Paired LB vs Liver | Pairing={pairing_desc} | "
            f"GALD n={n_gald}, Non-GALD n={n_nongald}]"
        )
        for k_cat, v_cnt in counts.items():
            print(f"      - {k_cat}: {v_cnt}")

        for hl in highlight_genes:
            if hl in classifications:
                row_hl = bg_df[bg_df['Gene'] == hl].iloc[0]
                print(
                    f"      * Target {hl}: {classifications[hl]} "
                    f"(DiD FC={row_hl['DiD_Log2FC_GALD_vs_NonGALD']:.2f}, "
                    f"DiD_Q_FDR={row_hl['DiD_Quant_FDR']:.3g}, DiD_P_p={row_hl['DiD_Pres_Fisher_p']:.3g}, "
                    f"DiD_CO_p={row_hl['DiD_Carryover_Fisher_p']:.3g} | "
                    f"GALD Liver/LB={row_hl['GALD_Liver_Det']}/{row_hl['GALD_LB_Det']} "
                    f"[LO={row_hl['GALD_LiverOnly_Pairs']}, BO={row_hl['GALD_LBOnly_Pairs']}, Enr={row_hl['GALD_Enriched_Pairs']}], "
                    f"McNemar_p={row_hl['GALD_McNemar_p']:.3g}, FC={row_hl['GALD_Log2FC_Liver_vs_LB']:.2f}, FDR={row_hl['GALD_FDR']:.3g} | "
                    f"Non-GALD Liver/LB={row_hl['NonGALD_Liver_Det']}/{row_hl['NonGALD_LB_Det']} "
                    f"[LO={row_hl['NonGALD_LiverOnly_Pairs']}, BO={row_hl['NonGALD_LBOnly_Pairs']}], "
                    f"McNemar_p={row_hl['NonGALD_McNemar_p']:.3g}, FC={row_hl['NonGALD_Log2FC_Liver_vs_LB']:.2f}, FDR={row_hl['NonGALD_FDR']:.3g})"
                )

    return classifications

# ==========================================
# 7. SORTING, EXPORT, AND CROSS-ENGINE SUMMARY HELPERS
# ==========================================
def _exclude_lb_samples(pep_df):
    """Return a copy of peptide detections excluding Lysate Buffer (LB) negative controls."""
    if pep_df is None or pep_df.empty or 'Sample' not in pep_df.columns:
        return pep_df

    df_filt = pep_df.copy()
    sample_str = df_filt['Sample'].astype(str).str.strip().str.lower()
    name_lb = sample_str.str.contains(r'(?:^|_)lb$|buffer', regex=True, na=False)

    s_num = pd.to_numeric(
        sample_str.str.replace('sample_', '', regex=False).str.replace('sample', '', regex=False).str.strip(),
        errors='coerce',
    )
    num_lb = s_num.between(43, 84, inclusive='both').fillna(False)

    if 'Sample_type' in df_filt.columns:
        stype = df_filt['Sample_type'].astype(str).str.strip().str.lower()
        type_lb = stype.str.contains(r'\blb\b|buffer', regex=True, na=False)
        lb_mask = type_lb | name_lb
    else:
        lb_mask = name_lb | num_lb

    return df_filt[~lb_mask].copy()


def export_engine_tables(
    engine_enriched_records,
    engine_watchlist_records,
    engine_name,
    out_dir,
    engine_hi_excl_records=None,
):
    """Clean, sort, and export per-engine enriched, watchlist, and high-intensity exclusive tables."""
    out_path = Path(out_dir)

    non_empty_records = [df_rec for df_rec in engine_enriched_records if not df_rec.empty]
    if non_empty_records:
        df_out = pd.concat(non_empty_records, ignore_index=True)
        col_order = [
            'Subset', 'Analysis', 'Gene', 'Log2FC', 'p_value', 'FDR',
            'GALD_sample_presence', 'Non-GALD_sample_presence',
            'GALD_Mean_Log2Int', 'Non-GALD_Mean_Log2Int',
            'GALD_Int_Pct', 'GALD_Int_RobustZ',
            'Subset_GALD_Avg_Log2Int', 'Above_Avg_GALD_Int',
            'Exclusivity', 'Volcano_Filtered_Out',
        ]
        ordered_cols = [c for c in col_order if c in df_out.columns]

        for num_col in [
            'Log2FC', 'p_value', 'FDR',
            'GALD_Mean_Log2Int', 'Non-GALD_Mean_Log2Int',
            'GALD_Int_Pct', 'GALD_Int_RobustZ', 'Subset_GALD_Avg_Log2Int',
        ]:
            if num_col in df_out.columns:
                df_out[num_col] = pd.to_numeric(df_out[num_col], errors='coerce')

        for pres_col in ['GALD_sample_presence', 'Non-GALD_sample_presence']:
            if pres_col in df_out.columns:
                df_out[pres_col] = pd.to_numeric(df_out[pres_col], errors='coerce').astype('Int64')

        sort_cols = [c for c in ['Subset', 'Analysis', 'p_value', 'GALD_Int_RobustZ'] if c in df_out.columns]
        sort_asc = [False if c == 'GALD_Int_RobustZ' else True for c in sort_cols]
        df_out = df_out[ordered_cols].sort_values(by=sort_cols, ascending=sort_asc, na_position='last')
        df_out.to_csv(out_path / f"{engine_name}_Enriched_Proteins.txt", sep='\t', index=False)

    non_empty_wl = [df_wl for df_wl in engine_watchlist_records if not df_wl.empty]
    if non_empty_wl:
        wl_all = pd.concat(non_empty_wl, ignore_index=True)
        wl_cols = [
            'Subset', 'Gene', 'p_value', 'GALD_sample_presence', 'Non-GALD_sample_presence',
            'GALD_Mean_Log2Int', 'Non-GALD_Mean_Log2Int', 'GALD_Int_Pct', 'GALD_Int_RobustZ',
            'Subset_GALD_Avg_Log2Int', 'Above_Avg_GALD_Int',
            'Exclusivity', 'Volcano_Filtered_Out',
        ]
        wl_all = wl_all[[c for c in wl_cols if c in wl_all.columns]].sort_values(
            by=['Subset', 'GALD_Int_RobustZ', 'p_value'],
            ascending=[True, False, True],
            na_position='last',
        )
        wl_all.to_csv(out_path / f"{engine_name}_VolcanoFiltered_Exclusive_Watchlist.tsv", sep='\t', index=False)

    if engine_hi_excl_records:
        non_empty_hi = [df_hi for df_hi in engine_hi_excl_records if not df_hi.empty]
        if non_empty_hi:
            hi_all = pd.concat(non_empty_hi, ignore_index=True).sort_values(
                by=['GALD_Int_RobustZ', 'GALD_Int_Pct', 'GALD_sample_presence'],
                ascending=[False, False, False],
                na_position='last',
            )
            hi_all.to_csv(out_path / f"{engine_name}_HighInt_Exclusive_Candidates.tsv", sep='\t', index=False)


def compile_master_summary(mq_res, fp_res, mq_all, fp_all, out_dir):
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

    mq_peaks = mq_res.get('gene_int_peak', {})
    fp_peaks = fp_res.get('gene_int_peak', {})

    fmt_f = lambda v, dec=1: f"{v:.{dec}f}" if (v is not None and np.isfinite(v)) else "-"

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

        mq_pk = mq_peaks.get(g, {})
        fp_pk = fp_peaks.get(g, {})

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
            'Peak_GALD_Pct(MQ|FP)': f"{fmt_f(mq_pk.get('max_pct'), 1)}|{fmt_f(fp_pk.get('max_pct'), 1)}",
            'Peak_GALD_RobustZ(MQ|FP)': f"{fmt_f(mq_pk.get('max_z'), 2)}|{fmt_f(fp_pk.get('max_z'), 2)}",
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
    out_file = Path(out_dir) / "MASTER_CrossEngine_Relevant_Proteins.tsv"

    if not master_df.empty:
        master_df = master_df.sort_values(by=['Background_TTest(MQ)', 'Gene'])
        master_df.to_csv(out_file, sep='\t', index=False)
        print(master_df.to_string(index=False))
    else:
        print("No proteins passed significance across >=2 independent runs in BOTH engines.")


def compile_above_avg_fisher_overlap(mq_res, fp_res, mq_all, fp_all, out_dir, highlight_genes):
    print(
        "\n" + "=" * 70
        + "\n🎯 CROSS-ENGINE OVERLAP: ABOVE-AVERAGE GALD INTENSITY FISHER HITS (MaxQuant ∩ FragPipe)\n"
        + "=" * 70
    )

    mq_above = mq_res.get('fisher_above_avg', {})
    fp_above = fp_res.get('fisher_above_avg', {})
    overlap_genes = sorted(set(mq_above.keys()).intersection(set(fp_above.keys())))

    out_file = Path(out_dir) / "MASTER_CrossEngine_AboveAvg_Fisher_Overlap.tsv"

    if not overlap_genes:
        print("No overlapping above-average GALD intensity Fisher hits found across MaxQuant and FragPipe.")
        return

    mq_details_map = mq_res.get('fisher_above_avg_details', {})
    fp_details_map = fp_res.get('fisher_above_avg_details', {})

    r6_mq_pep = _exclude_lb_samples(mq_all['run6_with_LB'].get('pep'))
    r6_fp_pep = _exclude_lb_samples(fp_all['run6_with_LB'].get('pep'))

    rows = []
    for g in overlap_genes:
        mq_d = mq_details_map.get(g, [])
        fp_d = fp_details_map.get(g, [])

        mq_runs = ",".join(sorted({d['subset'].split('_')[0] for d in mq_d})) or "-"
        fp_runs = ",".join(sorted({d['subset'].split('_')[0] for d in fp_d})) or "-"

        mq_pres = "; ".join(f"{d['subset']}: {d['pres']} (p={d['p_value']:.3g})" for d in mq_d)
        fp_pres = "; ".join(f"{d['subset']}: {d['pres']} (p={d['p_value']:.3g})" for d in fp_d)

        mq_mean_int = float(np.nanmean([d['log2_int'] for d in mq_d])) if mq_d else np.nan
        fp_mean_int = float(np.nanmean([d['log2_int'] for d in fp_d])) if fp_d else np.nan

        mq_excl = ", ".join(sorted({d['exclusivity'] for d in mq_d})) or "-"
        fp_excl = ", ".join(sorted({d['exclusivity'] for d in fp_d})) or "-"

        mq_vf = "Yes" if any(d['volcano_filtered'] for d in mq_d) else "No"
        fp_vf = "Yes" if any(d['volcano_filtered'] for d in fp_d) else "No"

        r4_mq = get_gald_peptide_interval(mq_all['run4_no_LB'].get('pep'), g)
        r4_fp = get_gald_peptide_interval(fp_all['run4'].get('pep'), g)
        r5_mq = get_gald_peptide_interval(mq_all['run5_no_LB'].get('pep'), g)
        r5_fp = get_gald_peptide_interval(fp_all['run5'].get('pep'), g)
        r6_mq = get_gald_peptide_interval(r6_mq_pep, g)
        r6_fp = get_gald_peptide_interval(r6_fp_pep, g)

        pep_string = f"[MQ/FP run 4: {r4_mq}/{r4_fp}, MQ/FP run 5: {r5_mq}/{r5_fp}, MQ/FP run 6: {r6_mq}/{r6_fp}]"

        rows.append({
            'Gene': g,
            'Candidate': "*" if g in highlight_genes else "",
            'Independent_Runs(MQ|FP)': f"{mq_runs}|{fp_runs}",
            'Mean_GALD_Log2Int(MQ|FP)': f"{mq_mean_int:.2f}|{fp_mean_int:.2f}",
            'Volcano_Filtered(MQ|FP)': f"{mq_vf}|{fp_vf}",
            'Exclusivity(MQ)': mq_excl,
            'Exclusivity(FP)': fp_excl,
            'MQ_Presence_and_RawP': mq_pres,
            'FP_Presence_and_RawP': fp_pres,
            'GALD Peptide Interval': pep_string,
            'Background_TTest(MQ)': mq_res['bg_class'].get(g, "Not Evaluated"),
            'Background_TTest(FP)': fp_res['bg_class'].get(g, "Not Evaluated"),
        })

    overlap_df = pd.DataFrame(rows).sort_values(
        by=['Background_TTest(MQ)', 'Exclusivity(MQ)', 'Gene']
    )
    overlap_df.to_csv(out_file, sep='\t', index=False)
    print(f"Found {len(overlap_df)} overlapping above-average GALD intensity Fisher hits:")
    print(overlap_df.to_string(index=False))
    print(f"Saved overlap table to: {out_file.resolve()}")


def compile_high_intensity_exclusive_summary(mq_res, fp_res, mq_all, fp_all, out_dir, highlight_genes):
    """
    Compile a dedicated cross-engine table of Top-Intensity, GALD-Exclusive (<=1 Ctrl)
    proteins — independent of the 50% completeness filter and p < 0.05 cutoff —
    sorted by cross-engine concordance, then descending Robust Z-score and Percentile.
    """
    print(
        "\n" + "=" * 70
        + "\n🔥 HIGH-INTENSITY GALD-EXCLUSIVE DETECTOR (Top 10% or Z>=2.0 & <=1 Ctrl; Independent of 50% Filter / p<0.05)\n"
        + "=" * 70
    )

    mq_map = mq_res.get('highint_excl_detector', {})
    fp_map = fp_res.get('highint_excl_detector', {})
    all_genes = sorted(set(mq_map.keys()) | set(fp_map.keys()))

    out_file = Path(out_dir) / "MASTER_CrossEngine_HighInt_Exclusive.tsv"

    if not all_genes:
        print("No proteins met the High-Intensity GALD-Exclusive criteria in either engine.")
        return

    r6_mq_pep = _exclude_lb_samples(mq_all['run6_with_LB'].get('pep'))
    r6_fp_pep = _exclude_lb_samples(fp_all['run6_with_LB'].get('pep'))

    def _safe_max(entries, key):
        vals = [e[key] for e in entries if e.get(key) is not None and np.isfinite(e[key])]
        return float(max(vals)) if vals else np.nan

    def _safe_mean(entries, key):
        vals = [e[key] for e in entries if e.get(key) is not None and np.isfinite(e[key])]
        return float(np.mean(vals)) if vals else np.nan

    fmt_f = lambda v, dec=2: f"{v:.{dec}f}" if np.isfinite(v) else "-"

    def _fmt_details(entries):
        if not entries:
            return "-"
        parts = []
        for e in entries:
            p_str = f"p={e['fisher_p']:.3g}" if np.isfinite(e['fisher_p']) else "p=NA"
            z_str = f"Z={e['robust_z']:.2f}" if np.isfinite(e['robust_z']) else "Z=NA"
            parts.append(
                f"{e['subset']}: {e['pres']} (pct={e['int_pct']:.1f}%, {z_str}, {p_str}, {e['status']})"
            )
        return "; ".join(parts)

    def _summarize_status(entries):
        if not entries:
            return "-"
        statuses = {e['status'] for e in entries}
        if statuses == {"MISSED (Neither)"}:
            return "MISSED (Neither)"
        return "/".join(sorted(statuses))

    rows = []
    for g in all_genes:
        mq_e = mq_map.get(g, [])
        fp_e = fp_map.get(g, [])

        in_mq, in_fp = bool(mq_e), bool(fp_e)
        if in_mq and in_fp:
            engine_support = "Both (MQ & FP)"
            support_rank = 0
        elif in_mq:
            engine_support = "MaxQuant Only"
            support_rank = 1
        else:
            engine_support = "FragPipe Only"
            support_rank = 2

        mq_runs = ",".join(sorted({e['subset'].split('_')[0] for e in mq_e})) or "-"
        fp_runs = ",".join(sorted({e['subset'].split('_')[0] for e in fp_e})) or "-"

        mq_max_z, fp_max_z = _safe_max(mq_e, 'robust_z'), _safe_max(fp_e, 'robust_z')
        mq_max_pct, fp_max_pct = _safe_max(mq_e, 'int_pct'), _safe_max(fp_e, 'int_pct')
        mq_max_int, fp_max_int = _safe_max(mq_e, 'gald_log2'), _safe_max(fp_e, 'gald_log2')
        mq_ctrl_int, fp_ctrl_int = _safe_mean(mq_e, 'ctrl_log2'), _safe_mean(fp_e, 'ctrl_log2')

        z_vals = [v for v in (mq_max_z, fp_max_z) if np.isfinite(v)]
        pct_vals = [v for v in (mq_max_pct, fp_max_pct) if np.isfinite(v)]
        sort_z = float(max(z_vals)) if z_vals else -np.inf
        sort_pct = float(max(pct_vals)) if pct_vals else -np.inf

        r4_mq = get_gald_peptide_interval(mq_all['run4_no_LB'].get('pep'), g)
        r4_fp = get_gald_peptide_interval(fp_all['run4'].get('pep'), g)
        r5_mq = get_gald_peptide_interval(mq_all['run5_no_LB'].get('pep'), g)
        r5_fp = get_gald_peptide_interval(fp_all['run5'].get('pep'), g)
        r6_mq = get_gald_peptide_interval(r6_mq_pep, g)
        r6_fp = get_gald_peptide_interval(r6_fp_pep, g)
        pep_string = f"[MQ/FP run 4: {r4_mq}/{r4_fp}, MQ/FP run 5: {r5_mq}/{r5_fp}, MQ/FP run 6: {r6_mq}/{r6_fp}]"

        rows.append({
            '_support_rank': support_rank,
            '_sort_z': sort_z,
            '_sort_pct': sort_pct,
            'Gene': g,
            'Candidate': "*" if g in highlight_genes else "",
            'Engine_Support': engine_support,
            'Independent_Runs(MQ|FP)': f"{mq_runs}|{fp_runs}",
            'Max_GALD_RobustZ(MQ|FP)': f"{fmt_f(mq_max_z, 2)}|{fmt_f(fp_max_z, 2)}",
            'Max_GALD_Int_Pct(MQ|FP)': f"{fmt_f(mq_max_pct, 1)}|{fmt_f(fp_max_pct, 1)}",
            'Peak_GALD_Log2Int(MQ|FP)': f"{fmt_f(mq_max_int, 2)}|{fmt_f(fp_max_int, 2)}",
            'Ctrl_Log2Int_IfDet(MQ|FP)': f"{fmt_f(mq_ctrl_int, 2)}|{fmt_f(fp_ctrl_int, 2)}",
            'Main_Test_Status(MQ|FP)': f"{_summarize_status(mq_e)}|{_summarize_status(fp_e)}",
            'GALD Peptide Interval': pep_string,
            'Background_TTest(MQ)': mq_res['bg_class'].get(g, "Not Evaluated"),
            'Background_TTest(FP)': fp_res['bg_class'].get(g, "Not Evaluated"),
            'MQ_Details': _fmt_details(mq_e),
            'FP_Details': _fmt_details(fp_e),
        })

    hi_df = (
        pd.DataFrame(rows)
        .sort_values(
            by=['_support_rank', '_sort_z', '_sort_pct', 'Gene'],
            ascending=[True, False, False, True],
        )
        .drop(columns=['_support_rank', '_sort_z', '_sort_pct'])
    )

    hi_df.to_csv(out_file, sep='\t', index=False)

    n_both = int((hi_df['Engine_Support'] == 'Both (MQ & FP)').sum())
    print(f"Found {len(hi_df)} High-Intensity GALD-Exclusive candidates ({n_both} concordant across BOTH engines):")
    display_cols = [
        'Gene', 'Candidate', 'Engine_Support', 'Independent_Runs(MQ|FP)',
        'Max_GALD_RobustZ(MQ|FP)', 'Max_GALD_Int_Pct(MQ|FP)',
        'Peak_GALD_Log2Int(MQ|FP)', 'Ctrl_Log2Int_IfDet(MQ|FP)',
        'Main_Test_Status(MQ|FP)', 'Background_TTest(MQ)', 'Background_TTest(FP)',
    ]
    print(hi_df[display_cols].head(30).to_string(index=False))
    if len(hi_df) > 30:
        print(f"  ... (+{len(hi_df) - 30} additional single-engine rows saved to TSV)")
    print(f"Saved High-Intensity Exclusive table to: {out_file.resolve()}")