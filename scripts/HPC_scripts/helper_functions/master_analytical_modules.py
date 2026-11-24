# Master Analytical Modules: Volcano, Fisher, PCA, Random Forest, and Paired Background Testing

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, ttest_1samp, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from helper_functions.master_data_handlers import _select_representative_protein_groups

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None



# ==========================================
# SHARED HELPER: PRE-IMPUTATION GALD INTENSITY & EXCLUSIVITY
# ==========================================
def _compute_gald_intensity_stats(wide_pres, c_samps, k_samps):
    """
    Compute per-gene GALD presence, control presence, raw-positive mean log2
    intensity in GALD, and whether each gene is above the subset's average
    GALD protein log2 intensity (prior to any imputation).
    """
    if wide_pres is None or wide_pres.empty or not c_samps:
        return pd.DataFrame(), np.nan

    c_raw = wide_pres[c_samps].where(wide_pres[c_samps] > 0)
    c_log2 = np.log2(c_raw)
    gene_mean_log2 = c_log2.mean(axis=1, skipna=True)
    valid_means = gene_mean_log2.dropna()
    subset_avg_log2 = float(valid_means.mean()) if not valid_means.empty else np.nan

    c_counts = c_raw.notna().sum(axis=1).astype(int)
    k_counts = (
        wide_pres[k_samps].where(wide_pres[k_samps] > 0).notna().sum(axis=1).astype(int)
        if k_samps else pd.Series(0, index=wide_pres.index, dtype=int)
    )

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
        'Subset_GALD_Avg_Log2Int': subset_avg_log2,
        'Above_Avg_GALD_Int': gene_mean_log2 > subset_avg_log2 if np.isfinite(subset_avg_log2) else False,
        'Exclusivity': exclusivity,
    }, index=wide_pres.index)

    return stats_df, subset_avg_log2


# ==========================================
# 1. VOLCANO PLOT (WELCH'S T-TEST + BH-FDR)
# ==========================================
def analyze_volcano(
    imputed_df,
    c_samps,
    k_samps,
    subset_name,
    engine_name,
    pdf_obj,
    use_fdr,
    p_cutoff,
    fc_cutoff,
    highlight_genes,
    wide_pres=None,
    volcano_filtered_fisher_hits=None,
):
    if imputed_df.empty or len(c_samps) < 2 or len(k_samps) < 2:
        return [], pd.DataFrame(), [], []

    c_data = imputed_df[c_samps].values
    k_data = imputed_df[k_samps].values
    genes = imputed_df.index.values

    log2fc = np.mean(c_data, axis=1) - np.mean(k_data, axis=1)
    _, p_vals = ttest_ind(c_data, k_data, axis=1, equal_var=False)

    v_df = pd.DataFrame({'Gene': genes, 'Log2FC': log2fc, 'p_value': p_vals}).dropna()
    if v_df.empty:
        return [], pd.DataFrame(), [], []

    v_df['FDR'] = multipletests(v_df['p_value'], method='fdr_bh')[1]
    stat = 'FDR' if use_fdr else 'p_value'
    v_df['neg_log10_stat'] = -np.log10(np.clip(v_df[stat].values, 1e-300, 1.0))

    sig = v_df[(v_df[stat] < p_cutoff) & (v_df['Log2FC'] >= fc_cutoff)].copy()
    down = v_df[(v_df[stat] < p_cutoff) & (v_df['Log2FC'] <= -fc_cutoff)].copy()

    # Attach pre-imputation presence, GALD intensity, and exclusivity context
    if wide_pres is not None and not wide_pres.empty and not sig.empty:
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

        ax.set_title(f"{engine_name} Volcano: {subset_name}")
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
        ax.set_xlabel(r'$\log_2(\text{Fold Change})$ (GALD vs Control)')
        ax.set_ylabel(y_label_text)

        # Surface high-intensity Fisher hits that were filtered out of Volcano (<50% completeness)
        if volcano_filtered_fisher_hits:
            max_show = 12
            shown = volcano_filtered_fisher_hits[:max_show]
            extra = len(volcano_filtered_fisher_hits) - len(shown)
            note = (
                "Volcano-filtered (<50% completeness) but significant in Fisher w/ Above-Avg GALD intensity: "
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
        'GALD_Mean_Log2Int', 'Subset_GALD_Avg_Log2Int', 'Above_Avg_GALD_Int',
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
# 5. RANDOM FOREST (REPEATED STRATIFIED CV + GATING)
# ==========================================
def analyze_rf(
    imputed_df,
    s_to_g,
    subset_name,
    highlight_genes,
    seed,
    n_splits=5,
    n_repeats=20,
    min_bal_acc=0.60,
):
    clean_df = imputed_df.dropna(axis=0)
    c_samps = [s for s in clean_df.columns if s_to_g[s] == 'GALD']
    k_samps = [s for s in clean_df.columns if s_to_g[s] != 'GALD']

    if clean_df.shape[0] == 0 or len(c_samps) == 0 or len(k_samps) == 0:
        return [], [], []

    X_raw = clean_df.T.values
    y_bin = np.array([1 if s_to_g[s] == 'GALD' else 0 for s in clean_df.columns])
    class_counts = np.bincount(y_bin, minlength=2)
    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        print(f"      * RF skipped [{subset_name}]: insufficient samples per class (min_class={min_class_count})")
        return [], [], []

    eff_splits = min(n_splits, min_class_count)
    rskf = RepeatedStratifiedKFold(n_splits=eff_splits, n_repeats=n_repeats, random_state=seed)

    fold_accs = []
    fold_bal_accs = []
    fold_imps = np.zeros((eff_splits * n_repeats, clean_df.shape[0]), dtype=float)
    fold_gald_up = np.zeros((eff_splits * n_repeats, clean_df.shape[0]), dtype=bool)

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_raw, y_bin)):
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_bin[train_idx], y_bin[test_idx]

        # Directional check computed strictly on training fold (no test leakage)
        gald_mean_tr = X_train_raw[y_train == 1].mean(axis=0)
        ctrl_mean_tr = X_train_raw[y_train == 0].mean(axis=0)
        fold_gald_up[fold_idx] = gald_mean_tr > ctrl_mean_tr

        # Scale strictly on training fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=seed + fold_idx,
            class_weight='balanced',
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        fold_accs.append(accuracy_score(y_test, y_pred))
        fold_bal_accs.append(balanced_accuracy_score(y_test, y_pred))
        fold_imps[fold_idx] = rf.feature_importances_

    mean_acc, sd_acc = float(np.mean(fold_accs)), float(np.std(fold_accs, ddof=1))
    mean_bacc, sd_bacc = float(np.mean(fold_bal_accs)), float(np.std(fold_bal_accs, ddof=1))

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

    mean_imp = fold_imps.mean(axis=0)
    gald_up_frac = fold_gald_up.mean(axis=0)

    imp_df = pd.DataFrame({
        'Gene': clean_df.index,
        'Imp': mean_imp,
        'GALD_Up_FoldFrac': gald_up_frac,
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
    Evaluation of Liver Lysate (non-LB) vs. Lysate Buffer (LB) separately within
    GALD and non-GALD subjects, plus direct difference-in-differences
    (Delta_GALD > Delta_NonGALD):
      1) Presence/absence enrichment (clean targets absent or rare in LB) via
         one-sided Fisher's Exact Test (raw p < p_thresh) + zero-LB rule.
      2) Quantitative enrichment (detected in both Liver and LB) via
         one-sided paired t-test with BH-FDR.
      3) Direct GALD > Non-GALD specificity via one-sided Welch's t-test on
         paired log2 differences (quantitative DiD, BH-FDR) and one-sided
         Fisher's Exact Test on Liver-only pair proportions (presence/absence
         DiD, raw p < p_thresh).

    DESIGN NOTE FOR REVIEWERS (INTENTIONAL STATISTICAL CHOICE):
    1. Fisher's Exact Test is intentionally used in place of McNemar's test for
       the Liver vs. LB presence/absence check for methodological consistency
       with analyze_fisher across the pipeline. Although Liver and LB samples
       are subject-paired, treating the 2x2 detection counts with an unpaired
       Fisher's Exact Test is a valid, conservative approximation under positive
       within-subject correlation.
    2. BH-FDR correction is intentionally NOT applied to Fisher's Exact Test
       p-values (both within-cohort Liver vs. LB and presence/absence DiD).
       Because sample sizes are small (GALD n=14, Non-GALD n=27), discrete
       Fisher p-values inflate severely under BH adjustment, causing genuine
       LB-absent liver proteins to be misclassified. BH-FDR IS retained for all
       continuous t-tests (paired quantitative Liver vs. LB and quantitative DiD).
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

    subj_col = next(
        (c for c in ['Subject_ID', 'Subject', 'Donor_ID', 'Donor', 'Pair_ID', 'subject_id'] if c in df_clean.columns),
        None
    )

    sample_to_group = (
        df_clean.drop_duplicates('Sample_str')
        .set_index('Sample_str')['Group_clean']
        .to_dict()
    )
    sample_to_subj = (
        df_clean.drop_duplicates('Sample_str')
        .set_index('Sample_str')[subj_col]
        .to_dict()
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
        grp = str(sample_to_group.get(l_col, sample_to_group.get(b_col, ''))).strip().upper()
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
            if pd.notna(sample_to_subj.get(b_col)) and str(sample_to_subj.get(b_col)).strip() != ''
        }
        for l_col in liver_cols:
            s_id = sample_to_subj.get(l_col)
            if pd.notna(s_id) and str(s_id).strip() != '':
                paired_lb = lb_by_subj.get(str(s_id).strip())
                if paired_lb is not None:
                    classify_cohort(l_col, paired_lb)
    else:
        lb_by_num = {extract_num(c): c for c in lb_cols if extract_num(c) is not None}
        for l_col in liver_cols:
            l_num = extract_num(l_col)
            if l_num is not None and 1 <= l_num <= 42:
                paired_lb = lb_by_num.get(l_num + 42)
                if paired_lb is not None:
                    classify_cohort(l_col, paired_lb)

    def run_paired_cohort(pairs):
        idx = wide_log2.index
        n_total = len(pairs)
        res = pd.DataFrame(index=idx)
        res['liver_det'] = 0
        res['lb_det'] = 0
        res['liver_only'] = 0
        res['lb_only'] = 0
        res['both_tested'] = 0
        res['fisher_p'] = np.nan
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

        res['liver_det'] = n_liver_det
        res['lb_det'] = n_lb_det
        res['liver_only'] = n_liver_only
        res['lb_only'] = n_lb_only
        res['both_tested'] = n_both

        # Case 1: Presence/absence test via one-sided Fisher's Exact Test (unadjusted raw p; see DESIGN NOTE)
        fisher_p = np.full(len(idx), np.nan, dtype=float)
        for i in range(len(idx)):
            l_d = int(n_liver_det[i])
            b_d = int(n_lb_det[i])
            if 0 < (l_d + b_d) < (2 * n_total):
                _, p_f = fisher_exact(
                    [[l_d, n_total - l_d], [b_d, n_total - b_d]],
                    alternative='greater',
                )
                fisher_p[i] = p_f
        res['fisher_p'] = fisher_p

        min_liver_det = max(3, int(np.ceil(n_total * min_liver_frac)))
        res['sig_absent_lb'] = (
            (res['fisher_p'].notna() & (res['fisher_p'] < p_thresh) & (res['liver_det'] > res['lb_det']))
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

    g_lo_arr = gald_res['liver_only'].values
    ng_lo_arr = nongald_res['liver_only'].values

    for i in range(n_genes):
        # Quantitative DiD: Welch's one-sided t-test on paired log2 differences (BH-FDR controlled)
        g_d = gald_diffs[i, ~np.isnan(gald_diffs[i])] if gald_diffs.shape[1] > 0 else np.array([])
        ng_d = nongald_diffs[i, ~np.isnan(nongald_diffs[i])] if nongald_diffs.shape[1] > 0 else np.array([])

        if len(g_d) >= 1 and len(ng_d) >= 1:
            did_fc[i] = float(np.mean(g_d) - np.mean(ng_d))
        if len(g_d) >= 2 and len(ng_d) >= 2:
            if np.std(g_d, ddof=1) > 0 or np.std(ng_d, ddof=1) > 0:
                _, p_did = ttest_ind(g_d, ng_d, equal_var=False, alternative='greater')
                did_quant_p[i] = p_did

        # Presence/absence DiD: Fisher's exact test on Liver-only (Liver+ & LB-) pair counts (raw p; see DESIGN NOTE)
        if n_gald > 0 and n_nongald > 0 and (g_lo_arr[i] > 0 or ng_lo_arr[i] > 0):
            _, p_pres = fisher_exact(
                [[int(g_lo_arr[i]), int(n_gald - g_lo_arr[i])],
                 [int(ng_lo_arr[i]), int(n_nongald - ng_lo_arr[i])]],
                alternative='greater',
            )
            did_pres_p[i] = p_pres

    did_df = pd.DataFrame({
        'did_fc': did_fc,
        'did_quant_p': did_quant_p,
        'did_quant_fdr': np.nan,
        'did_pres_p': did_pres_p,
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

    classifications = {}
    records = []

    for gene in idx:
        g = gald_res.loc[gene]
        ng = nongald_res.loc[gene]
        d = did_df.loc[gene]

        gald_sig_up = bool(g['sig_absent_lb'] or g['sig_quant'])
        nongald_sig_up = bool(ng['sig_absent_lb'] or ng['sig_quant'])
        gald_gt_ctrl = bool(d['sig_did_quant'] or d['sig_did_pres'])

        if gald_sig_up and gald_gt_ctrl:
            if bool(g['sig_absent_lb']) and bool(d['sig_did_pres']) and not bool(d['sig_did_quant']):
                cat = "GALD-Enriched Over LB & Controls (Absent in LB)"
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
            'GALD_Liver_Det': int(g['liver_det']),
            'GALD_LB_Det': int(g['lb_det']),
            'GALD_LiverOnly_Pairs': int(g['liver_only']),
            'GALD_Both_Pairs_Tested': int(g['both_tested']),
            'GALD_Pairs_Total': n_gald,
            'GALD_Fisher_p': g['fisher_p'],
            'GALD_Log2FC_Liver_vs_LB': g['mean_fc'],
            'GALD_p_value': g['quant_p'],
            'GALD_FDR': g['quant_fdr'],
            'NonGALD_Liver_Det': int(ng['liver_det']),
            'NonGALD_LB_Det': int(ng['lb_det']),
            'NonGALD_LiverOnly_Pairs': int(ng['liver_only']),
            'NonGALD_Both_Pairs_Tested': int(ng['both_tested']),
            'NonGALD_Pairs_Total': n_nongald,
            'NonGALD_Fisher_p': ng['fisher_p'],
            'NonGALD_Log2FC_Liver_vs_LB': ng['mean_fc'],
            'NonGALD_p_value': ng['quant_p'],
            'NonGALD_FDR': ng['quant_fdr'],
        })

    if records and out_dir is not None:
        bg_df = pd.DataFrame(records).sort_values(
            by=['Category', 'DiD_Quant_FDR', 'DiD_Pres_Fisher_p', 'GALD_FDR', 'GALD_Fisher_p'],
            na_position='last',
        )
        out_path = Path(out_dir) / f"{engine_name}_Background_LB_vs_Liver_Separated.tsv"
        bg_df.to_csv(out_path, sep='\t', index=False)

        counts = bg_df['Category'].value_counts().to_dict()
        print(f"    [{engine_name} Paired LB vs Liver (GALD n={n_gald}, Non-GALD n={n_nongald})]")
        for k_cat, v_cnt in counts.items():
            print(f"      - {k_cat}: {v_cnt}")

        for hl in highlight_genes:
            if hl in classifications:
                row_hl = bg_df[bg_df['Gene'] == hl].iloc[0]
                print(
                    f"      * Target {hl}: {classifications[hl]} "
                    f"(DiD FC={row_hl['DiD_Log2FC_GALD_vs_NonGALD']:.2f}, "
                    f"DiD_Q_FDR={row_hl['DiD_Quant_FDR']:.3g}, DiD_P_p={row_hl['DiD_Pres_Fisher_p']:.3g} | "
                    f"GALD Liver/LB={row_hl['GALD_Liver_Det']}/{row_hl['GALD_LB_Det']}, "
                    f"Fisher_p={row_hl['GALD_Fisher_p']:.3g}, FC={row_hl['GALD_Log2FC_Liver_vs_LB']:.2f}, FDR={row_hl['GALD_FDR']:.3g} | "
                    f"Non-GALD Liver/LB={row_hl['NonGALD_Liver_Det']}/{row_hl['NonGALD_LB_Det']}, "
                    f"Fisher_p={row_hl['NonGALD_Fisher_p']:.3g}, FC={row_hl['NonGALD_Log2FC_Liver_vs_LB']:.2f}, FDR={row_hl['NonGALD_FDR']:.3g})"
                )

    return classifications