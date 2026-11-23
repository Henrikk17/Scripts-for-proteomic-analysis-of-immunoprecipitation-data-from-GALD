from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_1samp, ttest_ind
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def analyze_volcano(imputed_df, c_samps, k_samps, subset_name, engine_name, pdf_obj, use_fdr, p_cutoff, fc_cutoff, highlight_genes):
    c_data = imputed_df[c_samps].values
    k_data = imputed_df[k_samps].values
    genes = imputed_df.index.values
    
    log2fc = np.mean(c_data, axis=1) - np.mean(k_data, axis=1)
    _, p_vals = ttest_ind(c_data, k_data, axis=1, equal_var=False)
    
    v_df = pd.DataFrame({'Gene': genes, 'Log2FC': log2fc, 'p_value': p_vals}).dropna()
    v_df['FDR'] = multipletests(v_df['p_value'], method='fdr_bh')[1]
    stat = 'FDR' if use_fdr else 'p_value'
    
    sig = v_df[(v_df[stat] < p_cutoff) & (v_df['Log2FC'] >= fc_cutoff)].copy()
    down = v_df[(v_df[stat] < p_cutoff) & (v_df['Log2FC'] <= -fc_cutoff)].copy()
    
    top_hits = sig.sort_values(by=stat).head(10)['Gene'].tolist()
    hl_hits = [g for g in highlight_genes if g in sig['Gene'].values]
    
    if pdf_obj:
        fig, ax = plt.subplots(figsize=(10,8))
        
        ax.scatter(v_df['Log2FC'], -np.log10(v_df[stat]), alpha=0.5, color='grey', s=10, zorder=1)
        ax.scatter(sig['Log2FC'], -np.log10(sig[stat]), alpha=0.8, color='firebrick', s=20, zorder=2)
        ax.scatter(down['Log2FC'], -np.log10(down[stat]), alpha=0.8, color='steelblue', s=20, zorder=2)
        
        ax.set_title(f"{engine_name} Volcano: {subset_name}")
        ax.axvline(x=fc_cutoff, color='firebrick', linestyle=':', linewidth=1.5)
        ax.axvline(x=-fc_cutoff, color='steelblue', linestyle=':', linewidth=1.5)
        ax.axhline(y=-np.log10(p_cutoff), color='black', linestyle=':', linewidth=1.5)
        
        for _, row in sig.iterrows():
            if row['Gene'] not in highlight_genes:
                ax.text(row['Log2FC'] + 0.05, -np.log10(row[stat]), str(row['Gene']), 
                        fontsize=6, color='firebrick', va='bottom', ha='left')
                
        for _, row in down.iterrows():
            if row['Gene'] not in highlight_genes:
                ax.text(row['Log2FC'] - 0.05, -np.log10(row[stat]), str(row['Gene']), 
                        fontsize=6, color='steelblue', va='bottom', ha='right')

        target_markers = ['*', '^', 'D', 's']
        target_colors = ['gold', 'lime', 'magenta', 'cyan']
        
        for idx, target in enumerate(highlight_genes):
            if target in v_df['Gene'].values:
                row = v_df[v_df['Gene'] == target].iloc[0]
                m = target_markers[idx % len(target_markers)]
                c = target_colors[idx % len(target_colors)]
                
                ax.scatter(row['Log2FC'], -np.log10(row[stat]), color=c, marker=m, 
                           s=120, edgecolor='black', linewidth=1, zorder=10)
                
                align_ha = 'left' if row['Log2FC'] >= 0 else 'right'
                offset = 0.08 if row['Log2FC'] >= 0 else -0.08
                
                ax.text(row['Log2FC'] + offset, -np.log10(row[stat]), str(row['Gene']), 
                        fontsize=7, color='black', fontweight='bold', va='bottom', ha=align_ha, zorder=11)

        y_label_text = r'$-\log_{10}(\text{FDR})$' if use_fdr else r'$-\log_{10}(\text{p-value})$'
        ax.set_xlabel(r'$\log_2(\text{Fold Change})$ (GALD vs Control)')
        ax.set_ylabel(y_label_text)
        
        pdf_obj.savefig(fig, bbox_inches='tight'); plt.close()
    
    details = [f"{subset_name}(Up)" for _, r in sig.iterrows()]
    return details, sig, top_hits, hl_hits

def analyze_fisher(wide_df, c_samps, k_samps, subset_name, p_cutoff, highlight_genes):
    c_pres = wide_df[c_samps].notna().sum(axis=1).values
    k_pres = wide_df[k_samps].notna().sum(axis=1).values
    genes = wide_df.index.values

    n_c, n_k = len(c_samps), len(k_samps)
    records = []

    for g, cp, kp in zip(genes, c_pres, k_pres):
        if cp > kp:
            _, p = fisher_exact([[cp, n_c - cp], [kp, n_k - kp]])
            records.append({
                'Gene': g,
                'p_value': p,
                'GALD_sample_presence': int(cp),
                'Non-GALD_sample_presence': int(kp)
            })

    f_df = pd.DataFrame(records).dropna(subset=['p_value'])
    if f_df.empty:
        return [], pd.DataFrame(), [], []

    # Raw p-value cutoff only (no FDR adjustment for Fisher's exact test)
    sig = f_df[f_df['p_value'] < p_cutoff].copy()
    top_hits = sig.sort_values(by='p_value').head(10)['Gene'].tolist()
    hl_hits = [g for g in highlight_genes if g in sig['Gene'].values]
    details = [
        f"{subset_name}({r['GALD_sample_presence']}/{n_c} vs {r['Non-GALD_sample_presence']}/{n_k})"
        for _, r in sig.iterrows()
    ]

    return details, sig, top_hits, hl_hits

def analyze_pca(imputed_df, s_to_g, subset_name, engine_name, pdf_obj, seed):
    clean_df = imputed_df.dropna(axis=0)
    if clean_df.shape[0] < 2 or clean_df.shape[1] < 2:
        return

    # Mean-center only (preserve log2 fold-change scale without inflating imputed noise)
    X_sc = StandardScaler(with_std=False).fit_transform(clean_df.T)
    y = np.array([s_to_g[s] for s in clean_df.columns])

    pca = PCA(n_components=2, random_state=seed)
    pcs = pca.fit_transform(X_sc)
    var_pct = pca.explained_variance_ratio_ * 100

    if pdf_obj:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=y, ax=ax, s=45, zorder=2)

        ax.set_title(f"{engine_name} PCA: {subset_name}")
        ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}%)")

        # Draw once so axis limits and transforms are finalized
        fig.canvas.draw()

        labels = [
            f"sample {str(col).lower().replace('sample_', '').replace('sample', '').strip()}"
            for col in clean_df.columns
        ]

        # Start labels centered on each point in display (pixel) coordinates
        disp_pts = ax.transData.transform(pcs)
        pos = disp_pts.copy()

        # Estimate label half-widths and half-heights in pixels
        fontsize = 6.5
        char_w = fontsize * 0.58
        box_h = fontsize * 1.35 + 4.0
        pad_x, pad_y = 4.0, 2.0

        half_w = np.array([(len(lbl) * char_w) / 2.0 + pad_x for lbl in labels])
        half_h = np.full(len(labels), box_h / 2.0 + pad_y)

        # Iterative bounding-box repulsion in pixel space to reduce overlap
        n_pts = len(labels)
        for _ in range(80):
            moved = False
            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    min_dx = half_w[i] + half_w[j]
                    min_dy = half_h[i] + half_h[j]

                    if abs(dx) < min_dx and abs(dy) < min_dy:
                        overlap_x = min_dx - abs(dx)
                        overlap_y = min_dy - abs(dy)

                        # Push along the axis with smaller overlap
                        if overlap_x < overlap_y:
                            sign = 1.0 if dx >= 0 else -1.0
                            shift = sign * (overlap_x * 0.5 + 0.5)
                            pos[i, 0] += shift
                            pos[j, 0] -= shift
                        else:
                            sign = 1.0 if dy >= 0 else -1.0
                            shift = sign * (overlap_y * 0.5 + 0.5)
                            pos[i, 1] += shift
                            pos[j, 1] -= shift
                        moved = True
            if not moved:
                break

        inv_trans = ax.transData.inverted()
        for idx, lbl in enumerate(labels):
            lx, ly = inv_trans.transform(pos[idx])
            ax.text(
                lx, ly, lbl,
                fontsize=fontsize,
                alpha=0.9,
                ha="center",
                va="center",
                zorder=3
            )

            # Draw a subtle leader line if a label was pushed away from its point
            dist = np.hypot(pos[idx, 0] - disp_pts[idx, 0], pos[idx, 1] - disp_pts[idx, 1])
            if dist > 6.0:
                ax.plot(
                    [pcs[idx, 0], lx], [pcs[idx, 1], ly],
                    color="grey", linewidth=0.5, alpha=0.6, zorder=2
                )

        # Expand axis limits slightly if pushed labels sit near the edge
        all_label_data = inv_trans.transform(pos)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(
            min(x_min, float(all_label_data[:, 0].min()) - 0.03 * abs(x_max - x_min)),
            max(x_max, float(all_label_data[:, 0].max()) + 0.03 * abs(x_max - x_min)),
        )
        ax.set_ylim(
            min(y_min, float(all_label_data[:, 1].min()) - 0.03 * abs(y_max - y_min)),
            max(y_max, float(all_label_data[:, 1].max()) + 0.03 * abs(y_max - y_min)),
        )

        pdf_obj.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def analyze_rf(imputed_df, s_to_g, subset_name, highlight_genes, seed, test_size=0.25):
    clean_df = imputed_df.dropna(axis=0)
    c_samps = [s for s in clean_df.columns if s_to_g[s] == 'GALD']
    k_samps = [s for s in clean_df.columns if s_to_g[s] != 'GALD']

    if clean_df.shape[0] == 0 or len(c_samps) == 0 or len(k_samps) == 0:
        return [], [], []

    y_bin = np.array([1 if s_to_g[s] == 'GALD' else 0 for s in clean_df.columns])
    n_samples = len(y_bin)
    class_counts = np.bincount(y_bin, minlength=2)
    min_class_count = int(class_counts.min())

    # Determine whether a stratified train/test split is possible
    n_test = max(2, int(np.ceil(n_samples * test_size)))
    can_split = (
        n_samples >= 6
        and min_class_count >= 2
        and (n_samples - n_test) >= 2
        and n_test < n_samples
    )

    if can_split:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            clean_df.T.values,
            y_bin,
            test_size=n_test,
            random_state=seed,
            stratify=y_bin
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"      * RF Test Accuracy [{subset_name}] (n_train={len(y_train)}, n_test={len(y_test)}): {acc:.3f} ({acc * 100:.1f}%)")
    else:
        X_all = StandardScaler().fit_transform(clean_df.T.values)
        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_all, y_bin)
        y_pred = rf.predict(X_all)
        acc = accuracy_score(y_bin, y_pred)
        print(f"      * RF Resubstitution Accuracy [{subset_name}] (too few samples for split, n={n_samples}): {acc:.3f} ({acc * 100:.1f}%)")

    imp_df = pd.DataFrame({'Gene': clean_df.index, 'Imp': rf.feature_importances_})

    # STRICT DIRECTIONAL FILTER: Force RF to only extract targets where GALD Mean > Control Mean
    gald_mean = clean_df[c_samps].mean(axis=1).values
    ctrl_mean = clean_df[k_samps].mean(axis=1).values

    imp_df['GALD_Abundant'] = gald_mean > ctrl_mean
    imp = imp_df[imp_df['GALD_Abundant']].sort_values('Imp', ascending=False).head(20)

    hl_hits = [g for g in highlight_genes if g in imp['Gene'].values]
    details = [f"{subset_name}(#{i + 1})" for i in range(len(imp))]
    return details, imp['Gene'].tolist(), hl_hits

def analyze_background_ttest(prot_with_lb, engine_name, out_dir, seed, highlight_genes, fc_thresh=0.5, p_thresh=0.05):
    """
    Paired evaluation of Liver Lysate (non-LB) vs. Lysate Buffer (LB)
    separately within GALD and non-GALD subjects, with BH-FDR adjustment.
    Uses explicit subject/donor IDs when available, falling back to the
    run-6 sample-number pairing (1..42 -> +42 -> 43..84).
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

    df_clean['Sample_num'] = df_clean['Sample_str'].apply(extract_num)

    has_group = 'Group' in df_clean.columns
    df_clean['Group_clean'] = df_clean['Group'].astype(str).str.strip() if has_group else 'Unknown'

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
        low = s_name.lower()
        if '_lb' in low or 'buffer' in low or low.endswith('lb'):
            return True
        s_num = extract_num(s_name)
        return s_num is not None and 43 <= s_num <= 84

    def is_liver_sample(s_name):
        if is_lb_sample(s_name):
            return False
        s_num = extract_num(s_name)
        return s_num is None or s_num <= 42

    wide_raw = df_clean.pivot_table(index='Gene', columns='Sample_str', values=quant_col, aggfunc='sum')
    wide_raw = wide_raw.where(wide_raw > 0)
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
        empty_fdr = pd.Series(np.nan, index=wide_log2.index, dtype=float)
        if len(pairs) < 2:
            return empty_fdr, empty_fdr, empty_fdr

        diffs = [wide_log2[l_c] - wide_log2[b_c] for l_c, b_c in pairs]
        diff_df = pd.concat(diffs, axis=1)
        mean_fc = diff_df.mean(axis=1, skipna=True)

        p_vals = pd.Series(np.nan, index=wide_log2.index, dtype=float)
        for gene, row in diff_df.iterrows():
            vals = row.dropna().values
            if len(vals) >= 2 and np.std(vals, ddof=1) > 0:
                _, p = ttest_1samp(vals, 0.0, alternative='greater')
                p_vals.loc[gene] = p

        fdr_vals = pd.Series(np.nan, index=wide_log2.index, dtype=float)
        valid_p = p_vals.dropna()
        if not valid_p.empty:
            _, fdr_adj, _, _ = multipletests(valid_p.values, method='fdr_bh')
            fdr_vals.loc[valid_p.index] = fdr_adj

        return mean_fc, p_vals, fdr_vals

    gald_fc, gald_p, gald_fdr = run_paired_cohort(gald_pairs)
    nongald_fc, nongald_p, nongald_fdr = run_paired_cohort(nongald_pairs)

    classifications = {}
    records = []

    for gene in wide_log2.index:
        g_fc = gald_fc.loc[gene]
        g_p = gald_p.loc[gene]
        g_fdr = gald_fdr.loc[gene]
        ng_fc = nongald_fc.loc[gene]
        ng_p = nongald_p.loc[gene]
        ng_fdr = nongald_fdr.loc[gene]

        gald_sig_up = pd.notna(g_fc) and pd.notna(g_fdr) and (g_fc >= fc_thresh) and (g_fdr < p_thresh)
        nongald_sig_up = pd.notna(ng_fc) and pd.notna(ng_fdr) and (ng_fc >= fc_thresh) and (ng_fdr < p_thresh)

        if gald_sig_up and nongald_sig_up:
            cat = "Shared Liver Enrichment (GALD & Non-GALD)"
        elif gald_sig_up and not nongald_sig_up:
            cat = "GALD-Specific Liver Enrichment"
        elif nongald_sig_up and not gald_sig_up:
            cat = "Non-GALD-Only Liver Enrichment"
        else:
            cat = "Contaminant / Carryover (Not Enriched Over LB)"

        classifications[gene] = cat
        records.append({
            'Gene': gene,
            'Category': cat,
            'GALD_Log2FC_Liver_vs_LB': g_fc,
            'GALD_p_value': g_p,
            'GALD_FDR': g_fdr,
            'NonGALD_Log2FC_Liver_vs_LB': ng_fc,
            'NonGALD_p_value': ng_p,
            'NonGALD_FDR': ng_fdr,
            'GALD_Pairs_n': len(gald_pairs),
            'NonGALD_Pairs_n': len(nongald_pairs),
        })

    if records and out_dir is not None:
        bg_df = pd.DataFrame(records).sort_values(by=['Category', 'GALD_FDR', 'NonGALD_FDR', 'GALD_p_value'])
        out_path = Path(out_dir) / f"{engine_name}_Background_LB_vs_Liver_Separated.tsv"
        bg_df.to_csv(out_path, sep='\t', index=False)

        counts = bg_df['Category'].value_counts().to_dict()
        print(f"    [{engine_name} Paired LB vs Liver (GALD n={len(gald_pairs)}, Non-GALD n={len(nongald_pairs)})]")
        for k_cat, v_cnt in counts.items():
            print(f"      - {k_cat}: {v_cnt}")

        for hl in highlight_genes:
            if hl in classifications:
                row_hl = bg_df[bg_df['Gene'] == hl].iloc[0]
                print(
                    f"      * Target {hl}: {classifications[hl]} "
                    f"(GALD FC={row_hl['GALD_Log2FC_Liver_vs_LB']:.2f}, FDR={row_hl['GALD_FDR']:.3g}, p={row_hl['GALD_p_value']:.3g} | "
                    f"Non-GALD FC={row_hl['NonGALD_Log2FC_Liver_vs_LB']:.2f}, FDR={row_hl['NonGALD_FDR']:.3g}, p={row_hl['NonGALD_p_value']:.3g})"
                )

    return classifications

def analyze_permutation_volcano(
    imputed_df, c_samps, k_samps, subset_name, engine_name,
    pdf_obj, p_cutoff, fc_cutoff, highlight_genes, seed, n_perm=250
):
    c_data = imputed_df[c_samps].values
    k_data = imputed_df[k_samps].values
    genes = imputed_df.index.values

    # Observed GALD vs Control (positive Log2FC = more abundant in GALD)
    log2fc_obs = np.mean(c_data, axis=1) - np.mean(k_data, axis=1)
    _, p_obs = ttest_ind(c_data, k_data, axis=1, equal_var=False, nan_policy='omit')
    neg_log10_p_obs = -np.log10(np.clip(p_obs, 1e-300, 1.0))

    # Build empirical null by shuffling sample labels across combined matrix
    X = np.hstack([c_data, k_data])
    n_c = len(c_samps)
    n_total = X.shape[1]
    rng = np.random.default_rng(seed)

    perm_up_counts = np.zeros(len(genes), dtype=float)
    perm_down_counts = np.zeros(len(genes), dtype=float)

    valid_obs = ~np.isnan(log2fc_obs) & ~np.isnan(neg_log10_p_obs)
    obs_indices = np.where(valid_obs)[0]

    # Pre-collect permuted (fc, -log10(p)) pairs for GALD-up and GALD-down directions
    perm_up_scores = []
    perm_down_scores = []

    for _ in range(n_perm):
        perm_idx = rng.permutation(n_total)
        c_perm = X[:, perm_idx[:n_c]]
        k_perm = X[:, perm_idx[n_c:]]

        fc_p = np.mean(c_perm, axis=1) - np.mean(k_perm, axis=1)
        _, p_p = ttest_ind(c_perm, k_perm, axis=1, equal_var=False, nan_policy='omit')
        nlp_p = -np.log10(np.clip(p_p, 1e-300, 1.0))

        up_mask = (~np.isnan(fc_p)) & (~np.isnan(nlp_p)) & (fc_p >= fc_cutoff)
        down_mask = (~np.isnan(fc_p)) & (~np.isnan(nlp_p)) & (fc_p <= -fc_cutoff)

        perm_up_scores.append(np.sort(nlp_p[up_mask]))
        perm_down_scores.append(np.sort(nlp_p[down_mask]))

    perm_fdr = np.ones(len(genes), dtype=float)

    # Empirical FDR for GALD-enriched candidates (Log2FC >= fc_cutoff)
    up_candidates = [
        i for i in obs_indices
        if log2fc_obs[i] >= fc_cutoff
    ]
    if up_candidates:
        up_candidates.sort(key=lambda idx: neg_log10_p_obs[idx], reverse=True)
        running_max_fdr = 0.0
        for rank_idx, i in enumerate(up_candidates, start=1):
            thresh = neg_log10_p_obs[i]
            # Count how many permuted hits exceed thresh in each permutation
            per_perm_hits = [
                len(arr) - np.searchsorted(arr, thresh, side='left')
                for arr in perm_up_scores
            ]
            mean_false = np.mean(per_perm_hits) if per_perm_hits else 0.0
            fdr_est = np.clip(mean_false / rank_idx, 0.0, 1.0)
            running_max_fdr = max(running_max_fdr, float(fdr_est))
            perm_fdr[i] = running_max_fdr

    # Empirical FDR for GALD-depleted candidates (Log2FC <= -fc_cutoff) for visual context
    down_candidates = [
        i for i in obs_indices
        if log2fc_obs[i] <= -fc_cutoff
    ]
    if down_candidates:
        down_candidates.sort(key=lambda idx: neg_log10_p_obs[idx], reverse=True)
        running_max_down_fdr = 0.0
        for rank_idx, i in enumerate(down_candidates, start=1):
            thresh = neg_log10_p_obs[i]
            per_perm_hits = [
                len(arr) - np.searchsorted(arr, thresh, side='left')
                for arr in perm_down_scores
            ]
            mean_false = np.mean(per_perm_hits) if per_perm_hits else 0.0
            fdr_est = np.clip(mean_false / rank_idx, 0.0, 1.0)
            running_max_down_fdr = max(running_max_down_fdr, float(fdr_est))
            perm_fdr[i] = running_max_down_fdr

    v_df = pd.DataFrame({
        'Gene': genes,
        'Log2FC': log2fc_obs,
        'p_value': p_obs,
        'Perm_FDR': perm_fdr,
        'neg_log10_Perm_FDR': -np.log10(np.clip(perm_fdr, 1e-10, 1.0))
    }).dropna(subset=['Gene', 'Log2FC', 'Perm_FDR'])

    # Strictly GALD-enriched significant hits
    sig = v_df[(v_df['Perm_FDR'] < p_cutoff) & (v_df['Log2FC'] >= fc_cutoff)].copy()
    down = v_df[(v_df['Perm_FDR'] < p_cutoff) & (v_df['Log2FC'] <= -fc_cutoff)].copy()

    top_hits = sig.sort_values(by=['Perm_FDR', 'p_value']).head(10)['Gene'].tolist()
    hl_hits = [g for g in highlight_genes if g in sig['Gene'].values]

    if pdf_obj:
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(v_df['Log2FC'], v_df['neg_log10_Perm_FDR'], alpha=0.5, color='grey', s=10, zorder=1)
        ax.scatter(sig['Log2FC'], sig['neg_log10_Perm_FDR'], alpha=0.8, color='firebrick', s=20, zorder=2)
        ax.scatter(down['Log2FC'], down['neg_log10_Perm_FDR'], alpha=0.8, color='steelblue', s=20, zorder=2)

        ax.set_title(f"{engine_name} Permutation Volcano ({n_perm} perms): {subset_name}")
        ax.axvline(x=fc_cutoff, color='firebrick', linestyle=':', linewidth=1.5)
        ax.axvline(x=-fc_cutoff, color='steelblue', linestyle=':', linewidth=1.5)
        ax.axhline(y=-np.log10(p_cutoff), color='black', linestyle=':', linewidth=1.5)

        for _, row in sig.iterrows():
            if row['Gene'] not in highlight_genes:
                ax.text(
                    row['Log2FC'] + 0.05, row['neg_log10_Perm_FDR'], str(row['Gene']),
                    fontsize=6, color='firebrick', va='bottom', ha='left'
                )

        for _, row in down.iterrows():
            if row['Gene'] not in highlight_genes:
                ax.text(
                    row['Log2FC'] - 0.05, row['neg_log10_Perm_FDR'], str(row['Gene']),
                    fontsize=6, color='steelblue', va='bottom', ha='right'
                )

        target_markers = ['*', '^', 'D', 's']
        target_colors = ['gold', 'lime', 'magenta', 'cyan']

        for idx, target in enumerate(highlight_genes):
            if target in v_df['Gene'].values:
                row = v_df[v_df['Gene'] == target].iloc[0]
                m = target_markers[idx % len(target_markers)]
                c = target_colors[idx % len(target_colors)]

                ax.scatter(
                    row['Log2FC'], row['neg_log10_Perm_FDR'], color=c, marker=m,
                    s=120, edgecolor='black', linewidth=1, zorder=10
                )

                align_ha = 'left' if row['Log2FC'] >= 0 else 'right'
                offset = 0.08 if row['Log2FC'] >= 0 else -0.08

                ax.text(
                    row['Log2FC'] + offset, row['neg_log10_Perm_FDR'], str(row['Gene']),
                    fontsize=7, color='black', fontweight='bold', va='bottom', ha=align_ha, zorder=11
                )

        ax.set_xlabel(r'$\log_2(\text{Fold Change})$ (GALD vs Control)')
        ax.set_ylabel(r'$-\log_{10}(\text{Permutation FDR})$')

        pdf_obj.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    return sig['Gene'].tolist(), top_hits, hl_hits
