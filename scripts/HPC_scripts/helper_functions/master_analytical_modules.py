import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


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
            records.append({'Gene': g, 'p_value': p, 'c_pres': cp, 'k_pres': kp})
    
    f_df = pd.DataFrame(records).dropna()
    if f_df.empty: return [], pd.DataFrame(), [], []
    
    sig = f_df[f_df['p_value'] < p_cutoff].copy()
    top_hits = sig.sort_values(by='p_value').head(10)['Gene'].tolist()
    hl_hits = [g for g in highlight_genes if g in sig['Gene'].values]
    details = [f"{subset_name}({r['c_pres']}/{n_c} vs {r['k_pres']}/{n_k})" for _, r in sig.iterrows()]
    
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

def analyze_rf(imputed_df, s_to_g, subset_name, highlight_genes, seed):
    X_sc = StandardScaler().fit_transform(imputed_df.T)
    y_bin = np.array([1 if s_to_g[s] == 'GALD' else 0 for s in imputed_df.columns])
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight='balanced', n_jobs=-1)
    rf.fit(X_sc, y_bin)
    
    imp_df = pd.DataFrame({'Gene': imputed_df.index, 'Imp': rf.feature_importances_})
    
    # STRICT DIRECTIONAL FILTER: Force RF to only extract targets where GALD Mean > Control Mean
    # BOUNDARY FIX: Only check samples that actually exist in the current imputed_df subset
    c_samps = [s for s in imputed_df.columns if s_to_g[s] == 'GALD']
    k_samps = [s for s in imputed_df.columns if s_to_g[s] != 'GALD']
    
    gald_mean = imputed_df[c_samps].mean(axis=1).values
    ctrl_mean = imputed_df[k_samps].mean(axis=1).values
    
    imp_df['GALD_Abundant'] = gald_mean > ctrl_mean
    imp = imp_df[imp_df['GALD_Abundant']].sort_values('Imp', ascending=False).head(20)
    
    hl_hits = [g for g in highlight_genes if g in imp['Gene'].values]
    details = [f"{subset_name}(#{i+1})" for i, r in enumerate(imp.iterrows())]
    return details, imp['Gene'].tolist(), hl_hits

def analyze_background_ttest(prot_df_with_lb, engine_name, out_dir, seed, highlight_genes):
    print(f"\n🔬 Running Welch's T-Test Contaminant Classification for {engine_name}...")
    quant_col = 'Razor Intensity' if 'Razor Intensity' in prot_df_with_lb.columns else ('MaxLFQ Intensity' if 'MaxLFQ Intensity' in prot_df_with_lb.columns else 'Intensity')
    df = prot_df_with_lb.dropna(subset=['Gene']).copy()
    df['Gene'] = df['Gene'].astype(str).str.split(';').str[0].str.strip()
    
    df['Sample_Num'] = df['Sample'].astype(str).str.lower().str.replace('sample_', '', regex=False).astype(int)
    
    # n <= 42 perfectly maps 1-42 as Liver, and pairs 43 (Buffer) to Subject 1
    df['Subject_ID'] = df['Sample_Num'].apply(lambda n: n if n <= 42 else n - 42)
    
    if 'Sample_type' in df.columns: df['Matrix'] = df['Sample_type'].apply(lambda x: 'Buffer' if 'buffer' in str(x).lower() else 'Liver')
    else: df['Matrix'] = df['Sample_Num'].apply(lambda n: 'Liver' if n <= 42 else 'Buffer')
        
    if 'Buffer' not in df['Matrix'].values:
        return {}

    sample_meta = df[['Sample', 'Subject_ID', 'Matrix', 'Group']].drop_duplicates().set_index('Sample')
    wide_df = df.pivot_table(index='Gene', columns='Sample', values=quant_col, aggfunc='sum').replace(0, np.nan).dropna(thresh=5, axis=0)
    
    rng = np.random.default_rng(seed)
    imputed_df = np.log2(wide_df).copy()
    for col in imputed_df.columns:
        valid = imputed_df[col].dropna()
        if len(valid) > 2:
            m, s = valid.mean(), valid.std()
            mask = imputed_df[col].isna()
            imputed_df.loc[mask, col] = rng.normal(loc=m - 1.8 * s, scale=s * 0.3, size=mask.sum())

    records = []
    for gene, row in imputed_df.iterrows():
        sub = pd.DataFrame({
            'Intensity': row.values, 
            'Matrix': [sample_meta.loc[c, 'Matrix'] for c in imputed_df.columns], 
            'Subject_ID': [sample_meta.loc[c, 'Subject_ID'] for c in imputed_df.columns], 
            'Cohort': [str(sample_meta.loc[c, 'Group']).strip() for c in imputed_df.columns]
        })
        
        # FIXED: Upgraded to pivot_table to prevent crashes on accidental technical replicates
        piv = sub.pivot_table(index='Subject_ID', columns='Matrix', values='Intensity', aggfunc='mean')
        if 'Liver' not in piv.columns or 'Buffer' not in piv.columns: continue
        
        piv['Delta'] = piv['Liver'] - piv['Buffer']
        piv['Cohort'] = sub.drop_duplicates('Subject_ID').set_index('Subject_ID')['Cohort']
        
        log2fc_matrix = sub[sub['Matrix'] == 'Liver']['Intensity'].mean() - sub[sub['Matrix'] == 'Buffer']['Intensity'].mean()
        
        gald_d = piv[piv['Cohort'] == 'GALD']['Delta'].dropna()
        ctrl_d = piv[piv['Cohort'] != 'GALD']['Delta'].dropna()
        
        p_delta = np.nan
        if len(gald_d) > 1 and len(ctrl_d) > 1:
            # FIXED (Directionality): Ensure the Delta is actually HIGHER in GALD before running the test
            if gald_d.mean() > ctrl_d.mean():
                p_delta = ttest_ind(gald_d, ctrl_d, equal_var=False)[1]
            else:
                p_delta = 1.0 # Force insignificance if controls have a higher tissue delta
                
        records.append({'Gene': gene, 'Log2FC_Liver_vs_Buffer': log2fc_matrix, 'P_Paired_Delta': p_delta})

    # FIXED: Drop NaNs before FDR to prevent statsmodels from crashing
    res = pd.DataFrame(records).dropna(subset=['P_Paired_Delta', 'Log2FC_Liver_vs_Buffer'])
    if res.empty: return {}
    
    res['FDR_Paired_Delta'] = multipletests(res['P_Paired_Delta'], method='fdr_bh')[1]
    
    def classify(r):
        if r['Log2FC_Liver_vs_Buffer'] <= 0.5: return "Contaminant (Maternal/Buffer Carryover)"
        elif r['P_Paired_Delta'] < 0.05: return "True Fetal Liver Target (GALD Enriched)"
        else: return "Non-Specific Fetal Liver Background"
            
    res['Classification'] = res.apply(classify, axis=1)
    
    promising = res[res['Classification'] == "True Fetal Liver Target (GALD Enriched)"]
    top_ttest_df = promising.sort_values(by='Log2FC_Liver_vs_Buffer', ascending=False).head(10)
    top_ttest = top_ttest_df['Gene'].tolist()
    
    print(f"    [{engine_name} Welch's T-Test Top 10 (Highest Fold Change True Targets)] -> {', '.join(top_ttest) if top_ttest else 'None found'}")
    
    candidate_hits = promising[promising['Gene'].isin(highlight_genes)]
    for _, row in candidate_hits.iterrows():
        print(f"      * Candidate Hit in Welch's T-Test: {row['Gene']} (Log2FC vs Buffer: {row['Log2FC_Liver_vs_Buffer']:.2f}, p-value: {row['P_Paired_Delta']:.2e})")
    
    res.to_csv(out_dir / f"Run6_Welchs_TTest_Classification_{engine_name}.txt", sep='\t', index=False)
    return res.set_index('Gene')['Classification'].to_dict()

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
