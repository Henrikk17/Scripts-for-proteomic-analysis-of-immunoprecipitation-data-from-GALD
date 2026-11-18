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
    X_sc = StandardScaler().fit_transform(imputed_df.T)
    y = np.array([s_to_g[s] for s in imputed_df.columns])
    pcs = PCA(n_components=2, random_state=seed).fit_transform(X_sc)
    
    if pdf_obj:
        fig, ax = plt.subplots()
        sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=y, ax=ax)
        ax.set_title(f"{engine_name} PCA: {subset_name}")
        pdf_obj.savefig(fig, bbox_inches='tight'); plt.close()

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