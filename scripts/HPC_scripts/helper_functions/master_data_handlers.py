import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from helper_functions.load_maxQuant_proteomic_data import load_maxQuant_proteomic_data, find_maxquant_txt_dir
from helper_functions.load_fragPipe_proteomic_data import load_fragPipe_proteomic_data


def get_fragpipe_imms_datasets(cache_dir, base_dir="~/results/LFQ_MBR_IM-MS"):
    cache_file = cache_dir / "fp_imms_master_data.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    base_path = Path(base_dir).expanduser()
    run_specs = {
        "run4": {"path": base_path / "run4_results", "annot": "../experiment_annotation/experiment_annotation_run4_R876.tsv"},
        "run5": {"path": base_path / "run5_results", "annot": "../experiment_annotation/experiment_annotation_run5_R1101.tsv"},
        "run6": {"path": base_path / "run6_results_with_LB", "annot": "../experiment_annotation/experiment_annotation_run6_R1344.tsv"}
    }

    fp_data = {}
    for key, spec in run_specs.items():
        prot_df, pep_df = load_fragPipe_proteomic_data(ROOT=spec["path"], annotation_file=spec["annot"])
        prot_df = prot_df[~prot_df['Sample'].isin(['sample_14', 'sample_56'])].copy()
        if pep_df is not None:
            pep_df = pep_df[~pep_df['Sample'].isin(['sample_14', 'sample_56'])].copy()

        if key == "run6":
            fp_data["run6_with_LB"] = {"prot": prot_df.copy(), "pep": pep_df.copy() if pep_df is not None else None}
            if 'Sample_type' in prot_df.columns:
                fp_data["run6_no_LB"] = {
                    "prot": prot_df[prot_df['Sample_type'].astype(str).str.strip() == 'Liver Lysate'].copy(),
                    "pep": pep_df[pep_df['Sample_type'].astype(str).str.strip() == 'Liver Lysate'].copy() if pep_df is not None else None
                }
            else:
                fp_data["run6_no_LB"] = {"prot": prot_df.copy(), "pep": pep_df.copy() if pep_df is not None else None}
        else:
            fp_data[key] = {"prot": prot_df, "pep": pep_df}

    with open(cache_file, "wb") as f:
        pickle.dump(fp_data, f)
    return fp_data


def get_maxquant_datasets(cache_dir, base_dir="~/DTU-GALD"):
    cache_file = cache_dir / "mq_master_data.pkl"
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
        prot_df, pep_df = load_maxQuant_proteomic_data(ROOT=txt_dir, annotation_file=spec["annot"], load_peptides=True)

        prot_df = prot_df[~prot_df['Sample'].isin(['sample_14', 'sample_56'])].copy()
        if pep_df is not None:
            pep_df = pep_df[~pep_df['Sample'].isin(['sample_14', 'sample_56'])].copy()

        mq_data[f"{key}_with_LB"] = {"prot": prot_df.copy(), "pep": pep_df.copy() if pep_df is not None else None}
        if 'Sample_type' in prot_df.columns:
            mq_data[f"{key}_no_LB"] = {
                "prot": prot_df[prot_df['Sample_type'].astype(str).str.strip() == 'Liver Lysate'].copy(),
                "pep": pep_df[pep_df['Sample_type'].astype(str).str.strip() == 'Liver Lysate'].copy() if pep_df is not None else None
            }
        else:
            mq_data[f"{key}_no_LB"] = {"prot": prot_df.copy(), "pep": pep_df.copy() if pep_df is not None else None}

    with open(cache_file, "wb") as f:
        pickle.dump(mq_data, f)
    return mq_data


def get_gald_peptide_interval(pep_df, gene):
    if pep_df is None or pep_df.empty:
        return "-"
    subset = pep_df[(pep_df['Gene'] == gene) & (pep_df['Group'] == 'GALD')]
    if subset.empty:
        return "0"
    counts = subset.groupby('Sample')['Peptide Sequence'].nunique()
    if counts.empty:
        return "0"
    min_c, max_c = counts.min(), counts.max()
    return str(min_c) if min_c == max_c else f"{min_c}-{max_c}"


def build_four_subsets(prot_r4, prot_r5, prot_r6):
    return {
        "R4_GALD_vs_Healthy": (prot_r4, "GALD", "Healthy donor"),
        "R5_GALD_vs_Spouse":  (prot_r5, "GALD", "GALD_spouse"),
        "R6_GALD_vs_Healthy": (prot_r6, "GALD", "Healthy donor"),
        "R6_GALD_vs_Spouse":  (prot_r6, "GALD", "GALD_spouse")
    }


def _select_representative_protein_groups(df_clean, quant_col, sample_col='Sample'):
    """
    Pivot at the Protein Group level (avoiding invalid summation of MaxLFQ intensities)
    and select one representative protein group per leading Gene (most valid detections,
    tie-broken by mean intensity).
    """
    if df_clean.empty:
        return pd.DataFrame()

    pg_col = (
        'Protein Group' if 'Protein Group' in df_clean.columns
        else ('Protein' if 'Protein' in df_clean.columns else 'Gene')
    )

    if pg_col == 'Gene':
        wide_df = df_clean.pivot_table(
            index='Gene',
            columns=sample_col,
            values=quant_col,
            aggfunc='max',
        )
        return wide_df.where(wide_df > 0)

    pg_wide = df_clean.pivot_table(
        index=[pg_col, 'Gene'],
        columns=sample_col,
        values=quant_col,
        aggfunc='max',
    )
    pg_wide = pg_wide.where(pg_wide > 0)

    valid_counts = pg_wide.notna().sum(axis=1)
    mean_int = pg_wide.mean(axis=1, skipna=True).fillna(0.0)

    rank_df = pd.DataFrame({
        'gene_key': pg_wide.index.get_level_values('Gene'),
        'valid_n': valid_counts.values,
        'mean_int': mean_int.values,
    }, index=pg_wide.index)

    best_idx = (
        rank_df.sort_values(by=['gene_key', 'valid_n', 'mean_int'], ascending=[True, False, False])
        .drop_duplicates(subset='gene_key', keep='first')
        .index
    )

    wide_df = pg_wide.loc[best_idx].copy()
    wide_df.index = wide_df.index.get_level_values('Gene')
    return wide_df


def prep_wide_df(df, group_case, group_ctrl, min_group_frac=0.5):
    quant_col = (
        'Razor Intensity' if 'Razor Intensity' in df.columns
        else ('MaxLFQ Intensity' if 'MaxLFQ Intensity' in df.columns else 'Intensity')
    )
    df_clean = df.dropna(subset=['Gene']).copy()
    df_clean['Gene'] = df_clean['Gene'].astype(str).str.split(';').str[0].str.strip()
    df_clean = df_clean[df_clean['Gene'].ne('') & df_clean['Gene'].str.lower().ne('nan')]

    s_to_g = (
        df_clean.drop_duplicates('Sample')
        .set_index('Sample')['Group']
        .astype(str).str.strip()
        .to_dict()
    )
    c_samps = [s for s, g in s_to_g.items() if g == group_case]
    k_samps = [s for s, g in s_to_g.items() if g == group_ctrl]

    wide_all = _select_representative_protein_groups(df_clean, quant_col, sample_col='Sample')
    target_samples = [s for s in (c_samps + k_samps) if s in wide_all.columns]
    c_valid = [s for s in c_samps if s in target_samples]
    k_valid = [s for s in k_samps if s in target_samples]

    if wide_all.empty or not target_samples:
        empty_df = pd.DataFrame()
        return empty_df, empty_df, c_valid, k_valid, s_to_g

    # Arm 1 (Presence/Absence): keep all genes detected in at least 1 target sample
    wide_pres = wide_all[target_samples].dropna(how='all', axis=0)

    # Arm 2 (Quantitative + Perseus imputation): >= min_group_frac valid in GALD OR Control
    min_c = max(2, int(np.ceil(len(c_valid) * min_group_frac))) if c_valid else 0
    min_k = max(2, int(np.ceil(len(k_valid) * min_group_frac))) if k_valid else 0

    c_ok = (
        wide_pres[c_valid].notna().sum(axis=1) >= min_c
        if c_valid else pd.Series(False, index=wide_pres.index)
    )
    k_ok = (
        wide_pres[k_valid].notna().sum(axis=1) >= min_k
        if k_valid else pd.Series(False, index=wide_pres.index)
    )
    wide_quant = wide_pres[c_ok | k_ok].copy()

    return wide_quant, wide_pres, c_valid, k_valid, s_to_g


def impute_perseus(wide_df, seed):
    rng = np.random.default_rng(seed)
    safe_df = wide_df.where(wide_df > 0)
    log2_df = np.log2(safe_df)
    imputed = log2_df.copy()

    # Global fallback in case a column has <= 2 valid proteins
    all_vals = log2_df.values[~np.isnan(log2_df.values)]
    global_m = float(np.mean(all_vals)) if len(all_vals) > 0 else 0.0
    global_s = float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 1.0

    for col in imputed.columns:
        val = imputed[col].dropna()
        mask = imputed[col].isna()
        if not mask.any():
            continue

        if len(val) > 2:
            m, s = val.mean(), val.std()
        else:
            m, s = global_m, global_s

        if np.isnan(s) or s == 0:
            s = global_s if global_s > 0 else 1.0

        imputed.loc[mask, col] = rng.normal(loc=m - 1.8 * s, scale=s * 0.3, size=mask.sum())

    return imputed