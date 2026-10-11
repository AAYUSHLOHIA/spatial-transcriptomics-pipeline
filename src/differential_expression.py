"""
Differential Expression (DGE) Module for Spatial Biomaterial-Host Interface.
Identifies significantly upregulated and downregulated genes across biomaterial morphologies
and spatial distance zones using parametric and non-parametric hypothesis testing with FDR control.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats


def run_differential_expression(
    adata,
    group_col: str,
    group_1: str,
    group_2: str,
    min_pct: float = 0.1,
    min_log2fc: float = 0.25
) -> pd.DataFrame:
    """
    Computes differential expression between two groups in adata.obs.
    Calculates log2 fold change, Welch's t-test statistic, p-value, and Benjamini-Hochberg FDR.
    """
    mask1 = adata.obs[group_col] == group_1
    mask2 = adata.obs[group_col] == group_2
    
    n1 = int(np.sum(mask1))
    n2 = int(np.sum(mask2))
    
    if n1 < 3 or n2 < 3:
        raise ValueError(f"Insufficient spots for DGE: group_1={n1}, group_2={n2}")
        
    X1 = adata.X[mask1]
    X2 = adata.X[mask2]
    
    genes = list(adata.var_names)
    results = []
    
    for i, g in enumerate(genes):
        v1 = X1[:, i]
        v2 = X2[:, i]
        
        pct1 = float(np.mean(v1 > 0))
        pct2 = float(np.mean(v2 > 0))
        
        # Filter if not expressed in sufficient proportion
        if pct1 < min_pct and pct2 < min_pct:
            continue
            
        m1 = float(np.mean(v1))
        m2 = float(np.mean(v2))
        
        # Log2 fold change (adding pseudocount to avoid division by zero)
        log2fc = float(np.log2((m1 + 1e-4) / (m2 + 1e-4)))
        
        # Welch's two-sample t-test (unequal variances)
        t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False, nan_policy="omit")
        
        if np.isnan(p_val):
            p_val = 1.0
            
        results.append({
            "gene": g,
            "mean_group1": m1,
            "mean_group2": m2,
            "log2_fold_change": log2fc,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "pct_group1": pct1,
            "pct_group2": pct2,
            "category": adata.var.loc[g, "marker_category"] if "marker_category" in adata.var else "Other"
        })
        
    df_res = pd.DataFrame(results)
    if len(df_res) == 0:
        return df_res
        
    # Multiple testing correction (Benjamini-Hochberg FDR)
    p_values = df_res["p_value"].values
    n_tests = len(p_values)
    sorted_order = np.argsort(p_values)
    sorted_p = p_values[sorted_order]
    
    q_values = np.zeros(n_tests)
    q_min = 1.0
    for idx in range(n_tests - 1, -1, -1):
        rank = idx + 1
        q_val = (sorted_p[idx] * n_tests) / rank
        q_min = min(q_min, q_val)
        q_values[idx] = min(q_min, 1.0)
        
    adjusted_p = np.zeros(n_tests)
    adjusted_p[sorted_order] = q_values
    df_res["adjusted_p_value"] = adjusted_p
    
    # Classification
    df_res["significance"] = "Not Significant"
    sig_up = (df_res["adjusted_p_value"] < 0.05) & (df_res["log2_fold_change"] >= min_log2fc)
    sig_down = (df_res["adjusted_p_value"] < 0.05) & (df_res["log2_fold_change"] <= -min_log2fc)
    df_res.loc[sig_up, "significance"] = f"Upregulated in {group_1}"
    df_res.loc[sig_down, "significance"] = f"Upregulated in {group_2}"
    
    # Sort by absolute log2FC
    df_res["abs_log2fc"] = df_res["log2_fold_change"].abs()
    df_res = df_res.sort_values(by="abs_log2fc", ascending=False).drop(columns=["abs_log2fc"])
    
    return df_res.reset_index(drop=True)


def run_cross_condition_interface_dge(adata_solid, adata_microgel) -> pd.DataFrame:
    """
    Compares interface spots between Solid and Microgel scaffold conditions.
    """
    # Filter to interface spots (d between 0 and 150 µm)
    mask_s = (adata_solid.obs["dist_to_interface_um"] >= 0) & (adata_solid.obs["dist_to_interface_um"] <= 150)
    mask_m = (adata_microgel.obs["dist_to_interface_um"] >= 0) & (adata_microgel.obs["dist_to_interface_um"] <= 150)
    
    X_s = adata_solid.X[mask_s]
    X_m = adata_microgel.X[mask_m]
    
    genes = list(adata_solid.var_names)
    records = []
    
    for i, g in enumerate(genes):
        vs = X_s[:, i]
        vm = X_m[:, i]
        
        ms = float(np.mean(vs))
        mm = float(np.mean(vm))
        
        log2fc = float(np.log2((mm + 1e-4) / (ms + 1e-4)))
        t_stat, p_val = stats.ttest_ind(vm, vs, equal_var=False, nan_policy="omit")
        if np.isnan(p_val):
            p_val = 1.0
            
        records.append({
            "gene": g,
            "mean_microgel": mm,
            "mean_solid": ms,
            "log2fc_microgel_vs_solid": log2fc,
            "p_value": float(p_val),
            "category": adata_solid.var.loc[g, "marker_category"] if "marker_category" in adata_solid.var else "Other"
        })
        
    df_res = pd.DataFrame(records)
    # B-H correction
    p_values = df_res["p_value"].values
    n_tests = len(p_values)
    sorted_order = np.argsort(p_values)
    sorted_p = p_values[sorted_order]
    
    q_values = np.zeros(n_tests)
    q_min = 1.0
    for idx in range(n_tests - 1, -1, -1):
        rank = idx + 1
        q_val = (sorted_p[idx] * n_tests) / rank
        q_min = min(q_min, q_val)
        q_values[idx] = min(q_min, 1.0)
        
    adjusted_p = np.zeros(n_tests)
    adjusted_p[sorted_order] = q_values
    df_res["adjusted_p_value"] = adjusted_p
    
    df_res["enrichment"] = "Neutral"
    df_res.loc[(df_res["adjusted_p_value"] < 0.05) & (df_res["log2fc_microgel_vs_solid"] > 0.3), "enrichment"] = "Enriched in Microgel"
    df_res.loc[(df_res["adjusted_p_value"] < 0.05) & (df_res["log2fc_microgel_vs_solid"] < -0.3), "enrichment"] = "Enriched in Solid"
    
    return df_res.sort_values(by="log2fc_microgel_vs_solid", ascending=False).reset_index(drop=True)
