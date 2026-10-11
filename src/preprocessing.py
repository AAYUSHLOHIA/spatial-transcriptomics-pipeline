"""
Preprocessing Module for Spatial Transcriptomics.
Performs Quality Control (QC), Library Size Normalization, Log1p Transformation,
Highly Variable Gene (HVG) selection, and Spatial Autocorrelation (Moran's I).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import sparse
from scipy.spatial import distance_matrix

try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


def calculate_qc_metrics(adata, min_counts: int = 10, min_genes: int = 5):
    """
    Computes quality control metrics per spot and per gene.
    Filters spots that fail quality thresholds.
    """
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X.copy()
    
    # Per-spot metrics
    total_counts = np.sum(X, axis=1)
    n_genes_by_counts = np.sum(X > 0, axis=1)
    
    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes_by_counts"] = n_genes_by_counts
    
    # Filter mask
    keep_mask = (total_counts >= min_counts) & (n_genes_by_counts >= min_genes)
    
    # Per-gene metrics
    n_cells_by_counts = np.sum(X > 0, axis=0)
    mean_counts = np.mean(X, axis=0)
    adata.var["n_cells_by_counts"] = n_cells_by_counts
    adata.var["mean_counts"] = mean_counts
    
    # Subsample adata if needed
    if not np.all(keep_mask):
        adata.X = adata.X[keep_mask]
        adata.obs = adata.obs.iloc[keep_mask].copy()
        if "spatial" in adata.obsm:
            adata.obsm["spatial"] = adata.obsm["spatial"][keep_mask]
            
    adata.uns["qc_summary"] = {
        "initial_spots": len(keep_mask),
        "retained_spots": int(np.sum(keep_mask)),
        "mean_library_size": float(np.mean(total_counts[keep_mask])),
        "mean_genes_per_spot": float(np.mean(n_genes_by_counts[keep_mask]))
    }
    return adata


def normalize_and_log(adata, target_sum: float = 1e4):
    """
    Performs library size normalization followed by natural logarithm (log1p) transformation.
    Preserves raw counts in layers['counts'].
    """
    # Save raw counts
    if hasattr(adata, "layers"):
        adata.layers["counts"] = adata.X.copy()
        
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X.copy()
    
    counts_per_spot = np.sum(X, axis=1, keepdims=True)
    counts_per_spot[counts_per_spot == 0] = 1.0  # Avoid division by zero
    
    # Scale to target sum
    norm_X = (X / counts_per_spot) * target_sum
    
    # log1p transformation
    log_norm_X = np.log1p(norm_X)
    
    adata.X = log_norm_X.astype(np.float32)
    adata.uns["normalization"] = {
        "target_sum": target_sum,
        "transformed": "log1p"
    }
    return adata


def select_highly_variable_genes(adata, top_n: int = 35) -> List[str]:
    """
    Selects highly variable genes based on variance-to-mean dispersion ratio.
    """
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    
    # Dispersion measure
    dispersion = np.where(mean > 0, var / mean, 0)
    adata.var["dispersions"] = dispersion
    adata.var["dispersions_norm"] = (dispersion - np.mean(dispersion)) / (np.std(dispersion) + 1e-8)
    
    # Top variable genes
    top_indices = np.argsort(dispersion)[::-1][:top_n]
    hvg_mask = np.zeros(adata.n_vars, dtype=bool)
    hvg_mask[top_indices] = True
    adata.var["highly_variable"] = hvg_mask
    
    return list(adata.var_names[hvg_mask])


def compute_spatial_moran_i(adata, gene_name: str, k_neighbors: int = 6) -> float:
    """
    Calculates spatial autocorrelation (Moran's I) for a specific gene
    to quantify spatial localized patterning across the tissue.
    
    Moran's I = (N / W) * (sum_i sum_j w_ij (x_i - mean_x)(x_j - mean_x)) / sum_i (x_i - mean_x)^2
    """
    if "spatial" not in adata.obsm:
        return 0.0
        
    if gene_name not in adata.var_names:
        return 0.0
        
    gene_idx = list(adata.var_names).index(gene_name)
    x = adata.X[:, gene_idx]
    
    coords = adata.obsm["spatial"]
    n = len(coords)
    if n < 2:
        return 0.0
        
    # Compute pairwise Euclidean distance in spatial coordinates
    dists = distance_matrix(coords, coords)
    
    # Build k-nearest neighbor spatial adjacency matrix W
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        # Nearest neighbors excluding self
        nn_idx = np.argsort(dists[i])[1:k_neighbors + 1]
        W[i, nn_idx] = 1.0
        
    # Row-normalize W
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums
    
    # Moran's I
    x_mean = np.mean(x)
    z = x - x_mean
    z_ss = np.sum(z**2)
    
    if z_ss == 0:
        return 0.0
        
    s0 = np.sum(W)
    numerator = np.sum(W * np.outer(z, z))
    moran_i = (n / s0) * (numerator / z_ss)
    
    return float(moran_i)
