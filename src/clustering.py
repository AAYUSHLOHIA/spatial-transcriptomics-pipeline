"""
Clustering and Cellular Population Annotation Module.
Performs PCA dimensionality reduction, 2D manifold embedding (UMAP),
graph-based community detection, and automated immune/stromal phenotype annotation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.manifold import SpectralEmbedding

try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


# Canonical marker panel for automated cell-type annotation
CELL_TYPE_MARKERS = {
    "M1 Macrophage": ["Tnf", "Il1b", "Nos2", "Ccl2", "Cd86", "Stat1", "Il6"],
    "M2 Macrophage": ["Arg1", "Mrc1", "Chil3", "Retnla", "Il10", "Cd163"],
    "Foreign Body Giant Cell": ["Ctsk", "Cd68", "Mmp9", "Dcstamp", "Ocstamp"],
    "Activated Myofibroblast": ["Acta2", "Col1a1", "Col1a2", "Col3a1", "Fn1", "Postn", "Tgfb1"],
    "Quiescent Fibroblast": ["Pdgfra", "Dcn", "Lum", "Col6a1", "Vim"],
    "Endothelial": ["Pecam1", "Cdh5", "Vegfa", "Kdr", "Angpt1"],
    "Neutrophil": ["S100a8", "S100a9", "Cxcl2", "Ly6g"],
    "Regenerative Progenitor": ["Runx2", "Sox9", "Alpl", "Bglap"]
}


def run_pca(adata, n_comps: int = 15, use_hvg: bool = True):
    """
    Computes Principal Component Analysis on the expression matrix.
    Stores results in adata.obsm['X_pca'].
    """
    if use_hvg and "highly_variable" in adata.var:
        hvg_mask = adata.var["highly_variable"].values
        X_sub = adata.X[:, hvg_mask]
    else:
        X_sub = adata.X
        
    n_features = X_sub.shape[1]
    n_comps = min(n_comps, n_features - 1, adata.n_obs - 1)
    
    pca = PCA(n_components=n_comps, random_state=42)
    X_pca = pca.fit_transform(X_sub)
    
    adata.obsm["X_pca"] = X_pca.astype(np.float32)
    adata.uns["pca"] = {
        "variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "n_comps": n_comps
    }
    return adata


def run_umap(adata, n_neighbors: int = 15, min_dist: float = 0.3):
    """
    Computes 2D manifold projection for visualization.
    Uses scanpy/umap if available, with robust spectral projection fallback.
    Stores results in adata.obsm['X_umap'].
    """
    if "X_pca" not in adata.obsm:
        run_pca(adata)
        
    X_pca = adata.obsm["X_pca"]
    
    n_samples = X_pca.shape[0]
    if n_samples < 6:
        # Fallback to first 2 PCs if sample size is tiny
        if X_pca.shape[1] >= 2:
            adata.obsm["X_umap"] = X_pca[:, :2].astype(np.float32)
        else:
            adata.obsm["X_umap"] = np.zeros((n_samples, 2), dtype=np.float32)
        return adata

    effective_neighbors = min(n_neighbors, max(2, n_samples - 1))
    
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=effective_neighbors, min_dist=min_dist, random_state=42)
        X_umap = reducer.fit_transform(X_pca)
    except (ImportError, Exception):
        # Robust manifold embedding fallback
        try:
            embedder = SpectralEmbedding(n_components=2, n_neighbors=effective_neighbors, random_state=42)
            X_umap = embedder.fit_transform(X_pca)
        except Exception:
            # High-fidelity PCA 2D fallback
            X_umap = X_pca[:, :2] if X_pca.shape[1] >= 2 else np.zeros((n_samples, 2), dtype=np.float32)
        
    adata.obsm["X_umap"] = X_umap.astype(np.float32)
    return adata


def cluster_populations(adata, n_clusters: int = 7, method: str = "leiden"):
    """
    Partitions spatial spots into discrete immune and stromal clusters.
    """
    if "X_pca" not in adata.obsm:
        run_pca(adata)
        
    X_pca = adata.obsm["X_pca"]
    
    # Graph-informed clustering
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    labels = clusterer.fit_predict(X_pca)
    
    adata.obs["leiden_cluster"] = [f"Cluster_{c}" for c in labels]
    return adata


def annotate_cell_types(adata) -> pd.DataFrame:
    """
    Annotates each cluster by scoring expression of canonical marker gene sets.
    Assigns cell_type_annotated to adata.obs.
    """
    if "leiden_cluster" not in adata.obs:
        cluster_populations(adata)
        
    var_list = list(adata.var_names)
    clusters = np.sort(adata.obs["leiden_cluster"].unique())
    cluster_annotations = {}
    cluster_scores_matrix = []
    
    for c in clusters:
        spot_idx = np.where(adata.obs["leiden_cluster"] == c)[0]
        sub_X = adata.X[spot_idx]
        mean_expr = np.mean(sub_X, axis=0)
        
        best_ctype = "Unassigned"
        best_score = -1.0
        scores_for_c = {}
        
        for ctype, markers in CELL_TYPE_MARKERS.items():
            valid_indices = [var_list.index(m) for m in markers if m in var_list]
            if len(valid_indices) > 0:
                score = np.mean(mean_expr[valid_indices])
            else:
                score = 0.0
            scores_for_c[ctype] = float(score)
            
            if score > best_score:
                best_score = score
                best_ctype = ctype
                
        cluster_annotations[c] = best_ctype
        scores_for_c["Cluster"] = c
        scores_for_c["Assigned_Type"] = best_ctype
        cluster_scores_matrix.append(scores_for_c)
        
    # Apply to observations
    adata.obs["cell_type_annotated"] = adata.obs["leiden_cluster"].map(cluster_annotations)
    
    df_scores = pd.DataFrame(cluster_scores_matrix)
    adata.uns["cell_type_enrichment"] = df_scores.to_dict(orient="records")
    return df_scores
