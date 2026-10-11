"""
Pathway Scoring and Immune Activation Analysis Module.
Quantifies functional pathway signatures per spatial spot:
- Foreign Body Reaction (FBR) Fibrosis
- Acute / Chronic Pro-inflammatory Immune Activation
- Pro-Regenerative M2 Tissue Resolution
- Neo-Vascularization & Angiogenesis
- Foreign Body Giant Cell (FBGC) Activation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


PATHWAY_DEFINITIONS = {
    "FBR_Fibrotic_Encapsulation": [
        "Acta2", "Col1a1", "Col1a2", "Col3a1", "Fn1", "Postn", "Tgfb1", "Timp1"
    ],
    "Pro_Inflammatory_M1": [
        "Tnf", "Il1b", "Nos2", "Ccl2", "Cd86", "Stat1", "Cxcl10", "Il6"
    ],
    "Pro_Regenerative_M2": [
        "Arg1", "Mrc1", "Chil3", "Retnla", "Il10", "Cd163", "Mmp12"
    ],
    "Neo_Vascularization": [
        "Pecam1", "Cdh5", "Vegfa", "Kdr", "Angpt1", "Tek"
    ],
    "Foreign_Body_Giant_Cell": [
        "Ctsk", "Cd68", "Mmp9", "Dcstamp", "Ocstamp", "Itgam"
    ]
}


def score_pathways(adata, pathways: Optional[Dict[str, List[str]]] = None):
    """
    Computes pathway activity scores for each spatial spot using
    a robust background-subtracted gene set scoring algorithm (similar to Scanpy score_genes).
    """
    if pathways is None:
        pathways = PATHWAY_DEFINITIONS
        
    var_list = list(adata.var_names)
    X = adata.X
    
    # Global mean gene expression for background reference matching
    mean_gene_expr = np.mean(X, axis=0)
    
    for pathway_name, gene_list in pathways.items():
        valid_indices = [var_list.index(g) for g in gene_list if g in var_list]
        
        if len(valid_indices) == 0:
            adata.obs[f"pathway_{pathway_name}"] = 0.0
            continue
            
        # Target gene set mean expression per spot
        target_expr = np.mean(X[:, valid_indices], axis=1)
        
        # Background control: select genes of similar average expression
        # Sample 3 control genes per target gene
        control_indices = []
        for tidx in valid_indices:
            t_mean = mean_gene_expr[tidx]
            diffs = np.abs(mean_gene_expr - t_mean)
            # Find nearest matched genes not in the target set
            nearest = np.argsort(diffs)
            matched = [idx for idx in nearest if idx not in valid_indices][:3]
            control_indices.extend(matched)
            
        control_indices = list(set(control_indices))
        if len(control_indices) > 0:
            ctrl_expr = np.mean(X[:, control_indices], axis=1)
        else:
            ctrl_expr = np.mean(X, axis=1)
            
        # Raw score = Target - Control
        raw_score = target_expr - ctrl_expr
        
        # Z-score normalization for standardized cross-sample comparison
        std = np.std(raw_score)
        z_score = (raw_score - np.mean(raw_score)) / (std if std > 1e-6 else 1.0)
        
        adata.obs[f"pathway_{pathway_name}"] = z_score
        
    return adata


def compare_pathways_across_morphologies(datasets_dict: Dict[str, object]) -> pd.DataFrame:
    """
    Compares pathway scores across biomaterial morphology conditions
    (Solid vs Microporous vs Microgel MAP) across spatial zones.
    """
    records = []
    
    for cond_name, adata in datasets_dict.items():
        if "pathway_FBR_Fibrotic_Encapsulation" not in adata.obs:
            score_pathways(adata)
            
        pathway_cols = [col for col in adata.obs.columns if col.startswith("pathway_")]
        
        for zone in adata.obs["distance_zone"].unique():
            zone_spots = adata.obs[adata.obs["distance_zone"] == zone]
            if len(zone_spots) == 0:
                continue
                
            rec = {
                "condition": cond_name,
                "distance_zone": zone,
                "n_spots": len(zone_spots)
            }
            
            for pcol in pathway_cols:
                clean_name = pcol.replace("pathway_", "")
                rec[f"{clean_name}_mean"] = float(np.mean(zone_spots[pcol]))
                rec[f"{clean_name}_std"] = float(np.std(zone_spots[pcol]))
                
            records.append(rec)
            
    return pd.DataFrame(records)
