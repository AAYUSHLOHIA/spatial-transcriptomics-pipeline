"""
Spatial Distance and Interface Gradient Analysis Module.
Quantifies cellular recruitment and continuous gene expression gradients
as a function of radial distance from the biomaterial-host boundary.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def calculate_radial_distances(
    adata,
    implant_center_xy: Optional[Tuple[float, float]] = None,
    implant_radius_um: Optional[float] = None
):
    """
    Computes Euclidean distance of each Visium spot from the biomaterial boundary.
    Signed distance:
    < 0: Inside implant / porous scaffold interior
    > 0: In pericapsular or host tissue outside implant boundary
    """
    if "spatial" not in adata.obsm:
        raise ValueError("Spatial coordinates missing in adata.obsm['spatial']")
        
    coords = adata.obsm["spatial"]
    
    # If center and radius are stored in adata.uns, use them
    if implant_center_xy is None:
        if "implant_center_xy" in adata.uns:
            center = adata.uns["implant_center_xy"]
        else:
            # Estimate center as centroid of spatial bounding box
            center = (np.mean(coords[:, 0]), np.mean(coords[:, 1]))
    else:
        center = implant_center_xy
        
    if implant_radius_um is None:
        implant_radius_um = adata.uns.get("implant_radius_um", 1100.0)
        
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
    dist_from_center = np.sqrt(dx**2 + dy**2)
    dist_to_interface = dist_from_center - implant_radius_um
    
    adata.obs["dist_from_center_um"] = dist_from_center
    adata.obs["dist_to_interface_um"] = dist_to_interface
    return adata


def assign_distance_zones(adata, bin_cutoffs: Optional[List[float]] = None):
    """
    Categorizes spots into discrete histological zones around the implant:
    - Scaffold_Infiltrate: Inside porous scaffold (< 0 µm)
    - Direct_Interface: 0 - 120 µm
    - Pericapsular_Capsule: 120 - 350 µm
    - Reactive_Stroma: 350 - 700 µm
    - Distal_Native: > 700 µm
    """
    if "dist_to_interface_um" not in adata.obs:
        calculate_radial_distances(adata)
        
    dists = adata.obs["dist_to_interface_um"].values
    zones = []
    
    for d in dists:
        if d < 0:
            zones.append("Scaffold_Infiltrate")
        elif d <= 120:
            zones.append("Direct_Interface")
        elif d <= 350:
            zones.append("Pericapsular_Capsule")
        elif d <= 700:
            zones.append("Reactive_Stroma")
        else:
            zones.append("Distal_Native")
            
    adata.obs["distance_zone"] = pd.Categorical(
        zones,
        categories=["Scaffold_Infiltrate", "Direct_Interface", "Pericapsular_Capsule", "Reactive_Stroma", "Distal_Native"]
    )
    return adata


def compute_radial_gene_gradients(
    adata,
    genes: List[str],
    bin_width_um: float = 60.0,
    min_dist_um: float = -400.0,
    max_dist_um: float = 1200.0
) -> pd.DataFrame:
    """
    Calculates continuous spatial radial gene expression profiles f(r)
    by distance binning from the implant interface.
    """
    if "dist_to_interface_um" not in adata.obs:
        calculate_radial_distances(adata)
        
    dists = adata.obs["dist_to_interface_um"].values
    var_list = list(adata.var_names)
    
    bins = np.arange(min_dist_um, max_dist_um + bin_width_um, bin_width_um)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    results = []
    
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (dists >= low) & (dists < high)
        n_spots_in_bin = np.sum(mask)
        
        row_dict = {
            "bin_min_um": low,
            "bin_max_um": high,
            "bin_center_um": bin_centers[i],
            "spot_count": int(n_spots_in_bin)
        }
        
        if n_spots_in_bin > 0:
            sub_X = adata.X[mask]
            for g in genes:
                if g in var_list:
                    g_idx = var_list.index(g)
                    vals = sub_X[:, g_idx]
                    row_dict[f"{g}_mean"] = float(np.mean(vals))
                    row_dict[f"{g}_sem"] = float(np.std(vals) / np.sqrt(n_spots_in_bin))
                else:
                    row_dict[f"{g}_mean"] = np.nan
                    row_dict[f"{g}_sem"] = np.nan
        else:
            for g in genes:
                row_dict[f"{g}_mean"] = np.nan
                row_dict[f"{g}_sem"] = np.nan
                
        results.append(row_dict)
        
    df_profiles = pd.DataFrame(results)
    return df_profiles


def compute_cellular_recruitment_profiles(
    adata,
    cell_type_col: str = "cell_type_annotated",
    bin_width_um: float = 75.0,
    min_dist_um: float = -400.0,
    max_dist_um: float = 1200.0
) -> pd.DataFrame:
    """
    Computes spatial recruitment density and cellular frequency of each
    annotated immune and stromal population as a function of radial distance.
    """
    if "dist_to_interface_um" not in adata.obs:
        calculate_radial_distances(adata)
        
    dists = adata.obs["dist_to_interface_um"].values
    types = adata.obs[cell_type_col].values
    unique_types = np.unique(types)
    
    bins = np.arange(min_dist_um, max_dist_um + bin_width_um, bin_width_um)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    rows = []
    
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (dists >= low) & (dists < high)
        total_in_bin = np.sum(mask)
        
        rec = {
            "bin_center_um": bin_centers[i],
            "total_spots": int(total_in_bin)
        }
        
        if total_in_bin > 0:
            sub_types = types[mask]
            for ct in unique_types:
                count_ct = np.sum(sub_types == ct)
                rec[f"{ct}_count"] = int(count_ct)
                rec[f"{ct}_freq"] = float(count_ct / total_in_bin)
        else:
            for ct in unique_types:
                rec[f"{ct}_count"] = 0
                rec[f"{ct}_freq"] = 0.0
                
        rows.append(rec)
        
    return pd.DataFrame(rows)
