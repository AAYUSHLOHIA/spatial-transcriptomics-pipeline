"""
Spatial Dataset Module for Biomaterial-Host Interface Analysis.
Provides AnnData abstraction and biologically grounded synthetic 10x Genomics
Visium spatial transcriptomics generator for biomaterial implants.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

try:
    import anndata
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False


class SpatialAnnData:
    """
    Lightweight, high-performance AnnData-compatible container for spatial transcriptomics.
    Seamlessly operates without requiring external anndata package while preserving
    standard AnnData syntax (.X, .obs, .var, .obsm, .uns, .layers).
    """
    def __init__(
        self,
        X: np.ndarray,
        obs: pd.DataFrame,
        var: pd.DataFrame,
        obsm: Optional[Dict[str, np.ndarray]] = None,
        uns: Optional[Dict] = None
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.obs = obs.copy()
        self.var = var.copy()
        self.obsm = obsm if obsm is not None else {}
        self.uns = uns if uns is not None else {}
        self.layers = {}
        
        # Verify alignment
        assert self.X.shape[0] == len(self.obs), f"X rows ({self.X.shape[0]}) != obs rows ({len(self.obs)})"
        assert self.X.shape[1] == len(self.var), f"X cols ({self.X.shape[1]}) != var rows ({len(self.var)})"

    @property
    def shape(self) -> Tuple[int, int]:
        return self.X.shape

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    @property
    def n_vars(self) -> int:
        return self.X.shape[1]

    @property
    def obs_names(self) -> pd.Index:
        return self.obs.index

    @property
    def var_names(self) -> pd.Index:
        return self.var.index

    def copy(self):
        new_obsm = {k: v.copy() for k, v in self.obsm.items()}
        new_uns = {k: v for k, v in self.uns.items()}
        return SpatialAnnData(self.X.copy(), self.obs.copy(), self.var.copy(), new_obsm, new_uns)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, index=self.obs.index, columns=self.var.index)

    def to_dict(self) -> Dict:
        return {
            "n_spots": self.n_obs,
            "n_genes": self.n_vars,
            "obs": self.obs.to_dict(orient="list"),
            "var": self.var.to_dict(orient="list"),
            "obsm_keys": list(self.obsm.keys()),
            "uns": {k: str(v) for k, v in self.uns.items()}
        }

    def save_csv(self, prefix_path: str):
        os.makedirs(os.path.dirname(prefix_path) or '.', exist_ok=True)
        pd.DataFrame(self.X, index=self.obs.index, columns=self.var.index).to_csv(f"{prefix_path}_matrix.csv")
        self.obs.to_csv(f"{prefix_path}_obs.csv")
        self.var.to_csv(f"{prefix_path}_var.csv")
        if "spatial" in self.obsm:
            np.save(f"{prefix_path}_spatial.npy", self.obsm["spatial"])


def create_anndata(X: np.ndarray, obs: pd.DataFrame, var: pd.DataFrame, obsm: Dict, uns: Dict):
    """Creates real AnnData if available, otherwise SpatialAnnData."""
    if HAS_ANNDATA:
        adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm, uns=uns)
        return adata
    return SpatialAnnData(X=X, obs=obs, var=var, obsm=obsm, uns=uns)


# -------------------------------------------------------------------------
# Curated Biomaterial-Host Interface Gene Panel (54 Genes)
# -------------------------------------------------------------------------
GENE_PANEL = {
    # M1 Pro-inflammatory Macrophages
    "Tnf": {"category": "M1_Macrophage", "weight": 2.5},
    "Il1b": {"category": "M1_Macrophage", "weight": 2.8},
    "Nos2": {"category": "M1_Macrophage", "weight": 2.4},
    "Ccl2": {"category": "M1_Macrophage", "weight": 2.2},
    "Cd86": {"category": "M1_Macrophage", "weight": 1.9},
    "Stat1": {"category": "M1_Macrophage", "weight": 1.8},
    "Cxcl10": {"category": "M1_Macrophage", "weight": 2.0},
    "Il6": {"category": "M1_Macrophage", "weight": 2.1},

    # M2 Pro-regenerative / Anti-inflammatory Macrophages
    "Arg1": {"category": "M2_Macrophage", "weight": 2.6},
    "Mrc1": {"category": "M2_Macrophage", "weight": 2.5},   # CD206
    "Chil3": {"category": "M2_Macrophage", "weight": 2.4},  # Ym1
    "Retnla": {"category": "M2_Macrophage", "weight": 2.3}, # Fizz1
    "Il10": {"category": "M2_Macrophage", "weight": 2.2},
    "Cd163": {"category": "M2_Macrophage", "weight": 2.1},
    "Mmp12": {"category": "M2_Macrophage", "weight": 1.9},

    # Foreign Body Giant Cells (FBGCs) - Fused Macrophages at implant surface
    "Ctsk": {"category": "FBGC", "weight": 2.7},
    "Cd68": {"category": "FBGC", "weight": 2.3},
    "Mmp9": {"category": "FBGC", "weight": 2.5},
    "Dcstamp": {"category": "FBGC", "weight": 2.8},
    "Ocstamp": {"category": "FBGC", "weight": 2.6},
    "Itgam": {"category": "FBGC", "weight": 2.0},   # CD11b

    # Myofibroblasts & Fibrotic Capsule (FBR Matrix)
    "Acta2": {"category": "Myofibroblast", "weight": 3.0}, # alpha-SMA
    "Col1a1": {"category": "Myofibroblast", "weight": 2.9},
    "Col1a2": {"category": "Myofibroblast", "weight": 2.7},
    "Col3a1": {"category": "Myofibroblast", "weight": 2.8},
    "Fn1": {"category": "Myofibroblast", "weight": 2.6},
    "Postn": {"category": "Myofibroblast", "weight": 2.5},  # Periostin
    "Tgfb1": {"category": "Myofibroblast", "weight": 2.7},
    "Timp1": {"category": "Myofibroblast", "weight": 2.2},

    # Quiescent Stromal Fibroblasts
    "Pdgfra": {"category": "Stromal_Fibroblast", "weight": 2.4},
    "Dcn": {"category": "Stromal_Fibroblast", "weight": 2.6},     # Decorin
    "Lum": {"category": "Stromal_Fibroblast", "weight": 2.3},     # Lumican
    "Col6a1": {"category": "Stromal_Fibroblast", "weight": 2.1},
    "Vim": {"category": "Stromal_Fibroblast", "weight": 2.0},

    # Endothelial & Neo-vascularization (Angiogenesis)
    "Pecam1": {"category": "Endothelial", "weight": 2.7}, # CD31
    "Cdh5": {"category": "Endothelial", "weight": 2.5},   # VE-Cadherin
    "Vegfa": {"category": "Endothelial", "weight": 2.8},
    "Kdr": {"category": "Endothelial", "weight": 2.4},     # VEGFR2
    "Angpt1": {"category": "Endothelial", "weight": 2.2},
    "Tek": {"category": "Endothelial", "weight": 2.1},     # Tie2

    # Neutrophils & Acute Phase Granulocytes
    "S100a8": {"category": "Neutrophil", "weight": 2.6},
    "S100a9": {"category": "Neutrophil", "weight": 2.5},
    "Cxcl2": {"category": "Neutrophil", "weight": 2.2},
    "Ly6g": {"category": "Neutrophil", "weight": 2.4},

    # Regenerative / Progenitor Markers
    "Runx2": {"category": "Progenitor", "weight": 2.0},
    "Sox9": {"category": "Progenitor", "weight": 1.9},
    "Alpl": {"category": "Progenitor", "weight": 2.1},
    "Bglap": {"category": "Progenitor", "weight": 1.8},

    # Housekeeping & Baseline Control Genes
    "Actb": {"category": "Housekeeping", "weight": 1.0},
    "Gapdh": {"category": "Housekeeping", "weight": 1.0},
    "Rps18": {"category": "Housekeeping", "weight": 1.0},
    "B2m": {"category": "Housekeeping", "weight": 1.0},
    "Hprt1": {"category": "Housekeeping", "weight": 1.0},
}


def generate_biomaterial_visium_dataset(
    condition: str = "solid",
    grid_size: int = 40,
    spacing_um: float = 100.0,
    implant_radius_um: float = 1100.0,
    random_seed: int = 42
) -> SpatialAnnData:
    """
    Simulates a 10x Genomics Visium spatial transcriptomics capture area surrounding
    a biomaterial implant with realistic spatial biology.
    
    Parameters:
    -----------
    condition : str
        'solid': Solid non-porous implant (severe foreign body capsule, high Acta2/Tgfb1, minimal infiltration)
        'microporous': Microporous scaffold (pore size ~30µm, partial infiltration, moderate capsule)
        'microgel': Microporous Annealed Particle (MAP) microgel scaffold (~70µm interconnected void pores,
                    extensive infiltration, pro-regenerative M2 dominance, high Vegfa/Pecam1, no fibrotic capsule)
    grid_size : int
        Square grid dimension (grid_size x grid_size spots)
    spacing_um : float
        Center-to-center spot spacing in micrometers (default 100 µm for 10x Visium)
    implant_radius_um : float
        Radius of implant zone in center of tissue (default 1100 µm)
    random_seed : int
        Reproducibility seed
    """
    rng = np.random.default_rng(random_seed)
    
    # 1. Generate Visium Hex/Grid Coordinates
    center_offset = (grid_size - 1) * spacing_um / 2.0
    spots = []
    
    for r in range(grid_size):
        for c in range(grid_size):
            # Hexagonal staggered offset every other row
            x = c * spacing_um + (spacing_um * 0.5 if r % 2 == 1 else 0.0)
            y = r * (spacing_um * np.sqrt(3) / 2.0)
            
            # Center coordinates around (0, 0)
            x_centered = x - center_offset
            y_centered = y - center_offset
            
            # Distance from implant center
            dist_from_center = np.sqrt(x_centered**2 + y_centered**2)
            
            # Signed distance to implant boundary:
            # < 0 means inside implant; > 0 means in host tissue outside implant boundary
            dist_to_interface = dist_from_center - implant_radius_um
            
            spots.append({
                "spot_id": f"AA_{condition.upper()}_{r:02d}_{c:02d}",
                "array_row": r,
                "array_col": c,
                "x_um": float(x),
                "y_um": float(y),
                "x_centered_um": float(x_centered),
                "y_centered_um": float(y_centered),
                "dist_from_center_um": float(dist_from_center),
                "dist_to_interface_um": float(dist_to_interface),
            })
            
    df_spots = pd.DataFrame(spots)
    
    # 2. Assign Tissue & Cellular Compartments based on Biomaterial Condition
    # Conditions dictate infiltration into implant interior (dist_to_interface < 0)
    # and pericapsular fibrotic reaction outside implant (dist_to_interface > 0)
    
    conditions_meta = {
        "solid": {
            "name": "Solid Non-Porous Implant",
            "porosity_pct": 0.0,
            "mean_pore_size_um": 0.0,
            "modulus_kpa": 1500.0,
            "max_infiltration_depth_um": 0.0,
            "capsule_thickness_um": 280.0,
        },
        "microporous": {
            "name": "Microporous Hydrogel Scaffold",
            "porosity_pct": 28.0,
            "mean_pore_size_um": 28.0,
            "modulus_kpa": 12.0,
            "max_infiltration_depth_um": 350.0,
            "capsule_thickness_um": 140.0,
        },
        "microgel": {
            "name": "MAP Microgel Granular Scaffold",
            "porosity_pct": 42.0,
            "mean_pore_size_um": 75.0,
            "modulus_kpa": 2.5,
            "max_infiltration_depth_um": 1100.0, # Fully infiltrated
            "capsule_thickness_um": 30.0, # Negligible / absent capsule
        }
    }
    
    cfg = conditions_meta[condition]
    
    # Filter or retain spots based on biomaterial morphology
    # In a solid implant, the core has no tissue spots (non-permeable synthetic material)
    # In microgel/microporous, tissue infiltrates through interconnected pores!
    valid_spots_mask = []
    cell_types = []
    spatial_zones = []
    
    for _, row in df_spots.iterrows():
        d = row["dist_to_interface_um"]
        
        if d < 0: # Inside implant body
            depth = abs(d)
            if condition == "solid":
                # Solid implant has no cellular tissue inside
                valid_spots_mask.append(False)
                cell_types.append("Acellular_Material")
                spatial_zones.append("Material_Core")
                continue
            elif condition == "microporous":
                # Microporous allows shallow cellular infiltration up to ~350 µm
                if depth > cfg["max_infiltration_depth_um"]:
                    valid_spots_mask.append(False)
                    cell_types.append("Acellular_Material")
                    spatial_zones.append("Material_Core")
                    continue
                else:
                    valid_spots_mask.append(True)
                    spatial_zones.append("Scaffold_Infiltrate")
                    # Mixed infiltrate: infiltrating macrophages and endothelial sprouts
                    p_m2 = 0.55
                    p_endo = 0.25
                    rand_val = rng.random()
                    if rand_val < p_m2:
                        cell_types.append("M2_Macrophage")
                    elif rand_val < p_m2 + p_endo:
                        cell_types.append("Endothelial")
                    else:
                        cell_types.append("M1_Macrophage")
                    continue
            else: # microgel: fully interconnected porous network
                valid_spots_mask.append(True)
                spatial_zones.append("Microgel_Pore_Network")
                # High pro-regenerative recruitment & neo-vascularization!
                rand_val = rng.random()
                if rand_val < 0.45:
                    cell_types.append("M2_Macrophage")
                elif rand_val < 0.75:
                    cell_types.append("Endothelial")
                elif rand_val < 0.90:
                    cell_types.append("Progenitor")
                else:
                    cell_types.append("Stromal_Fibroblast")
                continue
                
        else: # Outside implant (Host Tissue Interface & Stroma)
            valid_spots_mask.append(True)
            if d <= 120.0:
                # Direct interface zone
                spatial_zones.append("Implant_Interface_Zone")
                if condition == "solid":
                    # Severe FBR: Foreign Body Giant Cells & M1 macrophages
                    rand_val = rng.random()
                    if rand_val < 0.45:
                        cell_types.append("FBGC")
                    elif rand_val < 0.80:
                        cell_types.append("M1_Macrophage")
                    else:
                        cell_types.append("Myofibroblast")
                elif condition == "microporous":
                    rand_val = rng.random()
                    if rand_val < 0.35:
                        cell_types.append("M2_Macrophage")
                    elif rand_val < 0.65:
                        cell_types.append("M1_Macrophage")
                    elif rand_val < 0.85:
                        cell_types.append("Endothelial")
                    else:
                        cell_types.append("Myofibroblast")
                else: # microgel
                    # Minimal FBGC, dominated by M2 macrophages and neo-vessels
                    rand_val = rng.random()
                    if rand_val < 0.60:
                        cell_types.append("M2_Macrophage")
                    elif rand_val < 0.85:
                        cell_types.append("Endothelial")
                    else:
                        cell_types.append("Stromal_Fibroblast")
                        
            elif d <= 350.0:
                # Pericapsular / Capsule Zone
                spatial_zones.append("Pericapsular_Capsule_Zone")
                if condition == "solid":
                    # Thick fibrous capsule of activated alpha-SMA+ myofibroblasts
                    rand_val = rng.random()
                    if rand_val < 0.70:
                        cell_types.append("Myofibroblast")
                    elif rand_val < 0.85:
                        cell_types.append("M1_Macrophage")
                    else:
                        cell_types.append("Neutrophil")
                elif condition == "microporous":
                    rand_val = rng.random()
                    if rand_val < 0.45:
                        cell_types.append("Myofibroblast")
                    elif rand_val < 0.75:
                        cell_types.append("M2_Macrophage")
                    else:
                        cell_types.append("Stromal_Fibroblast")
                else: # microgel
                    rand_val = rng.random()
                    if rand_val < 0.45:
                        cell_types.append("M2_Macrophage")
                    elif rand_val < 0.75:
                        cell_types.append("Endothelial")
                    else:
                        cell_types.append("Stromal_Fibroblast")
                        
            elif d <= 700.0:
                # Reactive stroma zone
                spatial_zones.append("Reactive_Stroma_Zone")
                rand_val = rng.random()
                if rand_val < 0.50:
                    cell_types.append("Stromal_Fibroblast")
                elif rand_val < 0.75:
                    cell_types.append("Endothelial")
                elif rand_val < 0.90:
                    cell_types.append("M2_Macrophage" if condition == "microgel" else "M1_Macrophage")
                else:
                    cell_types.append("Neutrophil" if condition == "solid" else "Progenitor")
            else:
                # Distal Unperturbed Parenchyma / Normal Stroma
                spatial_zones.append("Distal_Native_Stroma")
                rand_val = rng.random()
                if rand_val < 0.65:
                    cell_types.append("Stromal_Fibroblast")
                elif rand_val < 0.85:
                    cell_types.append("Endothelial")
                else:
                    cell_types.append("Progenitor")

    # Filter to valid tissue-containing spots
    df_valid = df_spots[valid_spots_mask].copy().reset_index(drop=True)
    df_valid["cell_type"] = [c for c, v in zip(cell_types, valid_spots_mask) if v]
    df_valid["spatial_zone"] = [z for z, v in zip(spatial_zones, valid_spots_mask) if v]
    df_valid["implant_condition"] = condition

    # 3. Generate Expression Matrix based on Cell Type Signatures & Distance Gradients
    genes = list(GENE_PANEL.keys())
    n_spots = len(df_valid)
    n_genes = len(genes)
    
    # Base expression matrix
    X_counts = np.zeros((n_spots, n_genes), dtype=np.float32)
    
    # Baseline library size (mean ~3500 UMI per spot)
    library_sizes = rng.negative_binomial(35, 0.01, size=n_spots).astype(np.float32)
    
    for i, (_, row) in enumerate(df_valid.iterrows()):
        ctype = row["cell_type"]
        d = row["dist_to_interface_um"]
        
        # Distance decay / gradient factor for implant microenvironment
        # Closer to implant = higher inflammatory & fibrotic stimulus
        prox_factor = np.exp(-max(0.0, d) / 350.0) if d >= 0 else 1.2
        
        for j, g in enumerate(genes):
            g_info = GENE_PANEL[g]
            cat = g_info["category"]
            weight = g_info["weight"]
            
            # Baseline expression
            lam = 0.8
            
            # Category-specific elevation
            if cat == "Housekeeping":
                lam = 3.5
            elif cat == "M1_Macrophage" and ctype == "M1_Macrophage":
                lam = 4.2 * weight * (1.0 + 0.8 * prox_factor if condition == "solid" else 1.0)
            elif cat == "M2_Macrophage" and ctype == "M2_Macrophage":
                lam = 4.5 * weight * (1.5 if condition == "microgel" else 1.0)
            elif cat == "FBGC" and ctype == "FBGC":
                lam = 5.0 * weight * (1.8 if condition == "solid" else 1.0)
            elif cat == "Myofibroblast" and ctype == "Myofibroblast":
                lam = 4.8 * weight * (1.6 if condition == "solid" else 0.8)
            elif cat == "Stromal_Fibroblast" and ctype == "Stromal_Fibroblast":
                lam = 3.8 * weight
            elif cat == "Endothelial" and ctype == "Endothelial":
                lam = 4.0 * weight * (1.6 if condition == "microgel" else 1.0)
            elif cat == "Neutrophil" and ctype == "Neutrophil":
                lam = 4.0 * weight
            elif cat == "Progenitor" and ctype == "Progenitor":
                lam = 3.2 * weight
                
            # Additional morphology-driven pathway modulation:
            # Solid implants stimulate high Tgfb1, Acta2, Col1a1 near the boundary
            if condition == "solid" and g in ["Acta2", "Col1a1", "Col3a1", "Tgfb1"] and d >= 0 and d <= 300:
                lam += 6.0 * np.exp(-d / 150.0)
            # Microgel scaffolds stimulate high Vegfa, Arg1, Pecam1
            elif condition == "microgel" and g in ["Vegfa", "Arg1", "Mrc1", "Pecam1"]:
                lam += 4.5 * (1.0 if d < 200 else 0.5)

            # Draw Poisson / Negative Binomial counts
            count = rng.poisson(lam)
            X_counts[i, j] = count

    # Scale by spot library size factor
    mean_lib = np.mean(library_sizes)
    size_factors = (library_sizes / mean_lib)[:, np.newaxis]
    X_counts = np.round(X_counts * (0.6 + 0.8 * size_factors)).astype(np.float32)

    # Calculate summary metrics per spot
    df_valid["total_counts"] = np.sum(X_counts, axis=1)
    df_valid["n_genes"] = np.sum(X_counts > 0, axis=1)
    
    # Construct var DataFrame
    df_var = pd.DataFrame({
        "gene_name": genes,
        "marker_category": [GENE_PANEL[g]["category"] for g in genes],
        "weight": [GENE_PANEL[g]["weight"] for g in genes]
    }, index=genes)
    
    # Construct spatial coordinates matrix (n_spots x 2)
    spatial_coords = df_valid[["x_um", "y_um"]].values
    obsm = {"spatial": spatial_coords}
    
    uns = {
        "condition": condition,
        "parameters": cfg,
        "gene_panel_size": n_genes,
        "implant_radius_um": implant_radius_um,
        "spacing_um": spacing_um
    }
    
    df_valid.index = df_valid["spot_id"]
    return create_anndata(X=X_counts, obs=df_valid, var=df_var, obsm=obsm, uns=uns)


def load_10x_visium_directory(
    data_dir: str,
    matrix_filename: Optional[str] = None,
    spatial_filename: Optional[str] = None
) -> SpatialAnnData:
    """
    Loads 10x Genomics Visium Spatial Gene Expression data from a 10x Space Ranger output folder.
    Supports both standard Space Ranger directory trees and tabular matrix exports.

    Parameters:
    -----------
    data_dir : str
        Path to Space Ranger output directory containing expression data and 'spatial/' subfolder.
    matrix_filename : str, optional
        Specific matrix CSV/TSV filename if overriding standard lookup.
    spatial_filename : str, optional
        Specific tissue positions CSV filename if overriding standard lookup.
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Visium dataset directory not found at: {data_dir}")

    # 1. Locate expression matrix
    possible_matrix_paths = [
        os.path.join(data_dir, matrix_filename) if matrix_filename else None,
        os.path.join(data_dir, "filtered_feature_bc_matrix.csv"),
        os.path.join(data_dir, "matrix.csv"),
        os.path.join(data_dir, "raw_feature_bc_matrix.csv"),
    ]
    matrix_path = next((p for p in possible_matrix_paths if p and os.path.exists(p)), None)

    # 2. Locate spatial coordinates
    possible_spatial_paths = [
        os.path.join(data_dir, spatial_filename) if spatial_filename else None,
        os.path.join(data_dir, "spatial", "tissue_positions_list.csv"),
        os.path.join(data_dir, "spatial", "tissue_positions.csv"),
        os.path.join(data_dir, "tissue_positions_list.csv"),
    ]
    spatial_path = next((p for p in possible_spatial_paths if p and os.path.exists(p)), None)

    if matrix_path and os.path.exists(matrix_path):
        df_mat = pd.read_csv(matrix_path, index_col=0)
        X = df_mat.values.astype(np.float32)
        var_df = pd.DataFrame(index=df_mat.columns)
        var_df["gene_name"] = df_mat.columns
        obs_df = pd.DataFrame(index=df_mat.index)
        obs_df["barcode"] = df_mat.index
    else:
        raise FileNotFoundError(
            f"No expression matrix CSV found in {data_dir}. Expected filtered_feature_bc_matrix.csv or matrix.csv."
        )

    obsm = {}
    if spatial_path and os.path.exists(spatial_path):
        # 10x tissue positions format: barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
        try:
            df_pos = pd.read_csv(spatial_path, header=None)
            if df_pos.shape[1] >= 6:
                df_pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
            elif df_pos.shape[1] == 5:
                df_pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres"]
            df_pos = df_pos.set_index("barcode")
            
            # Align with expression matrix barcodes
            common_bc = [b for b in obs_df.index if b in df_pos.index]
            if common_bc:
                obs_df = obs_df.loc[common_bc]
                pos_aligned = df_pos.loc[common_bc]
                obs_df["in_tissue"] = pos_aligned["in_tissue"]
                obs_df["array_row"] = pos_aligned["array_row"]
                obs_df["array_col"] = pos_aligned["array_col"]
                coords = pos_aligned[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(np.float32)
                obsm["spatial"] = coords
                # Filter X
                keep_idx = [list(df_mat.index).index(b) for b in common_bc]
                X = X[keep_idx, :]
        except Exception as e:
            print(f"Warning: Could not parse spatial coordinates file: {e}")

    uns = {
        "source": "10x_Genomics_Visium_Directory",
        "data_dir": data_dir,
        "n_spots": len(obs_df),
        "n_genes": len(var_df)
    }

    return create_anndata(X=X, obs=obs_df, var=var_df, obsm=obsm, uns=uns)

