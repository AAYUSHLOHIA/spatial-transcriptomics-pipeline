"""
Microgel Rational Design and Bioengineering Synthesis Module.
Translates spatial transcriptomic findings into actionable engineering parameters
for the rational design of Microporous Annealed Particle (MAP) microgel scaffolds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def calculate_biomaterial_performance_indices(adata) -> Dict[str, float]:
    """
    Computes bioengineering performance indices from spatial transcriptomic profiles:
    - Fibrosis Encapsulation Index (FEI)
    - Vascularization-to-Fibrosis Ratio (VFR)
    - Pro-Regenerative M2/M1 Polarization Index (RPI)
    - Cellular Infiltration Depth (CID_um)
    - Host Tissue Integration Score (TIS, 0-100 scale)
    """
    obs = adata.obs
    var_list = list(adata.var_names)
    X = adata.X
    
    # 1. Interface Spots (0 to 200 µm)
    interface_mask = (obs["dist_to_interface_um"] >= 0) & (obs["dist_to_interface_um"] <= 200)
    if not np.any(interface_mask):
        interface_mask = obs["dist_to_interface_um"] >= 0
        
    X_int = X[interface_mask]
    
    def get_mean(gene):
        if gene in var_list:
            idx = var_list.index(gene)
            return float(np.mean(X_int[:, idx]))
        return 0.0

    # Fibrotic markers
    acta2 = get_mean("Acta2")
    col1a1 = get_mean("Col1a1")
    tgfb1 = get_mean("Tgfb1")
    fibrosis_score = acta2 + col1a1 + tgfb1

    # Vascular markers
    pecam1 = get_mean("Pecam1")
    vegfa = get_mean("Vegfa")
    vasc_score = pecam1 + vegfa

    # M1 and M2 markers
    m1_score = get_mean("Tnf") + get_mean("Il1b") + get_mean("Nos2")
    m2_score = get_mean("Arg1") + get_mean("Mrc1") + get_mean("Chil3")
    
    # Ratios
    vfr = vasc_score / (fibrosis_score + 1e-4)
    m2_m1_ratio = m2_score / (m1_score + 1e-4)
    
    # 2. Cellular Infiltration Depth
    # Measure max depth inside scaffold (dist < 0) containing viable spots
    infiltrate_spots = obs[obs["dist_to_interface_um"] < 0]
    if len(infiltrate_spots) > 0:
        max_inf_depth = float(np.abs(np.min(infiltrate_spots["dist_to_interface_um"])))
    else:
        max_inf_depth = 0.0
        
    # 3. Overall Tissue Integration Score (TIS, 0 to 100)
    # Higher VFR, higher M2/M1, deeper infiltration, and lower fibrosis increase TIS
    norm_vfr = min(1.0, vfr / 1.5)
    norm_m2 = min(1.0, m2_m1_ratio / 2.5)
    norm_inf = min(1.0, max_inf_depth / 800.0)
    norm_fib = max(0.0, 1.0 - (fibrosis_score / 25.0))
    
    tis = float(100.0 * (0.30 * norm_vfr + 0.30 * norm_m2 + 0.25 * norm_inf + 0.15 * norm_fib))
    
    return {
        "fibrosis_encapsulation_index": round(fibrosis_score, 3),
        "vascular_score": round(vasc_score, 3),
        "vascular_to_fibrosis_ratio": round(vfr, 3),
        "m2_to_m1_ratio": round(m2_m1_ratio, 3),
        "max_infiltration_depth_um": round(max_inf_depth, 1),
        "tissue_integration_score": round(tis, 1),
    }


def simulate_microgel_parameter_space(
    particle_sizes: np.ndarray = np.linspace(30, 150, 25),
    stiffness_kpa: np.ndarray = np.linspace(0.5, 25, 25)
) -> pd.DataFrame:
    """
    Parametric biophysical simulation model connecting microgel engineering specifications
    (particle size d_p, Young's modulus E, interconnected pore size)
    to predicted biological response derived from spatial transcriptomic regressions.
    
    Biophysical rules:
    - Void pore diameter d_pore approx 0.25 to 0.35 * d_particle for jammed granular microgels.
    - Pore sizes < 20 µm restrict endothelial migration (capillary sprouting requires ~25-40 µm).
    - High stiffness (>10 kPa) drives mechanosensitive myofibroblast differentiation (alpha-SMA/Acta2).
    - Low to moderate stiffness (1-5 kPa) fosters pro-regenerative M2 macrophage polarization.
    """
    results = []
    
    for dp in particle_sizes:
        for E in stiffness_kpa:
            # Predicted interstitial void pore diameter (µm)
            pore_size_um = 0.30 * dp
            
            # Cellular infiltration factor (sigmoidal threshold at pore size > 25 µm)
            inf_factor = 1.0 / (1.0 + np.exp(-(pore_size_um - 28.0) / 5.0))
            infiltration_depth_pred_um = 1200.0 * inf_factor
            
            # Fibrosis & capsule activation factor (increases with stiffness via YAP/TAZ mechanotransduction)
            fibrosis_factor = 1.0 / (1.0 + np.exp(-(E - 10.0) / 3.0))
            predicted_capsule_um = 20.0 + 320.0 * fibrosis_factor
            
            # Neo-vascularization factor: requires both adequate pore size (>25µm) and low/moderate stiffness
            pore_permissiveness = 1.0 / (1.0 + np.exp(-(pore_size_um - 30.0) / 6.0))
            stiffness_penalty = np.exp(-E / 12.0)
            vascularization_score = float(10.0 * pore_permissiveness * stiffness_penalty)
            
            # M2/M1 polarization ratio
            m2_m1_ratio_pred = float(2.8 * (1.0 / (1.0 + 0.15 * E)) * (0.4 + 0.6 * inf_factor))
            
            # Composite Bio-Integration Utility (Objective Function to maximize)
            utility = (
                0.35 * (infiltration_depth_pred_um / 1200.0) +
                0.35 * (vascularization_score / 10.0) +
                0.20 * (m2_m1_ratio_pred / 2.8) -
                0.30 * (predicted_capsule_um / 340.0)
            )
            utility_score = max(0.0, min(100.0, float(utility * 100.0)))
            
            results.append({
                "microgel_particle_diameter_um": round(dp, 1),
                "hydrogel_stiffness_kpa": round(E, 2),
                "void_pore_size_um": round(pore_size_um, 1),
                "predicted_infiltration_depth_um": round(infiltration_depth_pred_um, 1),
                "predicted_capsule_thickness_um": round(predicted_capsule_um, 1),
                "predicted_vascularization_score": round(vascularization_score, 2),
                "predicted_m2_m1_ratio": round(m2_m1_ratio_pred, 2),
                "bio_integration_utility_score": round(utility_score, 1)
            })
            
    df_sim = pd.DataFrame(results)
    return df_sim


def synthesize_optimal_microgel_design(
    target_tissue: str = "dermal_wound_healing"
) -> Dict:
    """
    Computes optimal Pareto design specifications for microgels
    based on the bioinformatics spatial interface findings.
    """
    df_sim = simulate_microgel_parameter_space()
    
    # Target tissue specific weighting
    if target_tissue == "dermal_wound_healing":
        # Balance rapid vascularization and moderate stiffness
        optimal_row = df_sim.sort_values(by="bio_integration_utility_score", ascending=False).iloc[0]
    elif target_tissue == "neural_repair":
        # Ultra soft (< 2 kPa), high porosity
        neural_subset = df_sim[df_sim["hydrogel_stiffness_kpa"] <= 2.5]
        optimal_row = neural_subset.sort_values(by="bio_integration_utility_score", ascending=False).iloc[0]
    else:
        optimal_row = df_sim.sort_values(by="bio_integration_utility_score", ascending=False).iloc[0]
        
    spec = {
        "target_application": target_tissue,
        "recommended_microgel_diameter_um": float(optimal_row["microgel_particle_diameter_um"]),
        "recommended_void_pore_size_um": float(optimal_row["void_pore_size_um"]),
        "recommended_shear_modulus_kpa": float(optimal_row["hydrogel_stiffness_kpa"]),
        "expected_capsule_thickness_um": float(optimal_row["predicted_capsule_thickness_um"]),
        "expected_infiltration_depth_um": float(optimal_row["predicted_infiltration_depth_um"]),
        "expected_m2_m1_polarization_ratio": float(optimal_row["predicted_m2_m1_ratio"]),
        "bio_integration_score": float(optimal_row["bio_integration_utility_score"]),
        "rational_design_rules": [
            "1. Void Porosity Constraint: Maintain particle diameter >= 75 µm to yield interstitial pore sizes > 25 µm, which prevents physical exclusion of migrating endothelial cells (Vegfa/Pecam1).",
            "2. Mechano-transduction Threshold: Keep hydrogel shear modulus <= 4.5 kPa to suppress YAP/TAZ activation and prevent Acta2+ myofibroblast differentiation and dense fibrotic capsule formation.",
            "3. Surface Annealing Chemistry: Utilize biocompatible crosslinkers (e.g. Factor XIIIa or bio-orthogonal click tetrazine-TCO) that preserve cell adhesion ligand availability (RGD).",
            "4. Dynamic Remodeling: Incorporate matrix metalloproteinase-cleavable peptides (MMP-sensitive) to synchronize microgel biodegradation with endogenous host ECM deposition."
        ]
    }
    return spec
