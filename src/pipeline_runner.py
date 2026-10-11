"""
Pipeline Runner Module.
Executes the end-to-end computational pipeline for spatial transcriptomics
analysis of biomaterial-host interfaces, generating comprehensive results and figures.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spatial_dataset import generate_biomaterial_visium_dataset
from src.preprocessing import calculate_qc_metrics, normalize_and_log, select_highly_variable_genes, compute_spatial_moran_i
from src.clustering import run_pca, run_umap, cluster_populations, annotate_cell_types
from src.spatial_distance import calculate_radial_distances, assign_distance_zones, compute_radial_gene_gradients, compute_cellular_recruitment_profiles
from src.pathway_scoring import score_pathways, compare_pathways_across_morphologies
from src.differential_expression import run_differential_expression, run_cross_condition_interface_dge
from src.microgel_design import calculate_biomaterial_performance_indices, simulate_microgel_parameter_space, synthesize_optimal_microgel_design
from src.visualizer import (
    plot_spatial_cluster_maps,
    plot_spatial_marker_grid,
    plot_radial_distance_comparison,
    plot_morphology_radar_comparison,
    plot_microgel_pareto_design_space
)


def run_pipeline(output_dir: str = "results", grid_size: int = 34, random_seed: int = 42):
    """
    Executes full pipeline across Solid, Microporous, and Microgel scaffold datasets.
    """
    print("=" * 75)
    print("  SPATIAL TRANSCRIPTOMICS & BIOMATERIAL-HOST INTERFACE ANALYSIS PIPELINE  ")
    print("=" * 75)
    
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs("dashboard", exist_ok=True)
    
    conditions = ["solid", "microporous", "microgel"]
    adatas = {}
    metrics_summary = {}
    radial_profiles = {}
    cellular_recruitment = {}
    
    for cond in conditions:
        print(f"\n[Step 1/6] Processing Condition: '{cond.upper()}' Scaffold...")
        
        # 1. Dataset Generation
        adata = generate_biomaterial_visium_dataset(
            condition=cond,
            grid_size=grid_size,
            spacing_um=100.0,
            implant_radius_um=1050.0,
            random_seed=random_seed
        )
        print(f"  -> Generated {adata.n_obs} spatial capture spots across {adata.n_vars} genes.")
        
        # 2. Preprocessing & Quality Control
        calculate_qc_metrics(adata)
        normalize_and_log(adata)
        hvgs = select_highly_variable_genes(adata, top_n=30)
        print(f"  -> QC complete. Mean library size: {adata.uns['qc_summary']['mean_library_size']:.1f} UMI. Selected {len(hvgs)} HVGs.")
        
        # 3. Dimensionality Reduction & Clustering
        run_pca(adata, n_comps=12)
        run_umap(adata)
        cluster_populations(adata, n_clusters=7)
        annotate_cell_types(adata)
        print(f"  -> Clustering complete. Identified immune & stromal populations: {list(adata.obs['cell_type_annotated'].unique())}")
        
        # 4. Spatial Distance & Zoning
        calculate_radial_distances(adata)
        assign_distance_zones(adata)
        
        # 5. Spatial Autocorrelation
        moran_acta2 = compute_spatial_moran_i(adata, "Acta2")
        moran_vegfa = compute_spatial_moran_i(adata, "Vegfa")
        moran_arg1 = compute_spatial_moran_i(adata, "Arg1")
        print(f"  -> Spatial Autocorrelation (Moran's I): Acta2={moran_acta2:.3f}, Vegfa={moran_vegfa:.3f}, Arg1={moran_arg1:.3f}")
        
        # 6. Pathway Scoring
        score_pathways(adata)
        
        # 7. Radial Gradient & Recruitment Profiling
        prof = compute_radial_gene_gradients(
            adata,
            genes=["Acta2", "Col1a1", "Vegfa", "Pecam1", "Arg1", "Mrc1", "Tnf", "Il1b", "Tgfb1"]
        )
        recruitment = compute_cellular_recruitment_profiles(adata)
        
        # 8. Biomaterial Indices
        indices = calculate_biomaterial_performance_indices(adata)
        indices["moran_i_Acta2"] = round(moran_acta2, 3)
        indices["moran_i_Vegfa"] = round(moran_vegfa, 3)
        indices["moran_i_Arg1"] = round(moran_arg1, 3)
        
        adatas[cond] = adata
        metrics_summary[cond] = indices
        radial_profiles[cond] = prof
        cellular_recruitment[cond] = recruitment
        
        # Plot individual spatial cluster map
        plot_spatial_cluster_maps(adata, save_path=f"{output_dir}/figures/{cond}_spatial_clusters.png")
        if cond in ["solid", "microgel"]:
            plot_spatial_marker_grid(
                adata,
                genes=["Acta2", "Vegfa", "Arg1", "Tnf"],
                save_path=f"{output_dir}/figures/{cond}_marker_grid.png"
            )
            
    print("\n" + "=" * 75)
    print("[Step 2/6] Comparative Biomaterial-Host Morphology Analysis")
    print("=" * 75)
    df_metrics = pd.DataFrame(metrics_summary).T
    print(df_metrics[["fibrosis_encapsulation_index", "vascular_to_fibrosis_ratio", "m2_to_m1_ratio", "max_infiltration_depth_um", "tissue_integration_score"]])
    
    # Differential Expression at Interface: Microgel MAP vs Solid
    print("\n[Step 3/6] Running Interface Differential Gene Expression (Microgel vs Solid)...")
    df_dge = run_cross_condition_interface_dge(adatas["solid"], adatas["microgel"])
    dge_path = f"{output_dir}/differential_expression_microgel_vs_solid.csv"
    df_dge.to_csv(dge_path, index=False)
    print(f"  -> Saved DGE results ({len(df_dge)} genes) to {dge_path}")
    print("  -> Top Upregulated in Microgel:")
    top_microgel = df_dge.head(5)[["gene", "log2fc_microgel_vs_solid", "adjusted_p_value", "category"]]
    print(top_microgel.to_string(index=False))
    print("  -> Top Upregulated in Solid (Fibrotic/Pro-inflammatory):")
    top_solid = df_dge.tail(5)[["gene", "log2fc_microgel_vs_solid", "adjusted_p_value", "category"]]
    print(top_solid.to_string(index=False))
    
    print("\n[Step 4/6] Microgel Rational Design Simulation & Multi-Objective Pareto Optimization...")
    df_sim = simulate_microgel_parameter_space()
    opt_specs = synthesize_optimal_microgel_design(target_tissue="dermal_wound_healing")
    
    print(f"  -> Optimal Microgel Particle Diameter: {opt_specs['recommended_microgel_diameter_um']:.1f} µm")
    print(f"  -> Resulting Interconnected Pore Size: {opt_specs['recommended_void_pore_size_um']:.1f} µm")
    print(f"  -> Optimal Hydrogel Shear Modulus (Stiffness): {opt_specs['recommended_shear_modulus_kpa']:.2f} kPa")
    print(f"  -> Predicted Bio-Integration Utility: {opt_specs['bio_integration_score']:.1f}/100")
    
    with open(f"{output_dir}/microgel_design_specs.json", "w") as f:
        json.dump(opt_specs, f, indent=2)
        
    with open(f"{output_dir}/metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print("\n[Step 5/6] Generating Publication Figures...")
    plot_radial_distance_comparison(
        radial_profiles,
        genes_to_plot=["Acta2", "Vegfa", "Arg1", "Tnf"],
        save_path=f"{output_dir}/figures/radial_distance_profiles.png"
    )
    plot_morphology_radar_comparison(
        metrics_summary,
        save_path=f"{output_dir}/figures/biomaterial_radar_comparison.png"
    )
    plot_microgel_pareto_design_space(
        df_sim,
        opt_specs,
        save_path=f"{output_dir}/figures/microgel_design_space.png"
    )
    print(f"  -> Figures successfully saved to {output_dir}/figures/")
    
    print("\n[Step 6/6] Exporting Data for Interactive Dashboard Portal...")
    # Export compact JSON for web dashboard
    dashboard_data = {
        "conditions": {},
        "metrics": metrics_summary,
        "dge_top": df_dge.to_dict(orient="records")[:30],
        "optimization": opt_specs,
        "simulation_sample": df_sim.iloc[::6].to_dict(orient="records")
    }
    
    for cond, adata in adatas.items():
        coords = adata.obsm["spatial"]
        sub_sample_step = 1  # keep all spots
        dashboard_data["conditions"][cond] = {
            "name": adata.uns["parameters"]["name"],
            "spots": [
                {
                    "x": round(float(coords[i, 0]), 1),
                    "y": round(float(coords[i, 1]), 1),
                    "dist": round(float(adata.obs["dist_to_interface_um"].iloc[i]), 1),
                    "zone": str(adata.obs["distance_zone"].iloc[i]),
                    "cell_type": str(adata.obs["cell_type_annotated"].iloc[i]),
                    "Acta2": round(float(adata.X[i, list(adata.var_names).index("Acta2")]), 2),
                    "Vegfa": round(float(adata.X[i, list(adata.var_names).index("Vegfa")]), 2),
                    "Arg1": round(float(adata.X[i, list(adata.var_names).index("Arg1")]), 2),
                    "Tnf": round(float(adata.X[i, list(adata.var_names).index("Tnf")]), 2),
                    "pathway_fibrosis": round(float(adata.obs["pathway_FBR_Fibrotic_Encapsulation"].iloc[i]), 2),
                    "pathway_m2": round(float(adata.obs["pathway_Pro_Regenerative_M2"].iloc[i]), 2),
                    "pathway_m1": round(float(adata.obs["pathway_Pro_Inflammatory_M1"].iloc[i]), 2),
                    "pathway_vasc": round(float(adata.obs["pathway_Neo_Vascularization"].iloc[i]), 2),
                }
                for i in range(0, adata.n_obs, sub_sample_step)
            ]
        }
        
    dashboard_json_path = "dashboard/data.json"
    with open(dashboard_json_path, "w") as f:
        json.dump(dashboard_data, f)
    print(f"  -> Exported dashboard payload to {dashboard_json_path}")
    
    print("\n" + "=" * 75)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!  ")
    print("=" * 75)
    return {
        "metrics": metrics_summary,
        "dge": df_dge,
        "optimal_design": opt_specs
    }


if __name__ == "__main__":
    run_pipeline()
