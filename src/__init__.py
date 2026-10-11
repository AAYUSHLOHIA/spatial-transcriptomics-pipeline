"""
Spatial Transcriptomics & Biomaterial-Host Interface Analysis Package
"""

from .spatial_dataset import generate_biomaterial_visium_dataset, load_10x_visium_directory, SpatialAnnData
from .preprocessing import calculate_qc_metrics, normalize_and_log, select_highly_variable_genes, compute_spatial_moran_i
from .clustering import run_pca, run_umap, cluster_populations, annotate_cell_types
from .spatial_distance import calculate_radial_distances, assign_distance_zones, compute_radial_gene_gradients, compute_cellular_recruitment_profiles
from .pathway_scoring import score_pathways, compare_pathways_across_morphologies
from .differential_expression import run_differential_expression, run_cross_condition_interface_dge
from .microgel_design import calculate_biomaterial_performance_indices, simulate_microgel_parameter_space, synthesize_optimal_microgel_design
from .visualizer import (
    plot_spatial_cluster_maps,
    plot_spatial_marker_grid,
    plot_radial_distance_comparison,
    plot_morphology_radar_comparison,
    plot_microgel_pareto_design_space
)

__version__ = "1.0.0"
