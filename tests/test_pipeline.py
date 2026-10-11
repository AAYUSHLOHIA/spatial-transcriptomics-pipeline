"""
Unit Tests for Spatial Transcriptomics & Biomaterial-Host Interface Pipeline.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spatial_dataset import generate_biomaterial_visium_dataset
from src.preprocessing import calculate_qc_metrics, normalize_and_log, select_highly_variable_genes, compute_spatial_moran_i
from src.clustering import run_pca, run_umap, cluster_populations, annotate_cell_types
from src.spatial_distance import calculate_radial_distances, assign_distance_zones, compute_radial_gene_gradients
from src.pathway_scoring import score_pathways
from src.differential_expression import run_differential_expression
from src.microgel_design import calculate_biomaterial_performance_indices, simulate_microgel_parameter_space, synthesize_optimal_microgel_design


class TestSpatialBiomaterialPipeline(unittest.TestCase):

    def setUp(self):
        # Generate appropriately proportioned dataset for fast testing
        self.adata_solid = generate_biomaterial_visium_dataset(
            "solid", grid_size=24, implant_radius_um=700.0, random_seed=42
        )
        self.adata_microgel = generate_biomaterial_visium_dataset(
            "microgel", grid_size=24, implant_radius_um=700.0, random_seed=42
        )

    def test_dataset_generation(self):
        self.assertGreater(self.adata_solid.n_obs, 100)
        self.assertGreater(self.adata_microgel.n_obs, 150)
        self.assertEqual(self.adata_solid.n_vars, 53)
        self.assertIn("spatial", self.adata_solid.obsm)
        self.assertEqual(self.adata_solid.obsm["spatial"].shape[1], 2)
        # Microgel should have infiltrated spots inside implant (dist < 0)
        has_negative_dist = np.any(self.adata_microgel.obs["dist_to_interface_um"] < 0)
        self.assertTrue(has_negative_dist)

    def test_preprocessing_and_normalization(self):
        adata = self.adata_solid.copy()
        raw_max = np.max(adata.X)
        calculate_qc_metrics(adata)
        normalize_and_log(adata)
        log_max = np.max(adata.X)
        self.assertLess(log_max, 15.0)
        hvgs = select_highly_variable_genes(adata, top_n=20)
        self.assertEqual(len(hvgs), 20)

    def test_clustering_and_cell_typing(self):
        adata = self.adata_solid.copy()
        normalize_and_log(adata)
        run_pca(adata, n_comps=8)
        self.assertIn("X_pca", adata.obsm)
        run_umap(adata)
        self.assertIn("X_umap", adata.obsm)
        cluster_populations(adata, n_clusters=5)
        self.assertIn("leiden_cluster", adata.obs)
        scores = annotate_cell_types(adata)
        self.assertIn("cell_type_annotated", adata.obs)
        self.assertFalse(adata.obs["cell_type_annotated"].isnull().any())

    def test_spatial_distances_and_zones(self):
        adata = self.adata_solid.copy()
        calculate_radial_distances(adata)
        assign_distance_zones(adata)
        self.assertIn("distance_zone", adata.obs)
        profiles = compute_radial_gene_gradients(adata, genes=["Acta2", "Vegfa"])
        self.assertGreater(len(profiles), 5)
        self.assertIn("Acta2_mean", profiles.columns)

    def test_pathway_scoring(self):
        adata = self.adata_microgel.copy()
        normalize_and_log(adata)
        score_pathways(adata)
        self.assertIn("pathway_FBR_Fibrotic_Encapsulation", adata.obs)
        self.assertIn("pathway_Neo_Vascularization", adata.obs)
        self.assertIn("pathway_Pro_Regenerative_M2", adata.obs)

    def test_microgel_rational_design_simulation(self):
        df_sim = simulate_microgel_parameter_space()
        self.assertGreater(len(df_sim), 50)
        self.assertIn("bio_integration_utility_score", df_sim.columns)
        spec = synthesize_optimal_microgel_design()
        self.assertGreater(spec["recommended_microgel_diameter_um"], 40.0)
        self.assertLess(spec["recommended_shear_modulus_kpa"], 10.0)
        self.assertIn("rational_design_rules", spec)
        self.assertEqual(len(spec["rational_design_rules"]), 4)


if __name__ == "__main__":
    unittest.main()
