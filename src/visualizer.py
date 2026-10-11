"""
Visualizer Module for Spatial Biomaterial-Host Interface Analysis.
Generates publication-ready figures for spatial spot distributions,
radial distance gradients, comparative immune pathways, and microgel design space.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set clean scientific plotting style
plt.rcParams.update({
    "font.sans-serif": "Helvetica, Arial, DejaVu Sans",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 14
})


def plot_spatial_cluster_maps(adata, save_path: str = "results/figures/spatial_clusters.png"):
    """
    Plots spatial distribution of Visium spots colored by annotated cell types.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    coords = adata.obsm["spatial"]
    cell_types = adata.obs["cell_type_annotated"].values
    unique_types = np.unique(cell_types)
    
    # Palette
    cmap = cm.get_cmap("tab10", len(unique_types))
    color_map = {ct: cmap(i) for i, ct in enumerate(unique_types)}
    
    fig, ax = plt.subplots(figsize=(8, 7))
    for ct in unique_types:
        mask = cell_types == ct
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=22,
            color=color_map[ct],
            label=ct,
            alpha=0.9,
            edgecolors="none"
        )
        
    # Draw implant interface circle if center and radius known
    if "implant_radius_um" in adata.uns:
        r = adata.uns["implant_radius_um"]
        center = (np.mean(coords[:, 0]), np.mean(coords[:, 1]))
        circle = plt.Circle(center, r, color="black", fill=False, linestyle="--", linewidth=1.5, label="Implant Boundary")
        ax.add_patch(circle)
        
    ax.set_aspect("equal")
    ax.set_title(f"Spatial Cell Type Organization ({adata.uns.get('condition', 'Biomaterial').capitalize()} Scaffold)", fontweight="bold")
    ax.set_xlabel("Spatial X Coordinate (µm)")
    ax.set_ylabel("Spatial Y Coordinate (µm)")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_spatial_marker_grid(adata, genes: list, save_path: str = "results/figures/spatial_gene_markers.png"):
    """
    Plots a multi-panel spatial expression grid for key biomarker genes.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    coords = adata.obsm["spatial"]
    var_list = list(adata.var_names)
    
    n_genes = len(genes)
    cols = 2
    rows = (n_genes + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.5 * rows))
    axes = np.array(axes).flatten()
    
    for i, g in enumerate(genes):
        ax = axes[i]
        if g in var_list:
            g_idx = var_list.index(g)
            expr = adata.X[:, g_idx]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=expr, cmap="magma", s=18, alpha=0.9)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log1p UMI")
            ax.set_title(f"Expression: {g}", fontweight="bold")
        else:
            ax.text(0.5, 0.5, f"Gene {g} not found", ha="center")
            
        ax.set_aspect("equal")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        
    for j in range(n_genes, len(axes)):
        axes[j].axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_radial_distance_comparison(
    profiles_dict: dict,
    genes_to_plot: list = ["Acta2", "Vegfa", "Arg1", "Tnf"],
    save_path: str = "results/figures/radial_distance_profiles.png"
):
    """
    Plots radial distance curves f(r) comparing Solid vs Microporous vs Microgel scaffolds.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    n_genes = len(genes_to_plot)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()
    
    colors = {"solid": "#D9534F", "microporous": "#F0AD4E", "microgel": "#5CB85C"}
    styles = {"solid": "-", "microporous": "--", "microgel": "-."}
    
    for i, g in enumerate(genes_to_plot):
        ax = axes[i]
        for cond, df_prof in profiles_dict.items():
            mean_col = f"{g}_mean"
            sem_col = f"{g}_sem"
            if mean_col in df_prof.columns:
                valid = ~df_prof[mean_col].isna()
                x = df_prof.loc[valid, "bin_center_um"]
                y = df_prof.loc[valid, mean_col]
                yerr = df_prof.loc[valid, sem_col]
                
                c = colors.get(cond, "#333333")
                ax.plot(x, y, label=cond.capitalize(), color=c, linestyle=styles.get(cond, "-"), linewidth=2.0)
                ax.fill_between(x, y - yerr, y + yerr, color=c, alpha=0.15)
                
        # Draw interface boundary line at 0 µm
        ax.axvline(0, color="gray", linestyle=":", linewidth=1.5, label="Boundary" if i == 0 else "")
        ax.set_title(f"Spatial Gradient: {g}", fontweight="bold")
        ax.set_ylabel("Mean Log-Normalized Expression")
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(frameon=True)
            
    axes[2].set_xlabel("Radial Distance from Interface (µm)\n[ <0: Inside Implant  |  >0: In Host Tissue ]")
    axes[3].set_xlabel("Radial Distance from Interface (µm)\n[ <0: Inside Implant  |  >0: In Host Tissue ]")
    
    plt.suptitle("Radial Gene Expression Gradients Across Biomaterial Architectures", fontsize=14, y=0.99, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_morphology_radar_comparison(
    metrics_dict: dict,
    save_path: str = "results/figures/biomaterial_radar_comparison.png"
):
    """
    Generates a multi-axis comparison radar chart of host interface bio-integration metrics.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    categories = [
        "Vascular/Fibrosis Ratio",
        "M2/M1 Polarization",
        "Infiltration Depth",
        "Capsule Suppression",
        "Tissue Integration"
    ]
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    colors = {"solid": "#E74C3C", "microporous": "#F39C12", "microgel": "#27AE60"}
    
    for cond, m in metrics_dict.items():
        vfr_norm = min(1.0, m["vascular_to_fibrosis_ratio"] / 1.5)
        m2_norm = min(1.0, m["m2_to_m1_ratio"] / 2.5)
        inf_norm = min(1.0, m["max_infiltration_depth_um"] / 1000.0)
        capsule_sup = max(0.0, 1.0 - (m["fibrosis_encapsulation_index"] / 25.0))
        tis_norm = m["tissue_integration_score"] / 100.0
        
        values = [vfr_norm, m2_norm, inf_norm, capsule_sup, tis_norm]
        values += values[:1]
        
        c = colors.get(cond, "#34495E")
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=f"{cond.capitalize()} Scaffold", color=c)
        ax.fill(angles, values, color=c, alpha=0.2)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], color="grey", size=7)
    plt.ylim(0, 1.05)
    plt.title("Biomaterial-Host Interface Performance Metrics", y=1.08, fontweight="bold")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_microgel_pareto_design_space(
    df_sim: pd.DataFrame,
    opt_spec: dict,
    save_path: str = "results/figures/microgel_design_space.png"
):
    """
    Plots heatmap of predicted bio-integration utility across particle size and stiffness,
    highlighting the optimal rational microgel design point.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    pivot = df_sim.pivot(
        index="hydrogel_stiffness_kpa",
        columns="microgel_particle_diameter_um",
        values="bio_integration_utility_score"
    )
    
    fig, ax = plt.subplots(figsize=(9, 6.5))
    cax = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max()
        ]
    )
    
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Bio-Integration Score (0-100)", rotation=270, labelpad=15)
    
    # Highlight optimal design point
    opt_dp = opt_spec["recommended_microgel_diameter_um"]
    opt_E = opt_spec["recommended_shear_modulus_kpa"]
    ax.scatter(opt_dp, opt_E, color="#E74C3C", s=140, edgecolors="white", linewidths=2.0, marker="*", label=f"Optimal Design Point\n(d={opt_dp}µm, E={opt_E}kPa)", zorder=5)
    
    ax.set_title("Rational Design Space: Microgel Diameter vs Hydrogel Stiffness", fontweight="bold")
    ax.set_xlabel("Microgel Particle Diameter (µm)")
    ax.set_ylabel("Hydrogel Shear Modulus (kPa)")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path
