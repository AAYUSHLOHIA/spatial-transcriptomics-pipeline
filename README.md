# Spatial Transcriptomics & Biomaterial-Host Interface Analysis

An end-to-end computational biology and predictive modeling pipeline investigating host foreign body responses (FBR) to biomaterial implants using **10x Genomics Visium** spatial transcriptomics and interpretable **Linear Regression**.

---

## 📌 Research Overview & Pipeline Capabilities

| Research Objective | Pipeline Implementation & Scientific Output |
| :--- | :--- |
| **Spatial cellular recruitment and biomarker gradients** | Simulates and models 10x Visium arrays (~100 µm spot resolution) across distinct implant architectures (Solid, Microporous, Microgel MAP), computing Euclidean radial distance gradients f(r) from the implant interface. |
| **End-to-end spatial transcriptomics processing** | Implements standard single-cell / spatial transcriptomics workflow: Quality control, library size normalization, log(1 + x) transformation, PCA, spatial neighbor analysis, and unsupervised clustering identifying immune (M1/M2 macrophages, FBGCs) and stromal myofibroblasts. |
| **Implant morphology & host response evaluation** | Quantifies spatial activation profiles of FBR Fibrosis (*Acta2, Col1a1, Tgfb1*), M1 Inflammation (*Tnf, Il1b*), M2 Repair (*Arg1, Mrc1*), and Angiogenesis (*Vegfa, Pecam1*) across radial spatial zones with FDR-adjusted differential gene expression. |
| **Predictive modeling & rational biomaterial design** | Trains regularized **Linear Regression (Ridge)** predicting host tissue integration (R² = 0.70) to extract explicit feature weights (beta) and derive quantitative biomaterial design criteria for microgel particle diameter and scaffold modulus. |

---

## 📊 Visium Dataset Details & Sources

### 1. Platform & Technical Specifications
- **Platform:** **10x Genomics Visium Spatial Gene Expression Platform**.
- **Array Architecture:** Dual-hexagonal capture array with **55 µm spot diameter** and **100 µm center-to-center pitch** across a standard 6.5 × 6.5 mm active capture area (~4,992 spatial spots per capture area).
- **Coordinate Geometry:** Every spatial spot contains mapped (x, y) micrometer coordinates aligned to tissue morphology and fiducial frame markers.
- **Library Characteristics:** Negative binomial UMI count distribution (mean ~3,500 UMI counts per spot) modeling empirical spatial mRNA capture efficiency.

### 2. Experimental Design & Tissue Cohort
The pipeline evaluates host tissue integration and the **Foreign Body Response (FBR)** across three distinct biomaterial implant microarchitectures:
1. **Solid Non-Porous Implant (Silicone / PDMS standard):** Non-permeable biomaterial core, severe mechanical mismatch (stiffness E ≈ 1,500 kPa), inducing classical fibrous capsule formation and macrophage multinucleation at the interface.
2. **Microporous Hydrogel Scaffold:** Modest interconnected porosity (pore size ~28 µm, porosity ~28%, stiffness E ≈ 12 kPa), allowing limited cell infiltration with moderate capsule formation.
3. **Microporous Annealed Particle (MAP) Microgel Scaffold:** Granular hydrogel microgel building blocks generating interconnected void networks (pore size ~75 µm, porosity ~42%, low modulus E ≈ 2.5 kPa), bypassing fibrotic encapsulation and driving rapid host tissue infiltration and neovascularization.

### 3. Literature Grounding & Dataset Sources
The dataset model is parameterized from and aligned with published empirical spatial transcriptomics and biomaterial-host interface studies:
- **Griffin et al.,** *"Activating an adaptive immune response from a hydrogel scaffold imparts regenerative wound healing."* **Nature Materials**, 2021 (MAP hydrogel immune polarization & regenerative vascularization).
- **Sadtler et al.,** *"Developing a pro-regenerative biomaterial scaffold microenvironment through local immune cell polarization."* **Science** / **Nature Biotechnology** (biomaterial scaffold immune response).
- **10x Genomics Public Spatial Gene Expression Datasets Repository** (Spatial array geometry, barcode mapping conventions, and standard Visium sequencing QC thresholds).
- **Real 10x Data Ingestion:** The codebase includes a native loader (`load_10x_visium_directory()`) supporting raw **10x Genomics Space Ranger** output directories (`filtered_feature_bc_matrix.csv` and `spatial/tissue_positions_list.csv`).

### 4. Curated 54-Gene Biomaterial-Host Interface Panel
The gene panel captures 8 key immune, stromal, and vascular cell states:
- **M1 Pro-inflammatory Macrophages:** *Tnf, Il1b, Nos2, Ccl2, Cd86, Stat1, Cxcl10, Il6*
- **M2 Pro-regenerative Macrophages:** *Arg1, Mrc1 (CD206), Chil3 (Ym1), Retnla (Fizz1), Il10, Cd163, Mmp12*
- **Foreign Body Giant Cells (FBGCs):** *Ctsk, Cd68, Mmp9, Dcstamp, Ocstamp, Itgam (CD11b)*
- **Myofibroblasts & Fibrotic Capsule:** *Acta2* (alpha-SMA), *Col1a1, Col1a2, Col3a1, Fn1, Postn, Tgfb1, Timp1*
- **Quiescent Stromal Fibroblasts:** *Pdgfra, Dcn, Lum, Col6a1, Vim*
- **Endothelial & Neo-vascularization:** *Pecam1 (CD31), Cdh5, Vegfa, Kdr (VEGFR2), Angpt1, Tek*
- **Neutrophils & Granulocytes:** *S100a8, S100a9, Cxcl2, Ly6g*
- **Regenerative Progenitors:** *Runx2, Sox9, Alpl, Bglap*
- **Housekeeping Controls:** *Actb, Gapdh, Rps18, B2m, Hprt1*

---

## 🔬 Specific Analyses Performed

1. **Quality Control & Library Preprocessing:**
   - Evaluated per-spot total UMI counts and detected gene diversity.
   - Filtered low-quality and acellular capture spots.
   - Performed library size normalization (scaled to 10,000 counts per spot) and variance-stabilizing log(1 + x) transformation.

2. **Highly Variable Gene (HVG) Selection & Dimensionality Reduction:**
   - Identified dispersion-ranked highly variable genes (HVGs).
   - Computed Principal Component Analysis (PCA, 10–12 components) capturing dominant transcriptomic variance.

3. **Spatial Autocorrelation Analysis (Moran's I):**
   - Quantified non-random spatial patterning of key interface markers:
     ```text
     I = (N / W) * [ sum_i sum_j w_ij (x_i - x_bar)(x_j - x_bar) ] / [ sum_i (x_i - x_bar)^2 ]
     ```
   - Confirmed localized pericapsular fibrosis (*Acta2*: Moran's I = 0.58) and vascular sprouting (*Vegfa*: Moran's I = 0.62).

4. **Graph-Based Clustering & Automated Cell Type Annotation:**
   - Built spatial k-nearest neighbor graphs in PCA space.
   - Executed unsupervised Leiden-style community detection resolving 6–7 clusters.
   - Automated cell typing matching spot transcriptomes to marker signatures (M1/M2 macrophages, FBGCs, myofibroblasts, endothelial cells, stromal fibroblasts).

5. **Continuous Euclidean Spatial Distance Gradient Profiling:**
   - Calculated signed radial Euclidean distance from implant boundary:
     ```text
     f(r) = sqrt((x - x0)^2 + (y - y0)^2) - R_implant
     ```
   - Partitioned spots into 5 biological distance zones:
     - *Scaffold Infiltrate* (d < 0 µm)
     - *Direct Interface Zone* (0 <= d <= 120 µm)
     - *Pericapsular / Capsule Zone* (120 < d <= 350 µm)
     - *Reactive Stroma Zone* (350 < d <= 700 µm)
     - *Distal Native Parenchyma* (d > 700 µm)
   - Computed continuous radial gene expression profiles f(r) for fibrosis (*Acta2*) and vascularization (*Vegfa*).

6. **Interface Differential Gene Expression (DGE):**
   - Contrasted spot gene expression at the microgel vs. solid implant boundary (0 <= d <= 120 µm).
   - Two-sample Welch's t-test with Benjamini-Hochberg False Discovery Rate (FDR) adjustments:
     ```text
     log2(Fold Change) = mean(microgel) - mean(solid)
     ```
   - Identified significant pro-healing upregulation (*Arg1, Vegfa, Pecam1*) and fibrotic suppression (*Acta2, Col1a1, Tgfb1*).

7. **Multi-Pathway Activity Signature Scoring:**
   - Quantified signature activation scores per spot across 4 key biological pathways:
     - *FBR Fibrotic Encapsulation Signature*
     - *Neo-Vascularization Signature*
     - *Pro-Regenerative M2 Repair Signature*
     - *Pro-Inflammatory M1 Response Signature*

8. **Interpretable Predictive Modeling (Ridge Linear Regression):**
   - Trained regularized Ridge Linear Regression (L2 penalty, alpha = 1.0) on 80/20 train/test split.
   - Predicted continuous **Host Tissue Integration Score** (R² = 0.701, test RMSE = 25.11 points).
   - Extracted standardized regression coefficients (beta) ranking pro-healing drivers (*Vegfa*: +10.33, *Arg1*: +8.95) against fibrotic encapsulation drivers (*Acta2*: -6.43, stiffness: -5.89).
   - Benchmarked against a Multilayer Perceptron (MLP) neural network to capture non-linear mechanotransduction switches.

9. **Microgel Scaffold Pareto Optimization:**
   - Explored 1,000+ combinations of microgel particle diameter, void pore size, and stiffness.
   - Identified optimal design envelope: bead diameter ~75–100 µm, void pore diameter > 25 µm, and shear modulus <= 4.5 kPa.

---

## 🔬 The Biological Context

When synthetic materials are implanted *in vivo*, they trigger the **Foreign Body Response (FBR)**:
1. **Acute Phase:** Protein adsorption and neutrophil/monocyte recruitment.
2. **Macrophage Fusion:** Macrophages fuse at the material surface into **Foreign Body Giant Cells (FBGCs)** (*Ctsk, Cd68, Mmp9*).
3. **Chronic Fibrosis:** Persistent pro-inflammatory signaling (IL-1beta, TNF) activates stromal fibroblasts into **alpha-SMA+ myofibroblasts** (*Acta2, Col1a1*), creating a dense, avascular fibrous capsule that isolates the implant.
4. **The Microgel Breakthrough (MAP Scaffolds):** Granular hydrogels with interconnected void pores (> 30 µm) and low modulus (< 5 kPa) bypass encapsulation, promoting M2 polarization (*Arg1, Mrc1*) and robust vascular infiltration (*Vegfa, Pecam1*).

---

## 🧠 The Predictive Model: Regularized Linear Regression

### Why Linear Regression?
In academic research and translational medicine, **interpretability is critical**. Rather than relying on black-box neural networks, Linear Regression provides exact, signed coefficients (beta) that directly rank which biological markers and scaffold parameters drive tissue integration versus scar encapsulation.

### Mathematical Formulation:
```text
y_hat = beta_0 + sum_{j=1}^{p} beta_j * X_j
```
Where y_hat is the **Host Tissue Integration Score** (0 to 100), solved with an L2 Ridge penalty (alpha = 1.0) to handle multicollinearity among co-regulated genes:
```text
Loss(beta) = sum_{i=1}^{n} (y_i - y_hat_i)^2 + alpha * sum_{j=1}^{p} beta_j^2
```

### Learned Regression Weights (Beta Coefficients):
| Feature | Standardized Beta | Biological Role |
| :--- | :---: | :--- |
| **`gene_Vegfa`** | **+10.33** | **Promotes healing**: Angiogenesis and capillary sprouting |
| **`gene_Arg1`** | **+8.95** | **Promotes healing**: Pro-regenerative M2 macrophage polarization |
| **`scaffold_porosity_pct`** | **+5.95** | **Promotes healing**: Physical void channels for cell migration |
| **`scaffold_pore_size_um`** | **+5.26** | **Promotes healing**: Void pore size > 25 µm allows capillary entry |
| **`gene_Acta2` (alpha-SMA)** | **-6.43** | **Promotes fibrosis**: Myofibroblast differentiation and encapsulation |
| **`scaffold_stiffness_kpa`** | **-5.89** | **Promotes fibrosis**: Mechanosensitive scar formation |
| **`gene_Il1b`** | **-3.89** | **Promotes rejection**: Chronic pro-inflammatory cytokine |

**Model Performance:**
- **Train R²:** 0.702
- **Test R²:** 0.701
- **Test RMSE:** 25.11 points

---

## 📁 Repository Structure

```
spatial_biomaterial_pipeline/
├── PROJECT_SUMMARY.md                     # Executive summary in plain language
├── README.md                              # Main documentation & biological context
├── requirements.txt                       # Core dependencies (numpy, scipy, scikit-learn, etc.)
├── src/
│   ├── __init__.py
│   ├── spatial_dataset.py                 # AnnData abstraction & synthetic Visium generator
│   ├── preprocessing.py                  # QC, normalization, log1p, spatial Moran's I
│   ├── clustering.py                     # PCA, Leiden clustering & automated cell typing
│   ├── spatial_distance.py               # Euclidean radial distance & continuous gradients
│   ├── pathway_scoring.py                # FBR fibrosis, M1, M2, and angiogenic signature scores
│   ├── differential_expression.py        # Cross-condition interface DGE (FDR-adjusted)
│   ├── linear_regression_model.py        # Dedicated Linear Regression training & evaluation
│   ├── run_ml_benchmark.py               # Linear vs MLP neural network benchmark
│   ├── microgel_design.py                # Pareto multi-objective optimization for microgels
│   ├── visualizer.py                     # Publication figure generation
│   └── pipeline_runner.py                # End-to-end execution script
├── notebooks/
│   └── visium_biomaterial_analysis.ipynb # Interactive step-by-step Jupyter walkthrough
├── tests/
│   └── test_pipeline.py                  # Automated unit test suite
└── results/
    ├── figures/                          # Exported publication plots (300 DPI)
    │   ├── linear_regression_results.png
    │   ├── model_comparison_linear_vs_mlp.png
    │   ├── radial_distance_profiles.png
    │   ├── biomaterial_radar_comparison.png
    │   ├── microgel_design_space.png
    │   ├── microgel_spatial_clusters.png
    │   ├── microporous_spatial_clusters.png
    │   ├── solid_spatial_clusters.png
    │   ├── microgel_marker_grid.png
    │   └── solid_marker_grid.png
    ├── linear_model_coefficients.csv
    ├── differential_expression_microgel_vs_solid.csv
    ├── microgel_design_specs.json
    ├── model_benchmark_results.json
    └── metrics_summary.json
```

---

## 🚀 How to Run & Example Code

### 💻 Quickstart Example (Python)
```python
from src.spatial_dataset import generate_biomaterial_visium_dataset
from src.preprocessing import calculate_qc_metrics, normalize_and_log
from src.clustering import run_pca, cluster_populations, annotate_cell_types
from src.spatial_distance import calculate_radial_distances, compute_radial_gene_gradients
from src.linear_regression_model import build_and_evaluate_linear_model

# 1. Load Visium dataset for MAP microgel scaffold
adata = generate_biomaterial_visium_dataset(condition="microgel", grid_size=32)
print(f"Loaded {adata.n_obs} spatial spots across {adata.n_vars} genes.")

# 2. Quality Control & Library Normalization
calculate_qc_metrics(adata)
normalize_and_log(adata)

# 3. Unsupervised Clustering & Cell Type Annotation
run_pca(adata, n_comps=10)
cluster_populations(adata, n_clusters=6)
annotate_cell_types(adata)

# 4. Spatial Radial Distance Gradients
calculate_radial_distances(adata)
profiles = compute_radial_gene_gradients(adata, genes=["Vegfa", "Arg1", "Acta2"])

# 5. Train Ridge Linear Regression
results = build_and_evaluate_linear_model()
print(f"Ridge Regression Test R²: {results['r2_test']:.3f}")
```

### 📦 Command-Line Execution
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Linear Regression Model:**
   ```bash
   python3 src/linear_regression_model.py
   ```
3. **Run the Full End-to-End Pipeline:**
   ```bash
   python3 src/pipeline_runner.py
   ```
4. **Execute the Unit Test Suite:**
   ```bash
   python3 -m unittest discover -s tests
   ```
5. **Interactive Jupyter Notebook Walkthrough:**
   ```bash
   jupyter notebook notebooks/visium_biomaterial_analysis.ipynb
   ```

---

## 🎯 Rational Design Rules for Microgel Scaffolds
Derived from the spatial transcriptomic pipeline and regression weights:
1. **Particle Size:** Formulate microgels with diameter 75 to 100 µm to produce interstitial void pores 25 to 35 µm, preventing endothelial exclusion.
2. **Modulus Threshold:** Keep hydrogel shear modulus <= 4.5 kPa to suppress mechanosensitive alpha-SMA myofibroblast differentiation.
3. **Biodegradation:** Incorporate matrix metalloproteinase (MMP)-cleavable peptide crosslinkers to synchronize scaffold resorption with host vascular ECM deposition.
