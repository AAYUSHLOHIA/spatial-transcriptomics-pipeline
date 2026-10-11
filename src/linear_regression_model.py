"""
Linear Regression Model for Biomaterial-Host Interface Analysis.
The simplest, most interpretable machine learning model to predict
host tissue integration and evaluate foreign body capsule risk.

Formula:
  y_hat = beta_0 + beta_1 * Vegfa + beta_2 * Arg1 + ... + beta_k * Stiffness + beta_m * Porosity

Why this model is ideal for biomaterial-host interface modeling:
1. Exact transparency: Every feature has a direct, signed coefficient (beta).
2. Biological meaning: Positive beta = pro-healing/vascularization; Negative beta = fibrosis/scarring.
3. No black-box complexity: Solved using standard Ordinary Least Squares (OLS) or Ridge regularization.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.spatial_dataset import generate_biomaterial_visium_dataset


def build_and_evaluate_linear_model(output_dir: str = "results"):
    """
    Builds, trains, and evaluates a transparent Ridge Linear Regression model.
    """
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    
    print("=" * 70)
    print("  STEP 1: GENERATING SPATIAL VISIUM DATASETS (SOLID vs MICROGEL)  ")
    print("=" * 70)
    adatas = {
        "solid": generate_biomaterial_visium_dataset("solid", grid_size=32, random_seed=42),
        "microporous": generate_biomaterial_visium_dataset("microporous", grid_size=32, random_seed=42),
        "microgel": generate_biomaterial_visium_dataset("microgel", grid_size=32, random_seed=42)
    }
    
    print(f"Loaded 3 biomaterial scaffolds: Solid, Microporous, and Microgel.")
    
    print("\n" + "=" * 70)
    print("  STEP 2: EXTRACTING EXPLANATORY FEATURES & BIOLOGICAL TARGET  ")
    print("=" * 70)
    
    key_genes = ["Vegfa", "Pecam1", "Arg1", "Mrc1", "Tnf", "Il1b", "Acta2", "Col1a1", "Tgfb1"]
    records = []
    
    for cond_name, adata in adatas.items():
        obs = adata.obs
        var_list = list(adata.var_names)
        cfg = adata.uns.get("parameters", {})
        
        porosity = float(cfg.get("porosity_pct", 0.0))
        pore_size = float(cfg.get("mean_pore_size_um", 0.0))
        stiffness = float(cfg.get("modulus_kpa", 10.0))
        
        for i in range(adata.n_obs):
            row = {
                "distance_to_interface_um": float(obs["dist_to_interface_um"].iloc[i]),
                "scaffold_porosity_pct": porosity,
                "scaffold_pore_size_um": pore_size,
                "scaffold_stiffness_kpa": stiffness,
            }
            
            for g in key_genes:
                if g in var_list:
                    idx = var_list.index(g)
                    row[f"gene_{g}"] = float(adata.X[i, idx])
                else:
                    row[f"gene_{g}"] = 0.0
                    
            # Target: Tissue Integration Score (0 - 100)
            # High angiogenesis (Vegfa) + M2 repair (Arg1) promotes healing
            # High myofibroblast (Acta2) + collagen (Col1a1) + high stiffness promotes fibrotic capsule
            y_val = (
                20.0
                + 18.0 * row.get("gene_Vegfa", 0.0)
                + 16.0 * row.get("gene_Arg1", 0.0)
                - 14.0 * row.get("gene_Acta2", 0.0)
                - 12.0 * row.get("gene_Col1a1", 0.0)
                + 0.25 * porosity
                - 0.05 * stiffness
            )
            row["tissue_integration_score"] = float(np.clip(y_val, 0.0, 100.0))
            records.append(row)
            
    df = pd.DataFrame(records)
    feature_cols = [c for c in df.columns if c != "tissue_integration_score"]
    
    X = df[feature_cols].values
    y = df["tissue_integration_score"].values
    
    print(f"Total spatial spots: {len(df)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    print("\n" + "=" * 70)
    print("  STEP 3: TRAINING LINEAR REGRESSION (80% TRAIN / 20% TEST)  ")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features so coefficients reflect relative importance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit Linear Model (Ridge with L2 penalty to handle correlated genes)
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Model Intercept (beta_0): {model.intercept_:.2f}")
    print(f"Train R²:   {r2_train:.3f}")
    print(f"Test  R²:   {r2_test:.3f} (explains {r2_test*100:.1f}% of variance)")
    print(f"Test  RMSE: {rmse_test:.2f} points (out of 100)")
    print(f"Test  MAE:  {mae_test:.2f} points")
    
    print("\n" + "=" * 70)
    print("  STEP 4: INTERPRETING BIOLOGICAL COEFFICIENTS (BETA WEIGHTS)  ")
    print("=" * 70)
    
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient_Beta": model.coef_
    }).sort_values(by="Coefficient_Beta", ascending=False).reset_index(drop=True)
    
    for _, row in coef_df.iterrows():
        sign = "(+) Pro-Healing / Vascularization" if row["Coefficient_Beta"] > 0 else "(-) Fibrotic Encapsulation"
        print(f"  {row['Feature']:<30}: {row['Coefficient_Beta']:>+7.3f}   {sign}")
        
    coef_df.to_csv(f"{output_dir}/linear_model_coefficients.csv", index=False)
    
    print("\n" + "=" * 70)
    print("  STEP 5: GENERATING PUBLICATION FIGURE  ")
    print("=" * 70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Actual vs Predicted Parity Plot
    ax1.scatter(y_test, y_pred_test, alpha=0.6, color="#2980B9", edgecolors="none", s=30)
    min_v, max_v = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    ax1.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Ideal (y = x)")
    ax1.set_title(f"Linear Regression: Actual vs Predicted\n(Test R² = {r2_test:.3f}, RMSE = {rmse_test:.2f})", fontweight="bold")
    ax1.set_xlabel("True Tissue Integration Score (0 - 100)")
    ax1.set_ylabel("Predicted Integration Score (0 - 100)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")
    
    # Subplot 2: Standardized Feature Importance Bar Chart
    colors = ["#27AE60" if c > 0 else "#C0392B" for c in coef_df["Coefficient_Beta"]]
    ax2.barh(coef_df["Feature"][::-1], coef_df["Coefficient_Beta"][::-1], color=colors[::-1])
    ax2.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_title("Standardized Regression Coefficients (β)\n[ Green: Promotes Healing  |  Red: Promotes Scar/FBR ]", fontweight="bold")
    ax2.set_xlabel("Effect Size (Standardized Beta)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    fig_path = f"{output_dir}/figures/linear_regression_results.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication figure to: {fig_path}")
    
    return {
        "r2_test": r2_test,
        "rmse_test": rmse_test,
        "coefficients": coef_df.to_dict(orient="records")
    }


if __name__ == "__main__":
    build_and_evaluate_linear_model()
