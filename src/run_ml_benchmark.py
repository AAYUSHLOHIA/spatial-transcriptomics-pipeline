"""
Script to execute Linear Regression vs. Basic Neural Network (MLP) Benchmark,
generate comparison parity plots, and output the Model Interpretation Guide.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spatial_dataset import generate_biomaterial_visium_dataset
from src.predictive_models import BiomaterialHostPredictor


def run_benchmark():
    print("=" * 75)
    print("  RUNNING MACHINE LEARNING BENCHMARK: LINEAR REGRESSION VS. NEURAL NETWORK  ")
    print("=" * 75)
    
    os.makedirs("results/figures", exist_ok=True)
    
    # 1. Generate or load datasets
    print("\n[Step 1/4] Preparing spatial datasets for Solid, Microporous, and Microgel scaffolds...")
    adatas = {
        "solid": generate_biomaterial_visium_dataset("solid", grid_size=34, random_seed=42),
        "microporous": generate_biomaterial_visium_dataset("microporous", grid_size=34, random_seed=42),
        "microgel": generate_biomaterial_visium_dataset("microgel", grid_size=34, random_seed=42)
    }
    
    # 2. Extract features & targets
    print("\n[Step 2/4] Extracting biophysical and spatial transcriptomic features...")
    predictor = BiomaterialHostPredictor(random_state=42)
    X, y, df_dataset = predictor.prepare_dataset(adatas)
    print(f"  -> Total dataset size: {X.shape[0]} samples with {X.shape[1]} features.")
    print(f"  -> Features: {predictor.feature_names}")
    
    # 3. Train & Evaluate
    print("\n[Step 3/4] Training Linear Regression (Ridge) and Basic Neural Network (MLP)...")
    results = predictor.train_and_evaluate(X, y, test_size=0.2)
    
    lin_metrics = results["metrics"]["Linear_Regression"]
    mlp_metrics = results["metrics"]["Basic_Neural_Network_MLP"]
    
    print("\n------------------------------------------------------------")
    print("                    MODEL PERFORMANCE COMPARISON             ")
    print("------------------------------------------------------------")
    print(f"  Metric              Linear Regression    Neural Network (MLP)")
    print(f"  Train R²:           {lin_metrics['train_r2']:<20} {mlp_metrics['train_r2']}")
    print(f"  Test R²:            {lin_metrics['test_r2']:<20} {mlp_metrics['test_r2']}")
    print(f"  Test RMSE:          {lin_metrics['test_rmse']:<20} {mlp_metrics['test_rmse']}")
    print(f"  Test MAE:           {lin_metrics['test_mae']:<20} {mlp_metrics['test_mae']}")
    print("------------------------------------------------------------")
    
    print("\nTop Linear Regression Feature Coefficients (Biological Drivers):")
    for feat, coef in list(results["linear_coefficients"].items())[:8]:
        direction = "Positive Driver (+)" if coef > 0 else "Negative Driver (-)"
        print(f"  {feat:<25}: {coef:>7.3f}  [{direction}]")
        
    # 4. Generate Publication Figure
    print("\n[Step 4/4] Generating side-by-side comparison figure...")
    y_true = np.array(results["test_predictions"]["y_true"])
    y_pred_lin = np.array(results["test_predictions"]["y_pred_linear"])
    y_pred_mlp = np.array(results["test_predictions"]["y_pred_mlp"])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Linear Regression Parity Plot
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred_lin, alpha=0.7, color="#3498DB", edgecolors="none", s=35)
    min_v, max_v = min(y_true.min(), y_pred_lin.min()), max(y_true.max(), y_pred_lin.max())
    ax1.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Ideal Line (y = x)")
    ax1.set_title(f"Linear Regression (Test R² = {lin_metrics['test_r2']:.3f})", fontweight="bold")
    ax1.set_xlabel("True Tissue Integration Score")
    ax1.set_ylabel("Predicted Score")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    
    # Panel B: Neural Network (MLP) Parity Plot
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred_mlp, alpha=0.7, color="#2ECC71", edgecolors="none", s=35)
    min_v2, max_v2 = min(y_true.min(), y_pred_mlp.min()), max(y_true.max(), y_pred_mlp.max())
    ax2.plot([min_v2, max_v2], [min_v2, max_v2], "r--", linewidth=1.5, label="Ideal Line (y = x)")
    ax2.set_title(f"Basic Neural Network / MLP (Test R² = {mlp_metrics['test_r2']:.3f})", fontweight="bold")
    ax2.set_xlabel("True Tissue Integration Score")
    ax2.set_ylabel("Predicted Score")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()
    
    # Panel C: Feature Coefficients (Linear Model Interpretability)
    ax3 = axes[1, 0]
    top_feats = list(results["linear_coefficients"].keys())[:10]
    top_vals = [results["linear_coefficients"][k] for k in top_feats]
    bar_colors = ["#2ECC71" if v > 0 else "#E74C3C" for v in top_vals]
    ax3.barh(top_feats[::-1], top_vals[::-1], color=bar_colors[::-1])
    ax3.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax3.set_title("Linear Regression Standardized Coefficients (β)", fontweight="bold")
    ax3.set_xlabel("Effect on Tissue Integration (+ Pro-healing | - Fibrosis)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    
    # Panel D: Bar chart comparing R² and RMSE
    ax4 = axes[1, 1]
    models = ["Linear Regression", "Basic Neural Net (MLP)"]
    r2_vals = [lin_metrics["test_r2"], mlp_metrics["test_r2"]]
    rmse_vals = [lin_metrics["test_rmse"], mlp_metrics["test_rmse"]]
    
    x = np.arange(len(models))
    width = 0.35
    ax4.bar(x - width/2, r2_vals, width, label="Test R² (higher is better)", color="#3498DB")
    ax4.bar(x + width/2, rmse_vals, width, label="Test RMSE (lower is better)", color="#E67E22")
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, fontweight="bold")
    ax4.set_title("Model Benchmark Metrics Comparison", fontweight="bold")
    ax4.legend()
    ax4.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Predictive Modeling of Biomaterial-Host Interface: Linear vs. Neural Network", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig_path = "results/figures/model_comparison_linear_vs_mlp.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved model comparison figure to {fig_path}")
    
    # Save results json
    with open("results/model_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Save model interpretation guide
    guide_text = predictor.generate_methodology_summary()
    with open("results/MODEL_INTERPRETATION_GUIDE.md", "w") as f:
        f.write(guide_text)
    print(f"  -> Saved Model Interpretation Guide to results/MODEL_INTERPRETATION_GUIDE.md")
    
    return results


if __name__ == "__main__":
    run_benchmark()
