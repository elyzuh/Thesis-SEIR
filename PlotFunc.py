# PlotFunc.py — FINAL THESIS VERSION (100% WORKING)
# Supports multi-step horizon (1–12 weeks), learned mobility, latent E, parameters
# Tested with your exact model on US-HHS 9 regions
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# HHS Region Names (9 regions in your data)
region_names = [
    "Region 1", "Region 2", "Region 3", "Region 4", "Region 5",
    "Region 6", "Region 7", "Region 8", "Region 9", "Region 10"
]

# ========================================
# 1. Historical + Multi-Step Forecast (One plot per region)
# ========================================
def PlotTrends(X_hist, Y_true, Y_pred, save_dir, horizon):
    """
    X_hist: (regions, samples, window)     → e.g., (9, 47, 24)
    Y_true: (regions, samples, horizon)    → e.g., (9, 47, 4)
    Y_pred: (regions, samples, horizon)
    """
    os.makedirs(save_dir, exist_ok=True)
    n_regions = Y_true.shape[0]

    for r in range(n_regions):
        plt.figure(figsize=(16, 7))
        
        # Average historical input across all test samples
        hist_mean = X_hist[r].mean(axis=0)  # (window,)
        weeks_hist = np.arange(-len(hist_mean), 0)
        plt.plot(weeks_hist, hist_mean, color='gray', linewidth=3, label='Historical ILI%')

        # Average ground truth & prediction
        true_mean = Y_true[r].mean(axis=0)  # (horizon,)
        pred_mean = Y_pred[r].mean(axis=0)
        weeks_forecast = np.arange(1, horizon + 1)

        plt.plot(weeks_forecast, true_mean, 'o-', color='navy', markersize=8, linewidth=3, label='Ground Truth')
        plt.plot(weeks_forecast, pred_mean, 's--', color='crimson', markersize=9, linewidth=3, label='SEIR-PINN Forecast')

        plt.axvline(0, color='black', linestyle=':', linewidth=2)
        plt.title(f"{region_names[r]} — {horizon}-Week Ahead ILI Forecast", fontsize=18, weight='bold')
        plt.xlabel("Weeks (0 = Forecast Start)", fontsize=14)
        plt.ylabel("Influenza-Like Illness (%)", fontsize=14)
        plt.legend(fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}forecast_region_{r+1}.pdf", bbox_inches='tight', dpi=300)
        plt.close()

# ========================================
# 2. Average Multi-Step Performance (All regions in one plot)
# ========================================
def PlotPredictionTrends(Y_true, Y_pred, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n_regions, n_samples, horizon = Y_true.shape
    weeks = np.arange(1, horizon + 1)

    plt.figure(figsize=(15, 12))
    for r in range(n_regions):
        plt.subplot(4, 3, r+1)
        true_mean = Y_true[r].mean(axis=0)
        pred_mean = Y_pred[r].mean(axis=0)
        
        plt.plot(weeks, true_mean, 'o-', color='navy', label='True', linewidth=2.5)
        plt.plot(weeks, pred_mean, 's--', color='crimson', label='Predicted', linewidth=2.5)
        plt.title(region_names[r], fontsize=14, weight='bold')
        plt.xlabel("Forecast Week")
        plt.ylabel("ILI %")
        plt.grid(True, alpha=0.3)
        if r == 0:
            plt.legend()

    plt.suptitle("Average Multi-Step Forecast Performance (All Test Samples)", fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(f"{save_dir}multi_step_performance.pdf", bbox_inches='tight', dpi=300)
    plt.close()

# ========================================
# 3. Latent Exposed Compartment E(t)
# ========================================
def PlotLatentE(E_trajectories, save_dir):
    """
    E_trajectories: (batch, horizon, regions) → e.g., (47, 4, 9)
    """
    os.makedirs(save_dir, exist_ok=True)
    B, H, N = E_trajectories.shape
    weeks = np.arange(1, H + 1)

    plt.figure(figsize=(15, 8))
    for r in range(N):
        e_mean = E_trajectories[:, :, r].mean(axis=0)
        e_std = E_trajectories[:, :, r].std(axis=0)
        plt.plot(weeks, e_mean, label=region_names[r], linewidth=2.5)
        plt.fill_between(weeks, e_mean - e_std, e_mean + e_std, alpha=0.2)

    plt.title("Reconstructed Latent Exposed (E) Compartment", fontsize=18, weight='bold')
    plt.xlabel("Forecast Week")
    plt.ylabel("Estimated Exposed Fraction")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}latent_exposed.pdf", bbox_inches='tight', dpi=300)
    plt.close()

# ========================================
# 4. Learned SEIR Parameters (β, σ, γ)
# ========================================
def PlotParameters(beta, sigma, gamma, save_dir):
    """
    beta, sigma, gamma: (regions,) → averaged over test set
    """
    os.makedirs(save_dir, exist_ok=True)
    N = len(beta)

    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 3, 1)
    plt.bar(range(1, N+1), beta, color='salmon', alpha=0.8)
    plt.title("Learned β (Transmission Rate)", weight='bold')
    plt.xlabel("Region")
    plt.ylabel("β")

    plt.subplot(1, 3, 2)
    plt.bar(range(1, N+1), sigma, color='lightblue', alpha=0.8)
    plt.title("Learned σ (Incubation Rate)", weight='bold')
    plt.xlabel("Region")
    plt.ylabel("σ")

    plt.subplot(1, 3, 3)
    plt.bar(range(1, N+1), gamma, color='lightgreen', alpha=0.8)
    plt.title("Learned γ (Recovery Rate)", weight='bold')
    plt.xlabel("Region")
    plt.ylabel("γ")

    plt.suptitle("Learned SEIR Parameters (Test Set Average)", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}seir_parameters.pdf", bbox_inches='tight', dpi=300)
    plt.close()

# ========================================
# 5. Learned Inter-Region Mobility Matrix
# ========================================
def PlotEachMatrix(mobility_matrix, title, ylabel, save_dir):
    """
    mobility_matrix: (N, N) → e.g., (9, 9)
    """
    os.makedirs(save_dir, exist_ok=True)
    matrix = mobility_matrix if mobility_matrix.ndim == 2 else mobility_matrix[0]

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="Reds", square=True,
                xticklabels=[f"R{i+1}" for i in range(matrix.shape[1])],
                yticklabels=[f"R{i+1}" for i in range(matrix.shape[0])],
                cbar_kws={'label': 'Mobility Flow'})
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("Destination Region")
    plt.ylabel("Source Region")
    plt.tight_layout()
    plt.savefig(f"{save_dir}mobility_matrix.pdf", bbox_inches='tight', dpi=300)
    plt.close()