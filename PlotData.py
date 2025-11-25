# PlotData.py
# Thesis-ready visualization for Epi-SEIR-CNNRNN-Res-PINN
# Plots + Saves + Shows all figures
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch
import torch.nn as nn
import argparse

# Import your updated utils and model
from utils_ModelTrainEval import evaluate, GetPrediction
from utils import Data_utility
from models.EpiSEIRCNNRNNRes_PINN import EpiSEIRCNNRNNRes_PINN  # YOUR MODEL


# ========================================
# 1. Plot Prediction vs Ground Truth
# ========================================
def PlotTrends(real, pred, save_dir, model_name=""):
    print(f"Plotting prediction trends for {real.shape[0]} locations...")
    os.makedirs(save_dir, exist_ok=True)
    T = real.shape[1]
    x = np.arange(T)

    for i in range(real.shape[0]):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, real[i], 'o-', color='tab:blue', label='Ground Truth', markersize=3)
        ax.plot(x, pred[i], 's--', color='tab:orange', label='Prediction', markersize=3)
        ax.set_title(f"Location {i+1} - {model_name}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Influenza Activity Level")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = f"{save_dir}Prediction_Location_{i+1}.pdf"
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()


# ========================================
# 2. Plot Latent Exposed E(t)
# ========================================
def PlotLatentE(E_sim, save_dir):
    print("Plotting inferred latent Exposed E(t)...")
    os.makedirs(save_dir, exist_ok=True)
    T = E_sim.shape[1]
    x = np.arange(T)

    for i in range(E_sim.shape[0]):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, E_sim[i], '^-', color='tab:green', label='Inferred E(t)', markersize=4)
        ax.set_title(f"Latent Exposed - Location {i+1}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Exposed (E)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = f"{save_dir}LatentE_Location_{i+1}.pdf"
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()


# ========================================
# 3. Plot Epidemiological Parameters
# ========================================
def PlotParameters(beta, sigma, gamma, save_dir):
    print("Plotting β, σ, γ over time...")
    os.makedirs(save_dir, exist_ok=True)
    T = beta.shape[1]
    x = np.arange(T)

    param_names = ['β (Transmission)', 'σ (Incubation Rate)', 'γ (Recovery Rate)']
    colors = ['tab:red', 'tab:purple', 'tab:cyan']
    data = [beta, sigma, gamma]

    for i, (param, name, color) in enumerate(zip(data, param_names, colors)):
        fig, ax = plt.subplots(figsize=(8, 4))
        for loc in range(param.shape[0]):
            ax.plot(x, param[loc], label=f"Loc {loc+1}" if loc < 5 else "", alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Week")
        ax.set_ylabel(name.split()[0])
        if param.shape[0] <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        fname = f"{save_dir}{name.split()[0]}_TimeSeries.pdf"
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()


# ========================================
# 4. Plot Heatmap (NGM, Mobility)
# ========================================
def PlotHeatmap(matrix, title, savename, save_dir, cmap='Spectral_r', annot=False):
    print(f"Plotting {title}...")
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix, cmap=cmap, annot=annot, fmt=".2f",
        square=True, cbar=True, ax=ax,
        xticklabels=False, yticklabels=False,
        linewidths=0.1, linecolor='gray'
    )
    ax.set_title(title)
    fname = f"{save_dir}{savename}.pdf"
    plt.savefig(fname, transparent=True, bbox_inches='tight')
    plt.show()
    plt.close()


# ========================================
# MAIN
# ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot SEIR-PINN Results')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--save_dir', type=str, default='./save', help='Model save dir')
    parser.add_argument('--save_name', type=str, default='seir_pinn_final', help='Model name')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, default='EpiSEIRCNNRNNRes_PINN')
    args = parser.parse_args()

    print(f"Loading model: {args.save_name}.pt")
    print(f"Data: {args.data}")

    # Load data
    Data = Data_utility(args)

    # Load model
    model = EpiSEIRCNNRNNRes_PINN(args, Data)
    model_path = f"{args.save_dir}/{args.save_name}.pt"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Evaluation
    evaluateL2 = nn.MSELoss(reduction='sum')
    evaluateL1 = nn.L1Loss(reduction='sum')
    test_rse, test_rae, test_corr = evaluate(
        Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size, args.model
    )
    print(f"Test RSE: {test_rse:.4f} | RAE: {test_rae:.4f} | Corr: {test_corr:.4f}")

    # Get predictions
    outputs = GetPrediction(
        Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size, args.model
    )

    if args.model == "EpiSEIRCNNRNNRes_PINN":
        X_true, Y_pred, Y_true, Beta, Sigma, Gamma, NGM, E_sim = outputs
    else:
        raise ValueError("This script is for EpiSEIRCNNRNNRes_PINN only")

    # Create output directory
    save_dir = f"./Figures/{args.save_name}/"
    os.makedirs(save_dir, exist_ok=True)

    # === PLOT ALL ===
    PlotTrends(Y_true, Y_pred, save_dir, "SEIR-PINN")
    PlotLatentE(E_sim, save_dir)
    PlotParameters(Beta, Sigma, Gamma, save_dir)

    # Average NGM and Mobility
    avg_ngm = NGM.mean(axis=0)  # (2m, 2m)
    I_ngm = avg_ngm[Data.m:, Data.m:]  # I→I block
    mobility = torch.softmax(model.mask_mat, dim=1).detach().cpu().numpy()

    PlotHeatmap(I_ngm, "NGM (I→I Block)", "NGM_II", save_dir)
    PlotHeatmap(mobility, "Learned Mobility Matrix", "Mobility", save_dir, cmap='Blues')

    print(f"All plots saved and displayed in: {save_dir}")