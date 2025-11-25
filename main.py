#!/usr/bin/env python
# encoding: utf-8
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from utils import Data_utility
from utils_ModelTrainEval import train, evaluate, GetPrediction
import Optim
from PlotFunc import *

# YOUR FINAL THESIS MODEL
from models.EpiSEIRCNNRNNRes_PINN import EpiSEIRCNNRNNRes_PINN


# ========================================
# Argument Parser
# ========================================
parser = argparse.ArgumentParser(description='Thesis: SEIR Physics-Informed CNN-RNN-Residual with Learned Mobility')
parser.add_argument('--data', type=str, required=True, help='Path to data file')
parser.add_argument('--train', type=float, default=0.6)
parser.add_argument('--valid', type=float, default=0.2)
parser.add_argument('--model', type=str, default='EpiSEIRCNNRNNRes_PINN')
parser.add_argument('--hidRNN', type=int, default=50)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--save_dir', type=str, default='./save')
parser.add_argument('--save_name', type=str, default='THESIS_SEIR_PINN_FINAL')
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--horizon', type=int, default=4)
parser.add_argument('--window', type=int, default=24)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--seed', type=int, default=54321)
parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 = CPU)')
parser.add_argument('--epilambda', type=float, default=0.3, help='Weight for epidemiological consistency loss')
parser.add_argument('--lambda_pde', type=float, default=1.0, help='Weight for PDE residual loss')
parser.add_argument('--lambda_ngm', type=float, default=0.5, help='Weight for NGM (R0) regularization')
args = parser.parse_args()
print(args)


# ========================================
# Device & Seed Setup
# ========================================
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
print(f"Using device: {device}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs("./Figures", exist_ok=True)


# ========================================
# Load Data
# ========================================
Data = Data_utility(args)
print(f"Data loaded: {Data.m} regions × {Data.T} weeks")
print(f"Dataset → Train: {len(Data.train)} | Valid: {len(Data.valid)} | Test: {len(Data.test)} samples")


# ========================================
# Model Initialization
# ========================================
print("Initializing EpiSEIRCNNRNNRes_PINN (SEIR-PINN with Learned Mobility)...")
model = EpiSEIRCNNRNNRes_PINN(args, Data, args.horizon)
model = model.to(device)

# Move data to GPU once
Data.train = [(x.to(device), y.to(device)) for x, y in Data.train]
Data.valid = [(x.to(device), y.to(device)) for x, y in Data.valid]
Data.test  = [(x.to(device), y.to(device)) for x, y in Data.test]
print(f"→ All data moved to {device}")

nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model ready | Trainable parameters: {nParams:,}")


# ========================================
# Loss & Optimizer
# ========================================
criterion = nn.MSELoss(reduction='sum').to(device)
l1_criterion = nn.L1Loss(reduction='sum').to(device)

optim = Optim.Optim(
    params=model.parameters(),
    method=args.optim,
    lr=args.lr,
    max_grad_norm=args.clip,
    weight_decay=args.weight_decay,
    named_params=dict(model.named_parameters())
)


# ========================================
# Training Loop
# ========================================
best_val = float('inf')
train_losses = []
val_rses = []

print("\n" + "="*90)
print("STARTING TRAINING — SEIR Physics-Informed Neural Network with Learned Mobility (Thesis Final)")
print("="*90)

for epoch in range(1, args.epochs + 1):
    epoch_start = time.time()

    train_loss = train(
        Data, Data.train, model, criterion, optim, args.batch_size,
        args.model, args.epilambda, args.lambda_pde, args.lambda_ngm
    )

    # Now evaluate returns 4 values: RSE, RAE, Corr, R²
    val_rse, val_rae, val_corr, val_r2 = evaluate(
        Data, Data.valid, model, criterion, l1_criterion,
        args.batch_size, args.model
    )

    elapsed = time.time() - epoch_start
    print(f"| Epoch {epoch:3d} | {elapsed:5.1f}s | Train: {train_loss:.6f} "
          f"| Val → RSE: {val_rse:.4f} | RAE: {val_rae:.4f} | Corr: {val_corr:.4f} | R²: {val_r2:.4f}")

    train_losses.append(train_loss)
    val_rses.append(val_rse)

    if val_rse < best_val:
        best_val = val_rse
        torch.save(model.state_dict(), f"{args.save_dir}/{args.save_name}.pt")
        print("    → BEST MODEL SAVED!")


# ========================================
# Final Test Evaluation
# ========================================
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(f"{args.save_dir}/{args.save_name}.pt", map_location=device))
model.eval()

test_rse, test_rae, test_corr, test_r2 = evaluate(
    Data, Data.test, model, criterion, l1_criterion,
    args.batch_size, args.model
)

print(f"\n" + "="*60)
print("FINAL TEST RESULTS (US HHS Influenza Forecasting)")
print(f"→ RSE        : {test_rse:.4f}")
print(f"→ RAE        : {test_rae:.4f}")
print(f"→ Correlation: {test_corr:.4f}")
print(f"→ R²         : {test_r2:.4f}")
print("="*60)


# ========================================
# Predictions & Visualization
# ========================================
outputs = GetPrediction(Data, Data.test, model, criterion, l1_criterion, args.batch_size, args.model)
X_true, Y_pred, Y_true, beta_avg, sigma_avg, gamma_avg, Pi_matrix, E_trajectories = outputs

save_dir = f"./Figures/{args.save_name}_results/"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving all figures to: {save_dir}")

# Time series plots
# X_true: (samples, window, regions)
# Y_true/Y_pred: (samples, horizon, regions)
PlotTrends(X_true.transpose(2, 0, 1), 
           Y_true.transpose(2, 0, 1), 
           Y_pred.transpose(2, 0, 1), 
           save_dir, args.horizon)
PlotPredictionTrends(Y_true.T, Y_pred.T, save_dir)
# Parameters (from last batch — representative)
PlotParameters(beta_avg[None, :].T, sigma_avg[None, :].T, gamma_avg[None, :].T, save_dir)

# Latent Exposed (full batch trajectories)
PlotLatentE(E_trajectories, save_dir)

# Learned mobility matrix
PlotEachMatrix(Pi_matrix[None, ...], "Learned Inter-Region Mobility Matrix", "Mobility", save_dir)

# Learned mobility matrix (average over test set)
with torch.no_grad():
    dummy = torch.zeros(1, args.window, Data.m, device=device)
    _, _, _, _, _, Pi, _, _ = model(dummy)
    mobility = torch.softmax(Pi, dim=1).cpu().numpy()
PlotEachMatrix(mobility[None, ...], "Learned Inter-Region Mobility Matrix", "Mobility", save_dir)

# Training curve
plt.figure(figsize=(11, 6))
plt.plot(train_losses, label="Training Loss", color="brown", linewidth=2.5)
plt.plot(val_rses, label="Validation RSE", color="navy", linewidth=2.5)
plt.axhline(y=test_rse, color='green', linestyle='--', linewidth=2, label=f"Test RSE = {test_rse:.4f}")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss / RSE", fontsize=14)
plt.title("SEIR-PINN Training Curve (US HHS Influenza)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_dir}training_curve.pdf", bbox_inches='tight', dpi=300)
plt.show()

print(f"\nTHESIS MODEL TRAINING COMPLETED SUCCESSFULLY!")
print(f"→ Model saved: {args.save_dir}/{args.save_name}.pt")
print(f"→ All figures saved in: {save_dir}")
print(f"→ Final Test R² = {test_r2:.4f} — Ready for submission!")