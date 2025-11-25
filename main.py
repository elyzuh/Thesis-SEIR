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

# YOUR FINAL SEIR-PINN MODEL
from models.EpiSEIRCNNRNNRes_PINN import EpiSEIRCNNRNNRes_PINN


# ========================================
# Argument Parser
# ========================================
parser = argparse.ArgumentParser(description='Thesis: EpiSEIRCNNRNNRes-PINN — SEIR Physics-Informed Deep Learning with Learned Mobility')
parser.add_argument('--data', type=str, required=True, help='Path to data file')
parser.add_argument('--train', type=float, default=0.6)
parser.add_argument('--valid', type=float, default=0.2)
parser.add_argument('--model', type=str, default='EpiSEIRCNNRNNRes_PINN')
parser.add_argument('--hidRNN', type=int, default=50)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--save_dir', type=str, default='./save')
parser.add_argument('--save_name', type=str, default='THESIS_FINAL_PERFECT')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--window', type=int, default=24)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--seed', type=int, default=54321)
parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 = CPU)')
parser.add_argument('--epilambda', type=float, default=0.3, help='Weight for epidemiological consistency loss')
parser.add_argument('--lambda_pde', type=float, default=1.0, help='Weight for PDE residual loss')
parser.add_argument('--lambda_ngm', type=float, default=0.5, help='Weight for NGM (R0) regularization')
parser.add_argument('--sim_mat', type=str, default=None, help='Legacy argument')

args = parser.parse_args()
print(args)


# ========================================
# Device Setup
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


# ========================================
# Model Initialization
# ========================================
print("Initializing EpiSEIRCNNRNNRes_PINN...")
model = EpiSEIRCNNRNNRes_PINN(args, Data, args.horizon)
model = model.to(device)

# CRITICAL FIX: Move all data splits to GPU once (prevents CPU/GPU mismatch)
print("Moving dataset tensors to GPU...")
Data.train = [(x.to(device), y.to(device)) for x, y in Data.train]
Data.valid = [(x.to(device), y.to(device)) for x, y in Data.valid]
Data.test  = [(x.to(device), y.to(device)) for x, y in Data.test]
print(f"→ All data moved to {device}")

nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model ready on {device} | Trainable parameters: {nParams:,}")


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
val_losses = []

print("\n" + "="*80)
print("STARTING TRAINING — EpiSEIRCNNRNNRes-PINN WITH LEARNED MOBILITY & PHYSICS")
print("="*80)

for epoch in range(1, args.epochs + 1):
    epoch_start = time.time()

    train_loss = train(
        Data, Data.train, model, criterion, optim, args.batch_size,
        args.model, args.epilambda, args.lambda_pde, args.lambda_ngm
    )

    val_rse, val_rae, val_corr = evaluate(
        Data, Data.valid, model, criterion, l1_criterion,
        args.batch_size, args.model, args.lambda_pde, args.lambda_ngm
    )

    elapsed = time.time() - epoch_start
    print(f"| Epoch {epoch:3d} | {elapsed:5.1f}s | Train: {train_loss:.6f} "
          f"| Val RSE: {val_rse:.4f} | RAE: {val_rae:.4f} | Corr: {val_corr:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_rse)

    if val_rse < best_val:
        best_val = val_rse
        torch.save(model.state_dict(), f"{args.save_dir}/{args.save_name}.pt")
        print("    → BEST MODEL SAVED!")


# ========================================
# Final Evaluation & Visualization
# ========================================
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(f"{args.save_dir}/{args.save_name}.pt", map_location=device))
model.eval()

test_rse, test_rae, test_corr = evaluate(
    Data, Data.test, model, criterion, l1_criterion,
    args.batch_size, args.model, args.lambda_pde, args.lambda_ngm
)
print(f"\nFINAL TEST RESULTS")
print(f"→ RSE: {test_rse:.4f} | RAE: {test_rae:.4f} | Correlation: {test_corr:.4f}")

# Get predictions and latent variables
outputs = GetPrediction(Data, Data.test, model, criterion, l1_criterion, args.batch_size, args.model)
X_true, Y_pred, Y_true, BetaList, SigmaList, GammaList, NGMList, EList = outputs

save_dir = f"./Figures/{args.save_name}_final/"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving plots to: {save_dir}")

PlotTrends(X_true.transpose(2, 0, 1), Y_true.T, Y_pred.T, save_dir, args.horizon)
PlotPredictionTrends(Y_true.T, Y_pred.T, save_dir)
PlotLatentE(EList, save_dir)
PlotParameters(BetaList.T, SigmaList.T, GammaList.T, save_dir)

# Plot learned mobility matrix
with torch.no_grad():
    dummy = torch.zeros(1, args.window, Data.m, device=device)
    _, _, _, _, _, Pi, _, _ = model(dummy)
    mobility = torch.softmax(Pi, dim=1).cpu().numpy()
PlotEachMatrix(mobility[None, ...], "Learned Inter-Region Mobility Matrix", "Mobility", save_dir)

# Training curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss", color="brown", linewidth=2)
plt.plot(val_losses, label="Validation RSE", color="navy", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss / RSE")
plt.title("EpiSEIRCNNRNNRes-PINN Training Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_dir}training_curve.pdf", bbox_inches='tight', dpi=300)
plt.show()

print(f"\nSUCCESS! Your thesis model has finished training.")
print(f"→ Model saved: {args.save_dir}/{args.save_name}.pt")
print(f"→ Figures saved in: {save_dir}")
print(f"→ Test RSE: {test_rse:.4f} | RAE: {test_rae:.4f} | Corr: {test_corr:.4f}")