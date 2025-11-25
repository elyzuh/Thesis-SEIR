#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function
import argparse
import math
import time
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from utils import Data_utility
from utils_ModelTrainEval import train, evaluate, GetPrediction
import Optim
from PlotFunc import *  # Updated with plt.show()
import xlwt
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === CHANGE 1: Import your SEIR-PINN model ===
from models.EpiSEIRCNNRNNRes_PINN import EpiSEIRCNNRNNRes_PINN  # YOUR MODEL

# ========================================
# Argument Parser
# ========================================
parser = argparse.ArgumentParser(description='SEIR-PINN-NGM Forecasting')

# --- Data
parser.add_argument('--data', type=str, required=True, help='CSV file path')
parser.add_argument('--train', type=float, default=0.6, help='Training ratio')
parser.add_argument('--valid', type=float, default=0.2, help='Validation ratio')
parser.add_argument('--model', type=str, default='EpiSEIRCNNRNNRes_PINN', help='Model name')

# --- CNNRNN (legacy)
parser.add_argument('--sim_mat', type=str, help='Similarity matrix (for CNNRNN_Res_epi)')
parser.add_argument('--hidRNN', type=int, default=50, help='RNN hidden units')
parser.add_argument('--residual_window', type=int, default=4, help='Residual window')
parser.add_argument('--ratio', type=float, default=1.0, help='CNNRNN-residual ratio')
parser.add_argument('--output_fun', type=str, default=None, help='Output activation')

# --- Logging
parser.add_argument('--save_dir', type=str, default='./save', help='Model save dir')
parser.add_argument('--save_name', type=str, default='seir_pinn_final', help='Model filename')

# --- Optimization
parser.add_argument('--optim', type=str, default='adam', help='Optimizer')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

# --- Prediction
parser.add_argument('--horizon', type=int, default=1, help='Forecast horizon')
parser.add_argument('--window', type=int, default=24, help='Input window')
parser.add_argument('--metric', type=int, default=1, help='Normalize RSE/RAE')
parser.add_argument('--normalize', type=int, default=0, help='Normalization method')
parser.add_argument('--seed', type=int, default=54321, help='Random seed')
parser.add_argument('--gpu', type=int, default=None, help='GPU ID')
parser.add_argument('--cuda', type=str, default=None, help='Use GPU')

# --- Epidemiology
parser.add_argument('--epilambda', type=float, default=0.2, help='NGM loss weight')
parser.add_argument('--lambda_pde', type=float, default=1.0, help='PDE loss weight')
parser.add_argument('--lambda_ngm', type=float, default=1.0, help='NGM consistency weight')

args = parser.parse_args()
print(args)

# ========================================
# Setup
# ========================================
os.makedirs(args.save_dir, exist_ok=True)

# Validate legacy model
if args.model == "CNNRNN_Res_epi" and args.sim_mat is None:
    raise ValueError("CNNRNN_Res_epi requires --sim_mat")

# GPU
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)

# Seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
Data = Data_utility(args)
print(f"Data loaded: {Data.m} locations, {Data.T} weeks")

# ========================================
# Model
# ========================================
model = EpiSEIRCNNRNNRes_PINN(args, Data)
if args.cuda:
    model.cuda()

nParams = sum(p.numel() for p in model.parameters())
print(f"Model: {args.model} | Parameters: {nParams:,}")

# Loss
criterion = nn.MSELoss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')
if args.cuda:
    criterion = criterion.cuda()
    evaluateL2 = evaluateL2.cuda()
    evaluateL1 = evaluateL1.cuda()

# Optimizer
optim = Optim.Optim(
    params=model.parameters(),
    method=args.optim,
    lr=args.lr,
    max_grad_norm=args.clip,
    named_params=dict(model.named_parameters()),
    weight_decay=args.weight_decay
)

# ========================================
# Training
# ========================================
best_val = float('inf')
train_losses, val_losses = [], []

print("Starting training...")
for epoch in range(1, args.epochs + 1):
    epoch_start = time.time()

    train_loss = train(
        Data, Data.train, model, criterion, optim, args.batch_size,
        args.model, args.epilambda, args.lambda_pde, args.lambda_ngm
    )

    val_rse, val_rae, val_corr = evaluate(
        Data, Data.valid, model, evaluateL2, evaluateL1, args.batch_size,
        args.model, args.lambda_pde, args.lambda_ngm
    )

    print(f"| Epoch {epoch:3d} | Time: {time.time()-epoch_start:5.1f}s "
          f"| Train: {train_loss:5.6f} | Val RSE: {val_rse:5.4f} | RAE: {val_rae:5.4f} | Corr: {val_corr:5.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_rse)

    # Save best
    if val_rse < best_val:
        best_val = val_rse
        torch.save(model.state_dict(), f"{args.save_dir}/{args.save_name}.pt")
        print("  → Best model saved")

# ========================================
# Final Evaluation & Plotting
# ========================================
model.load_state_dict(torch.load(f"{args.save_dir}/{args.save_name}.pt"))
model.eval()

test_rse, test_rae, test_corr = evaluate(
    Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size,
    args.model, args.lambda_pde, args.lambda_ngm
)
print(f"\nTEST → RSE: {test_rse:.4f} | RAE: {test_rae:.4f} | Corr: {test_corr:.4f}")

# Get predictions
outputs = GetPrediction(
    Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size, args.model
)
X_true, Y_pred, Y_true, BetaList, SigmaList, GammaList, NGMList, EList = outputs

# Plotting directory
save_dir = f"./Figures/{args.save_name}.epo-{args.epochs}/"
os.makedirs(save_dir, exist_ok=True)

print(f"Generating plots in: {save_dir}")

# 1. Input + Forecast
PlotTrends(X_true.transpose(2, 0, 1), Y_true.T, Y_pred.T, save_dir, args.horizon)

# 2. Test Prediction
PlotPredictionTrends(Y_true.T, Y_pred.T, save_dir)

# 3. Latent E(t)
PlotLatentE(EList, save_dir)

# 4. Parameters
PlotParameters(BetaList.T, SigmaList.T, GammaList.T, save_dir)

# 5. NGM (I→I block)
avg_ngm = NGMList.mean(axis=0)
I_ngm = avg_ngm[Data.m:, Data.m:]  # I→I
PlotEachMatrix(I_ngm[None, ...], "Next Generation Matrix (I→I)", "NGM_II", save_dir)
PlotAllMatrices(I_ngm[None, ...], "NGM", "NGM_II", save_dir)

# 6. Mobility
mobility = torch.softmax(model.mask_mat, dim=1).cpu().numpy()
PlotEachMatrix(mobility[None, ...], "Learned Mobility", "Mobility", save_dir)
PlotAllMatrices(mobility[None, ...], "Mobility", "Mobility", save_dir)

# 7. Training Curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_losses, label="Train Loss", color="brown")
ax.plot(val_losses, label="Val RSE", color="navy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig(f"{save_dir}training_curve.pdf", transparent=True, bbox_inches='tight')
plt.show()
plt.close()

print(f"\nAll done! Results saved in:\n  → {args.save_dir}/{args.save_name}.pt\n  → {save_dir}")