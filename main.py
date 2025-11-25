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

# === Import your final working SEIR-PINN model ===
from models.EpiSEIRCNNRNNRes_PINN import EpiSEIRCNNRNNRes_PINN


# ========================================
# Argument Parser
# ========================================
parser = argparse.ArgumentParser(description='SEIR-PINN-NGM: Epidemiology-aware Deep Forecasting')

parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
parser.add_argument('--train', type=float, default=0.6, help='Train ratio')
parser.add_argument('--valid', type=float, default=0.2, help='Validation ratio')
parser.add_argument('--model', type=str, default='EpiSEIRCNNRNNRes_PINN', help='Model name')

# Legacy CNNRNN args (kept for compatibility)
parser.add_argument('--sim_mat', type=str, default=None)
parser.add_argument('--hidRNN', type=int, default=50)
parser.add_argument('--residual_window', type=int, default=4)
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--output_fun', type=str, default=None)

# Training & saving
parser.add_argument('--save_dir', type=str, default='./save', help='Save directory')
parser.add_argument('--save_name', type=str, default='seir_pinn_final', help='Model filename')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--optim', type=str, default='adam')

# Prediction
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--window', type=int, default=24)
parser.add_argument('--normalize', type=int, default=2, help='0: none, 1: z-score, 2: min-max')
parser.add_argument('--seed', type=int, default=54321)

# Hardware
parser.add_argument('--gpu', type=int, default=0, help='GPU ID (set -1 for CPU)')
parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')

# Physics loss weights
parser.add_argument('--epilambda', type=float, default=0.3, help='Total physics loss weight')
parser.add_argument('--lambda_pde', type=float, default=1.0, help='PDE consistency weight')
parser.add_argument('--lambda_ngm', type=float, default=0.5, help='NGM/R0 regularization weight')

args = parser.parse_args()
print(args)


# ========================================
# Setup: Device, Seed, Folders
# ========================================
args.cuda = args.gpu >= 0 and torch.cuda.is_available()
device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs("./Figures", exist_ok=True)


# ========================================
# Load Data
# ========================================
Data = Data_utility(args)
print(f"Data loaded: {Data.m} regions × {Data.T} weeks")


# ========================================
# Model + Forward Wrapper (GPU-SAFE!)
# ========================================
print("Initializing EpiSEIRCNNRNNRes_PINN...")
model = EpiSEIRCNNRNNRes_PINN

if args.cuda:
    model = model.cuda()

nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"→ Model loaded | Trainable parameters: {nParams:,}")


# === CRITICAL: Wrap forward() to match old train/evaluate interface ===
# === FIXED: GPU-safe forward wrapper ===
_original_forward = model.forward
def wrapped_forward(x):
    if args.cuda:
        x = x.cuda()  # ← THIS LINE WAS MISSING! 
    final_pred, dl_pred, physics_pred, Pi = _original_forward(x)
    return final_pred, physics_pred
model.forward = wrapped_forward


# ========================================
# Loss & Optimizer
# ========================================
criterion = nn.MSELoss(reduction='sum')
if args.cuda:
    criterion = criterion.cuda()

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

print("\nStarting training...\n" + "="*60)
for epoch in range(1, args.epochs + 1):
    epoch_start = time.time()

    train_loss = train(
        Data, Data.train, model, criterion, optim, args.batch_size,
        args.model, args.epilambda, args.lambda_pde, args.lambda_ngm
    )

    val_rse, val_rae, val_corr = evaluate(
        Data, Data.valid, model, criterion, nn.L1Loss(reduction='sum'),
        args.batch_size, args.model, args.lambda_pde, args.lambda_ngm
    )

    elapsed = time.time() - epoch_start
    print(f"| Ep {epoch:3d} | {elapsed:5.1f}s | Train: {train_loss:.6f} "
          f"| Val RSE: {val_rse:.4f} | RAE: {val_rae:.4f} | Corr: {val_corr:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_rse)

    if val_rse < best_val:
        best_val = val_rse
        torch.save(model.state_dict(), f"{args.save_dir}/{args.save_name}.pt")
        print("    → New best model saved!")


# ========================================
# Final Test & Visualization
# ========================================
print("\n" + "="*60)
print("Loading best model for final evaluation...")
model.load_state_dict(torch.load(f"{args.save_dir}/{args.save_name}.pt"))
model.eval()

test_rse, test_rae, test_corr = evaluate(
    Data, Data.test, model, criterion, nn.L1Loss(reduction='sum'),
    args.batch_size, args.model, args.lambda_pde, args.lambda_ngm
)
print(f"TEST → RSE: {test_rse:.4f} | RAE: {test_rae:.4f} | Corr: {test_corr:.4f}")

# Get full predictions and latent states
outputs = GetPrediction(
    Data, Data.test, model, criterion, nn.L1Loss(reduction='sum'),
    args.batch_size, args.model
)
X_true, Y_pred, Y_true, BetaList, SigmaList, GammaList, NGMList, EList = outputs

# Plotting
save_dir = f"./Figures/{args.save_name}_ep{args.epochs}_h{args.horizon}_w{args.window}/"
os.makedirs(save_dir, exist_ok=True)
print(f"Generating plots → {save_dir}")

PlotTrends(X_true.transpose(2, 0, 1), Y_true.T, Y_pred.T, save_dir, args.horizon)
PlotPredictionTrends(Y_true.T, Y_pred.T, save_dir)
PlotLatentE(EList, save_dir)
PlotParameters(BetaList.T, SigmaList.T, GammaList.T, save_dir)

# Mobility matrix (from physics module)
with torch.no_grad():
    dummy_input = torch.zeros(1, args.window, Data.m)
    if args.cuda:
        dummy_input = dummy_input.cuda()
    _, _, _, Pi = _original_forward(dummy_input)
    mobility = torch.softmax(Pi, dim=1).cpu().numpy()

PlotEachMatrix(mobility[None, ...], "Learned Mobility Matrix", "Mobility", save_dir)

# Training curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss", color="brown")
plt.plot(val_losses, label="Val RSE", color="navy")
plt.xlabel("Epoch")
plt.ylabel("Loss / RSE")
plt.title("Training Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{save_dir}training_curve.pdf", bbox_inches='tight', transparent=True)
plt.show()

print(f"\nAll done! Model: {args.save_dir}/{args.save_name}.pt")
print(f"Figures saved in: {save_dir}")
print("Your SEIR-PINN model is now fully trained and ready for thesis results!")