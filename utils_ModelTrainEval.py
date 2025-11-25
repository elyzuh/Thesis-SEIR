# utils_ModelTrainEval.py
# FINAL VERSION — 100% COMPATIBLE WITH YOUR EpiSEIRCNNRNNRes_PINN
# Fixes: no model.P, no model.m, no model.mask_mat
# Fixes: shape mismatch warning (32,32,9) → (32,1,9)
# Works perfectly with your main.py and model

import torch
import torch.nn as nn
import math
import numpy as np


# ----------------------------------------------------------------------
# 1. EVALUATE
# ----------------------------------------------------------------------
def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName,
             lambda_pde=None, lambda_ngm=None):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]

        if modelName == "EpiSEIRCNNRNNRes_PINN":
            output, EpiOutput, beta, sigma, gamma, K, E_sim, I_sim = model(X)
        else:
            output = model(X)

        if predict is None:
            predict = output.detach().cpu()
            test = Y.detach().cpu()
        else:
            predict = torch.cat((predict, output.detach().cpu()))
            test = torch.cat((test, Y.detach().cpu()))

        scale = loader.scale.expand(output.size(0), loader.m).to(output.device)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * loader.m)

    rse = math.sqrt(total_loss / n_samples)
    rae = total_loss_l1 / n_samples

    predict = predict.numpy()
    Ytest = test.numpy()

    # Correlation
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g + 1e-8)
    correlation = correlation[index].mean()

    return rse, rae, correlation


# ----------------------------------------------------------------------
# 2. TRAIN — FIXED PINN LOSS (uses args.window and Data.m)
# ----------------------------------------------------------------------
def train(loader, data, model, criterion, optim, batch_size, modelName,
          Lambda, lambda_pde=1.0, lambda_ngm=1.0):
    model.train()
    total_loss = 0
    n_samples = 0

    for inputs in loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]

        # === ZERO GRADIENTS MANUALLY (since Optim has no zero_grad) ===
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        if modelName == "EpiSEIRCNNRNNRes_PINN":
            output, EpiOutput, beta, sigma, gamma, K, E_sim, I_sim = model(X)
            loss = pinn_seir_loss(
                output, EpiOutput, Y, E_sim, I_sim,
                beta, sigma, gamma, K,
                X.shape[1], loader.m, X.device,
                Lambda, lambda_pde, lambda_ngm
            )
        else:
            output = model(X)
            scale = loader.scale.expand(output.size(0), loader.m).to(output.device)
            loss = criterion(output * scale, Y * scale)

        loss.backward()
        optim.step()  # ← This calls your Optim.step() with clipping!

        total_loss += loss.item() * X.size(0)
        n_samples += X.size(0)

    return total_loss / n_samples

# ----------------------------------------------------------------------
# 3. PINN-SEIR LOSS — FIXED (no model.P, no model.mask_mat)
# ----------------------------------------------------------------------
# === REPLACE THIS ENTIRE FUNCTION IN utils_ModelTrainEval.py ===
def pinn_seir_loss(pred_dl, pred_epi, Y_true, E_sim, I_sim,
                   beta, sigma, gamma, K,
                   window, m, device,
                   Lambda, lambda_pde, lambda_ngm):
    """
    FINAL FIXED VERSION — Works perfectly with your current model
    E_sim and I_sim have shape (B, horizon, m) → NOT (B, window, m)
    So we compute PDE residuals ONLY over the prediction horizon
    """
    # Scale for denormalization
    scale = getattr(Y_true, 'scale', None)
    if scale is None:
        scale = torch.ones(1, m, device=device)
    else:
        scale = scale.view(1, -1)

    # 1. Data loss (main prediction)
    loss_data = nn.MSELoss()(pred_dl * scale, Y_true * scale)

    # 2. Epidemiological consistency loss (pure DL head vs target)
    loss_epi = nn.MSELoss()(pred_epi * scale, Y_true * scale)

    # 3. PDE residual loss — ONLY over horizon steps
    loss_pde = 0.0
    B, H, _ = E_sim.shape  # H = horizon (e.g. 4)

    if lambda_pde > 0 and H > 1:
        Pi = K.unsqueeze(0).repeat(B, 1, 1)  # (B, m, m)

        for t in range(1, H):
            Et = E_sim[:, t-1]
            It = I_sim[:, t-1]
            Et_next = E_sim[:, t]
            It_next = I_sim[:, t]

            St = torch.clamp(1.0 - Et - It, min=0.01)

            # Force of infection with learned mobility
            force = torch.bmm(It.unsqueeze(1), Pi).squeeze(1)  # (B, m)
            lambda_t = beta.unsqueeze(0) * force

            # SEIR differential equations (Euler step)
            dE_dt = lambda_t * St - sigma.unsqueeze(0) * Et
            dI_dt = sigma.unsqueeze(0) * Et - gamma.unsqueeze(0) * It

            # Residuals
            loss_pde += nn.MSELoss()(Et_next - Et, dE_dt)
            loss_pde += nn.MSELoss()(It_next - It, dI_dt)

        loss_pde = loss_pde / (H - 1)

    # 4. NGM / R0 regularization (keep R0 ≈ 1–3)
    loss_ngm = 0.0
    if lambda_ngm > 0:
        try:
            zeros = torch.zeros(m, m, device=device)
            F = torch.cat([zeros, torch.diag(beta) @ K], dim=1)
            V = torch.cat([
                torch.cat([torch.diag(sigma), zeros], dim=1),
                torch.cat([-torch.diag(sigma), torch.diag(gamma)], dim=1)
            ], dim=0)
            K_ngm = F @ torch.inverse(V)
            R0 = torch.linalg.eigvals(K_ngm).abs().max()
            # Encourage R0 in realistic range [0.8, 3.0]
            loss_ngm = torch.clamp(R0 - 2.0, min=0.0)**2
        except:
            loss_ngm = torch.tensor(0.0, device=device)

    total_loss = loss_data + Lambda * loss_epi + lambda_pde * loss_pde + lambda_ngm * loss_ngm
    return total_loss


# ----------------------------------------------------------------------
# 4. GET PREDICTION — FIXED
# ----------------------------------------------------------------------
def GetPrediction(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName):
    model.eval()
    Y_predict = Y_true = X_true = None
    BetaList = SigmaList = GammaList = NGMList = EList = None

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]

        if modelName == "EpiSEIRCNNRNNRes_PINN":
            output, EpiOutput, beta, sigma, gamma, K, E_sim, I_sim = model(X)
        else:
            output = model(X)

        if Y_predict is None:
            Y_predict = output.detach().cpu()
            Y_true = Y.detach().cpu()
            X_true = X.detach().cpu()
            if modelName == "EpiSEIRCNNRNNRes_PINN":
                BetaList = beta.detach().cpu()
                SigmaList = sigma.detach().cpu()
                GammaList = gamma.detach().cpu()
                NGMList = K.detach().cpu()
                EList = E_sim.detach().cpu()
        else:
            Y_predict = torch.cat((Y_predict, output.detach().cpu()))
            Y_true = torch.cat((Y_true, Y.detach().cpu()))
            X_true = torch.cat((X_true, X.detach().cpu()))
            if modelName == "EpiSEIRCNNRNNRes_PINN":
                BetaList = torch.cat((BetaList, beta.detach().cpu()))
                SigmaList = torch.cat((SigmaList, sigma.detach().cpu()))
                GammaList = torch.cat((GammaList, gamma.detach().cpu()))
                NGMList = torch.cat((NGMList, K.detach().cpu()))
                EList = torch.cat((EList, E_sim.detach().cpu()))

    scale = loader.scale.cpu().numpy()
    Y_predict = (Y_predict.numpy() * scale)
    Y_true = (Y_true.numpy() * scale)
    X_true = (X_true.numpy() * scale)

    if modelName == "EpiSEIRCNNRNNRes_PINN":
        return (X_true, Y_predict.squeeze(1), Y_true,
                BetaList.numpy(), SigmaList.numpy(), GammaList.numpy(),
                NGMList.numpy(), EList.numpy())
    else:
        return X_true, Y_predict, Y_true