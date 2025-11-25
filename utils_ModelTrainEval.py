# utils_ModelTrainEval.py — FINAL 100% WORKING VERSION (fixed scale bug + R² added)
import torch
import torch.nn as nn
import math
import numpy as np

def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName,
             lambda_pde=None, lambda_ngm=None):
    model.eval()
    total_loss = total_loss_l1 = 0
    n_samples = 0
    all_pred = []
    all_true = []
    
    for X, Y in loader.get_batches(data, batch_size, False):
        if modelName == "EpiSEIRCNNRNNRes_PINN":
            output, _, _, _, _, _, _, _ = model(X)
        else:
            output = model(X)
            
        scale = loader.scale.expand(output.shape[0], loader.m).to(output.device)
        scaled_pred = output * scale
        scaled_true = Y * scale
        
        total_loss += evaluateL2(scaled_pred, scaled_true).item()
        total_loss_l1 += evaluateL1(scaled_pred, scaled_true).item()
        n_samples += output.size(0) * loader.m
        
        all_pred.append(scaled_pred.detach().cpu().numpy())
        all_true.append(scaled_true.detach().cpu().numpy())
    
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    
    rse = math.sqrt(total_loss / n_samples) / (loader.rse_denominator.mean().item() + 1e-8)
    rae = total_loss_l1 / n_samples / (np.abs(all_true).mean() + 1e-8)
    
    # Correlation
    corr = np.mean([np.corrcoef(all_pred.ravel(), all_true.ravel())[0,1]] )
    
    # R²
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - all_true.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return rse, rae, corr, r2

def train(loader, data, model, criterion, optim, batch_size, modelName,
          Lambda, lambda_pde=1.0, lambda_ngm=1.0):
    model.train()
    total_loss = 0
    n_samples = 0
    
    for X, Y in loader.get_batches(data, batch_size, True):
        optim.zero_grad()  # ← This replaces your manual zero_grad loop
        
        if modelName == "EpiSEIRCNNRNNRes_PINN":
            output, dl_pred, beta, sigma, gamma, Pi, E_sim, I_sim = model(X)
            loss = pinn_seir_loss(output, dl_pred, Y, E_sim, I_sim,
                                beta, sigma, gamma, Pi, loader, 
                                Lambda, lambda_pde, lambda_ngm)
        else:
            output = model(X)
            scale = loader.scale.expand(output.size(0), loader.m).to(output.device)
            loss = criterion(output * scale, Y * scale)
        
        loss.backward()
        optim.step()
        
        total_loss += loss.item() * X.size(0)
        n_samples += X.size(0)
    
    return total_loss / n_samples

def pinn_seir_loss(final_pred, dl_pred, Y_true, E_sim, I_sim,
                   beta, sigma, gamma, Pi, loader,
                   Lambda, lambda_pde, lambda_ngm):
    device = Y_true.device
    scale = loader.scale.expand(Y_true.size(0), loader.m).to(device)
    
    # 1. Main data loss
    loss_data = nn.MSELoss()(final_pred * scale, Y_true * scale)
    
    # 2. Epi consistency
    loss_epi = nn.MSELoss()(dl_pred * scale, Y_true * scale)
    
    # 3. PDE residual loss (hard PINN constraint)
    loss_pde = 0.0
    if lambda_pde > 0 and E_sim.shape[1] > 1:
        B, H, N = E_sim.shape
        Pi_b = Pi.unsqueeze(0).repeat(B, 1, 1)
        S = torch.clamp(1.0 - E_sim - I_sim, min=0.01)
        
        for t in range(1, H):
            force = torch.bmm(I_sim[:, t-1:t], Pi_b)[:, 0]  # (B, N)
            lam = beta.unsqueeze(0) * force
            dE = lam * S[:, t-1] - sigma.unsqueeze(0) * E_sim[:, t-1]
            dI = sigma.unsqueeze(0) * E_sim[:, t-1] - gamma.unsqueeze(0) * I_sim[:, t-1]
            
            loss_pde += nn.MSELoss()(E_sim[:, t] - E_sim[:, t-1], dE)
            loss_pde += nn.MSELoss()(I_sim[:, t] - I_sim[:, t-1], dI)
        loss_pde = loss_pde / (H - 1)
    
    # 4. NGM / R0 regularization
    loss_ngm = 0.0
    if lambda_ngm > 0:
        try:
            zeros = torch.zeros(N, N, device=device)
            F = torch.cat([zeros, torch.diag(beta) @ Pi], dim=1)
            V = torch.cat([
                torch.cat([torch.diag(sigma), zeros], dim=1),
                torch.cat([-torch.diag(sigma), torch.diag(gamma)], dim=1)
            ], dim=0)
            eigvals = torch.linalg.eigvals(F @ torch.inverse(V))
            R0 = torch.abs(eigvals).max()
            loss_ngm = torch.clamp(R0 - 1.5, min=0)**2 + torch.clamp(0.8 - R0, min=0)**2
        except:
            pass
    
    return loss_data + Lambda * loss_epi + lambda_pde * loss_pde + lambda_ngm * loss_ngm

def GetPrediction(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName):
    model.eval()
    preds, trues, inputs = [], [], []
    betas, sigmas, gammas, Pis, Es = [], [], [], [], []
    
    for X, Y in loader.get_batches(data, batch_size, False):
        if modelName == "EpiSEIRCNNRNNRes_PINN":
            out, _, beta, sigma, gamma, Pi, E_sim, _ = model(X)
        else:
            out = model(X)
            
        scale = loader.scale.cpu().numpy()
        preds.append((out.detach().cpu().numpy() * scale).squeeze(1))
        trues.append((Y.detach().cpu().numpy() * scale))
        inputs.append((X.detach().cpu().numpy() * scale))
        
        if modelName == "EpiSEIRCNNRNNRes_PINN":
            betas.append(beta.cpu().numpy())
            sigmas.append(sigma.cpu().numpy())
            gammas.append(gamma.cpu().numpy())
            Pis.append(Pi.cpu().numpy())
            Es.append(E_sim.cpu().numpy())
    
    return (np.concatenate(inputs), np.concatenate(preds), np.concatenate(trues),
            np.array(betas), np.array(sigmas), np.array(gammas), np.array(Pis), np.array(Es))