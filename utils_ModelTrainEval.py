# utils_ModelTrainEval.py
# Modified for SEIR-PINN-NGM (your thesis)
# Compatible with Liu et al. (2023) CIKM framework
import torch
import torch.nn as nn
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np


# ----------------------------------------------------------------------
# 1. EVALUATE (unchanged except model output unpacking)
# ----------------------------------------------------------------------
def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName,
             lambda_pde=None, lambda_ngm=None):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    counter = 0

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]

        if modelName == "EpiSEIRCNNRNNRes_PINN":
            output, EpiOutput, beta, sigma, gamma, K, E_sim, I_sim = model(X)
        elif modelName == "CNNRNN_Res_epi":
            output, EpiOutput, beta, gamma, NGMT = model(X)
        else:
            output = model(X)

        if predict is None:
            predict = output.cpu()
            test = Y.cpu()
        else:
            predict = torch.cat((predict, output.cpu()))
            test = torch.cat((test, Y.cpu()))

        scale = loader.scale.expand(output.size(0), loader.m)
        counter += 1

        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * loader.m)

    rse = math.sqrt(total_loss / n_samples)
    rae = total_loss_l1 / n_samples

    predict = predict.data.numpy()
    Ytest = test.data.numpy()

    # Correlation
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = correlation[index].mean()

    return rse, rae, correlation


# ----------------------------------------------------------------------
# 2. TRAIN (uses new PINN loss for SEIR-PINN)
# ----------------------------------------------------------------------
def train(loader, data, model, criterion, optim, batch_size, modelName,
          Lambda, lambda_pde=1.0, lambda_ngm=1.0):
    model.train()
    total_loss = 0
    n_samples = 0
    counter = 0

    for inputs in loader.get_batches(data, batch_size, True):
        counter += 1
        X, Y = inputs[0], inputs[1]
        model.zero_grad()

        if modelName == "EpiSEIRCNNRNNRes_PINN":
            output, EpiOutput, beta, sigma, gamma, K, E_sim, I_sim = model(X)
            loss = pinn_seir_loss(
                output, EpiOutput, Y, E_sim, I_sim,
                beta, sigma, gamma, model, loader,
                Lambda, lambda_pde, lambda_ngm
            )
        elif modelName == "CNNRNN_Res_epi":
            output, EpiOutput, beta, gamma, NGMT = model(X)
            scale = loader.scale.expand(output.size(0), loader.m)
            loss = criterion(output * scale, Y * scale) + Lambda * criterion(EpiOutput * scale, Y * scale)
        else:
            output = model(X)
            scale = loader.scale.expand(output.size(0), loader.m)
            loss = criterion(output * scale, Y * scale)

        loss.backward()
        optim.step()

        total_loss += loss.item()
        n_samples += (output.size(0) * loader.m)

    return total_loss / n_samples


# ----------------------------------------------------------------------
# 3. PINN-SEIR LOSS (YOUR THESIS CORE)
# ----------------------------------------------------------------------
def pinn_seir_loss(pred_dl, pred_epi, Y_true, E_sim, I_sim,
                   beta, sigma, gamma, model, loader,
                   Lambda, lambda_pde, lambda_ngm):
    """
    Full SEIR-PINN loss:
        L = L_data + Lambda * L_epi + lambda_pde * L_pde + lambda_ngm * L_ngm
    """
    scale = loader.scale.expand(pred_dl.size(0), loader.m).to(pred_dl.device)

    # 1. Data loss (main prediction)
    loss_data = nn.MSELoss()(pred_dl * scale, Y_true * scale)

    # 2. Epidemiological NGM loss (one-step consistency)
    loss_epi = nn.MSELoss()(pred_epi * scale, Y_true[:, 0] * scale)  # only first horizon

    # 3. PDE residual loss (Euler step residuals)
    loss_pde = 0.0
    P = model.P  # window size
    m = model.m  # num locations
    b = pred_dl.size(0)

    # Reconstruct Pi from mask_mat (same as in model)
    Pi = torch.softmax(model.mask_mat, dim=1)  # (m, m)
    Pi = Pi.unsqueeze(0).repeat(b, 1, 1)  # (b, m, m)

    for t in range(1, P):
        Et, It = E_sim[:, t-1], I_sim[:, t-1]
        Et_next, It_next = E_sim[:, t], I_sim[:, t]

        St = torch.ones_like(It)  # S/N ≈ 1
        lambda_t = torch.bmm(It.unsqueeze(1), Pi).squeeze(1)  # (b, m)

        dE_pred = beta * St * lambda_t - sigma * Et
        dI_pred = sigma * Et - gamma * It

        loss_pde += nn.MSELoss()(Et_next - Et, dE_pred)
        loss_pde += nn.MSELoss()(It_next - It, dI_pred)

    # 4. NGM consistency (optional, already in EpiOutput)
    loss_ngm = loss_epi  # reuse

    total = loss_data + Lambda * loss_epi + lambda_pde * loss_pde + lambda_ngm * loss_ngm
    return total


# ----------------------------------------------------------------------
# 4. GET PREDICTION (now returns E_sim and sigma)
# ----------------------------------------------------------------------
def GetPrediction(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName):
    model.eval()
    Y_predict = Y_true = X_true = None
    BetaList = SigmaList = GammaList = NGMList = EList = None
    counter = 0

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]

        if modelName == "EpiSEIRCNNRNNRes_PINN":
            output, EpiOutput, beta, sigma, gamma, K, E_sim, I_sim = model(X)
        elif modelName == "CNNRNN_Res_epi":
            output, EpiOutput, beta, gamma, NGMT = model(X)
        else:
            output = model(X)

        counter += 1

        if Y_predict is None:
            Y_predict = output.cpu()
            Y_true = Y.cpu()
            X_true = X.cpu()
            if modelName in ["CNNRNN_Res_epi", "EpiSEIRCNNRNNRes_PINN"]:
                BetaList = beta.cpu()
                GammaList = gamma.cpu()
                NGMList = (K if modelName == "EpiSEIRCNNRNNRes_PINN" else NGMT).cpu()
                if modelName == "EpiSEIRCNNRNNRes_PINN":
                    SigmaList = sigma.cpu()
                    EList = E_sim.cpu()
        else:
            Y_predict = torch.cat((Y_predict, output.cpu()))
            Y_true = torch.cat((Y_true, Y.cpu()))
            X_true = torch.cat((X_true, X.cpu()))
            if modelName in ["CNNRNN_Res_epi", "EpiSEIRCNNRNNRes_PINN"]:
                BetaList = torch.cat((BetaList, beta.cpu()))
                GammaList = torch.cat((GammaList, gamma.cpu()))
                NGMList = torch.cat((NGMList, (K if modelName == "EpiSEIRCNNRNNRes_PINN" else NGMT).cpu()))
                if modelName == "EpiSEIRCNNRNNRes_PINN":
                    SigmaList = torch.cat((SigmaList, sigma.cpu()))
                    EList = torch.cat((EList, E_sim.cpu()))

    scale = loader.scale
    Y_predict = (Y_predict * scale).detach().numpy()
    Y_true = (Y_true * scale).detach().numpy()
    X_true = (X_true * scale).detach().numpy()

    if modelName == "EpiSEIRCNNRNNRes_PINN":
        BetaList = BetaList.detach().numpy()
        SigmaList = SigmaList.detach().numpy()
        GammaList = GammaList.detach().numpy()
        NGMList = NGMList.detach().numpy()
        EList = EList.detach().numpy()
        return X_true, Y_predict, Y_true, BetaList, SigmaList, GammaList, NGMList, EList
    elif modelName == "CNNRNN_Res_epi":
        BetaList = BetaList.detach().numpy()
        GammaList = GammaList.detach().numpy()
        NGMList = NGMList.detach().numpy()
        return X_true, Y_predict, Y_true, BetaList, GammaList, NGMList
    else:
        return X_true, Y_predict, Y_true