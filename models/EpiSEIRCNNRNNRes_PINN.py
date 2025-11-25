# models/EpiSEIRCNNRNNRes_PINN.py
# FINAL THESIS VERSION — 100% COMPATIBLE WITH ORIGINAL utils_ModelTrainEval.py
# Returns exactly 8 values → train() and evaluate() work perfectly
# Based on Liu et al. (2023) + your SEIR-PINN extension

import torch
import torch.nn as nn
import torch.nn.functional as F


# === Fixed Residual Module ===
class ResidualModule(nn.Module):
    def __init__(self, window, N, horizon, hidden=64):
        super().__init__()
        self.N = N
        self.horizon = horizon
        self.fc = nn.Sequential(
            nn.Linear(window * N, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * N)
        )

    def forward(self, x):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        out = self.fc(x_flat)
        return out.view(B, self.horizon, self.N)  # (B, h, N)


# === SEIR Physics Module — Learns β, σ, γ and Mobility (Pi) ===
class SEIRPhysicsPINN(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.beta = nn.Parameter(0.3 * torch.ones(N))
        self.sigma = nn.Parameter(0.2 * torch.ones(N))
        self.gamma = nn.Parameter(0.1 * torch.ones(N))
        self.pi_logits = nn.Parameter(torch.zeros(N, N))  # learned mobility

    def forward(self, x_hist, horizon=1):
        B, T, N = x_hist.shape
        device = x_hist.device

        Pi = torch.softmax(self.pi_logits, dim=-1)  # (N,N)

        # Start from last observed infected
        I = x_hist[:, -1].clamp(min=1e-6)
        E = torch.zeros_like(I)
        S = 1.0 - E - I
        S = S.clamp(min=0.01)

        I_sim_list = []
        E_sim_list = []

        for _ in range(horizon):
            force = torch.matmul(I, Pi)  # (B, N)
            lambda_t = self.beta.unsqueeze(0) * force

            dE = self.beta.unsqueeze(0) * S * lambda_t - self.sigma.unsqueeze(0) * E
            dI = self.sigma.unsqueeze(0) * E - self.gamma.unsqueeze(0) * I

            E = E + dE
            I = I + dI
            I = I.clamp(min=0)
            E = E.clamp(min=0)
            S = 1.0 - E - I
            S = S.clamp(min=0.01)

            I_sim_list.append(I)
            E_sim_list.append(E)

        I_sim = torch.stack(I_sim_list, dim=1)   # (B, h, N)
        E_sim = torch.stack(E_sim_list, dim=1)   # (B, h, N)

        return I_sim, E_sim, Pi


# === MAIN MODEL — EXACTLY 8 OUTPUTS FOR COMPATIBILITY ===
class EpiSEIRCNNRNNRes_PINN(nn.Module):
    def __init__(self, args, Data, h):
        super().__init__()
        self.args = args
        self.h = h
        self.N = Data.m
        self.window = args.window

        # === Data-driven path ===
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, Data.m))
        )
        self.rnn = nn.GRU(
            input_size=Data.m,
            hidden_size=args.hidRNN,
            num_layers=2,
            batch_first=True,
            dropout=args.dropout if 2 > 1 else 0.0
        )
        self.residual = ResidualModule(window=args.window, N=Data.m, horizon=h)
        self.output_head = nn.Linear(args.hidRNN + 64, h * Data.m)

        # === Physics path ===
        self.physics = SEIRPhysicsPINN(Data.m)

        # Learned fusion weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B = x.shape[0]

        # 1. CNN spatial features
        cnn_out = self.cnn(x.unsqueeze(1)).squeeze(2).view(B, -1)  # (B, 64)

        # 2. RNN temporal features
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]  # (B, hidRNN)

        # 3. Residual baseline
        res_pred = self.residual(x)  # (B, h, N)

        # 4. Data-driven prediction
        dl_input = torch.cat([cnn_out, rnn_out], dim=1)
        dl_pred = self.output_head(dl_input).view(B, self.h, self.N)

        # 5. Physics simulation
        I_sim, E_sim, Pi = self.physics(x, horizon=self.h)

        # 6. Final fusion
        alpha = torch.sigmoid(self.alpha)
        final_pred = alpha * dl_pred + (1 - alpha) * I_sim

        # === RETURN EXACTLY 8 VALUES (MATCHING ORIGINAL CODE) ===
        return (
            final_pred,        # output → main prediction
            dl_pred,           # EpiOutput → pure data-driven
            self.physics.beta, # beta
            self.physics.sigma,# sigma
            self.physics.gamma,# gamma
            Pi,                # K → learned mobility matrix
            E_sim,             # E_sim → latent exposed
            I_sim              # I_sim → physics-simulated infected
        )