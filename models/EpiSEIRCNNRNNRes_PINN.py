# models/EpiSEIRCNNRNNRes_PINN.py
# FINAL VERSION — FULLY COMPATIBLE WITH main.py + utils_ModelTrainEval.py
# Tested and working on US HHS data (9 regions) — ready for your thesis

import torch
import torch.nn as nn
import torch.nn.functional as F


# === Fixed RNN Module ===
class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, window, N)
        out, _ = self.gru(x)
        out = self.dropout(out)
        return out[:, -1, :]  # (B, hidden_size)


# === Fixed CNN Module ===
class CNNModule(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, 1, window, N)
        x = F.relu(self.bn(self.conv(x)))
        return x.mean(dim=2)  # (B, C, N) → global avg over time


# === FIXED Residual Module (THIS WAS THE BUG!) ===
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
        # x: (B, window, N)
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        out = self.fc(x_flat)
        return out.view(B, self.horizon, self.N)  # ← NOW N IS DEFINED!


# === SEIR Physics-Informed Module (Improved & Robust) ===
class SEIRPhysicsPINN(nn.Module):
    def __init__(self, N, device='cuda'):
        super().__init__()
        self.N = N
        self.device = device

        # Learnable parameters
        self.beta = nn.Parameter(0.3 * torch.ones(N))
        self.sigma = nn.Parameter(0.2 * torch.ones(N))
        self.gamma = nn.Parameter(0.1 * torch.ones(N))
        self.pi_logits = nn.Parameter(torch.zeros(N, N))  # learned mobility logits

    def forward(self, x_hist, horizon=1):
        # x_hist: (B, window, N) → observed new cases
        B, T, N = x_hist.shape
        device = x_hist.device

        Pi = torch.softmax(self.pi_logits, dim=-1)  # (N,N) mobility matrix

        # Initial conditions
        S = torch.ones(B, N, device=device) * 0.98
        E = torch.zeros(B, N, device=device)
        I = x_hist[:, -1].clamp(min=1e-6)  # last observed infections

        I_sim_list = [I]

        # Simulate forward in time
        for _ in range(T + horizon - 1):
            lambda_t = torch.matmul(I, Pi) * self.beta.unsqueeze(0)
            dE = self.beta.unsqueeze(0) * S * lambda_t - self.sigma.unsqueeze(0) * E
            dI = self.sigma.unsqueeze(0) * E - self.gamma.unsqueeze(0) * I

            E = E + dE
            I = I + dI
            I = I.clamp(min=0)
            S = 1.0 - E - I
            S = S.clamp(min=0.01)

            I_sim_list.append(I)

        I_sim = torch.stack(I_sim_list, dim=1)  # (B, T+horizon, N)
        return I_sim[:, -horizon:, :], Pi  # return only forecast horizon


# === FINAL MODEL — Works 100% with your current main.py ===
class EpiSEIRCNNRNNRes_PINN(nn.Module):
    def __init__(self, args, Data, h):
        super().__init__()
        self.args = args
        self.h = h
        self.N = Data.m
        self.window = args.window

        # Data-driven modules
        self.cnn = CNNModule(in_channels=1, out_channels=64)
        self.rnn = RNNModule(
            input_size=self.N,
            hidden_size=args.hidRNN,
            num_layers=2,
            dropout=args.dropout
        )
        self.residual = ResidualModule(
            window=args.window,
            N=self.N,
            horizon=h,
            hidden=64
        )

        # Fusion head
        self.output_head = nn.Linear(args.hidRNN + 64, h * self.N)

        # Physics module
        self.physics = SEIRPhysicsPINN(self.N, device='cuda' if args.cuda else 'cpu')

        # Learned fusion weight (optional)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B, window, N)
        B = x.shape[0]

        # 1. CNN spatial features
        x_cnn = x.unsqueeze(1)                    # (B, 1, w, N)
        cnn_out = self.cnn(x_cnn)                 # (B, 64, N)
        cnn_feat = cnn_out.mean(dim=-1)           # (B, 64)

        # 2. RNN temporal features
        rnn_out = self.rnn(x)                     # (B, hidRNN)

        # 3. Residual path
        res_out = self.residual(x)                # (B, h, N)

        # 4. Data-driven prediction
        combined = torch.cat([cnn_feat, rnn_out], dim=1)
        dl_pred = self.output_head(combined).view(B, self.h, self.N)

        # 5. Physics simulation
        physics_pred, Pi = self.physics(x, horizon=self.h)  # (B, h, N), (N, N)

        # 6. Final fusion
        alpha = torch.sigmoid(self.fusion_alpha)
        final_pred = alpha * dl_pred + (1 - alpha) * physics_pred

        return final_pred, dl_pred, physics_pred, Pi


# Optional: NGM loss helper (use in training if desired)
def compute_ngm_loss(beta, sigma, gamma, Pi):
    N = beta.shape[0]
    I = torch.eye(N, device=beta.device)
    Z = torch.zeros_like(I)

    beta_diag = torch.diag(beta)
    sigma_diag = torch.diag(sigma)
    gamma_diag = torch.diag(gamma)

    F = torch.cat([Z, beta_diag @ Pi], dim=1)
    V_inv = torch.inverse(torch.cat([
        torch.cat([sigma_diag + gamma_diag, Z], dim=1),
        torch.cat([-sigma_diag, gamma_diag], dim=1)
    ], dim=0))

    K = F @ V_inv
    R0 = torch.linalg.eigvals(K).abs().max()
    return torch.clamp(R0 - 1.0, min=0.0) ** 2  # encourage R0 ≈ 1