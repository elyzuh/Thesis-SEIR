# models/EpiSEIRCNNRNNRes_PINN.py
# FINAL FIXED VERSION — Works with your current main.py + utils.py
# Fixes: RNN input size, constructor args, forward shape, physics integration

import torch
import torch.nn as nn
import torch.nn.functional as F


# === Fixed RNN Module (Temporal) ===
class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super(RNNModule, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,      # ← N (number of locations)
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
        return out[:, -1, :]   # (B, hidden_size)


# === Fixed CNN Module (Spatial) ===
class CNNModule(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, 1, window, N)
        x = F.relu(self.bn(self.conv(x)))
        return x.mean(dim=2)  # Global avg pool over time → (B, C, N)


# === Residual Module (kept from Liu et al.) ===
class ResidualModule(nn.Module):
    def __init__(self, window, N, horizon, hidden=64):
        super().__init__()
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
        return out.view(B, -1, N)  # (B, horizon, N)


# === SEIR Physics-Informed Module (Yours — fixed) ===
class SEIRPhysicsPINN(nn.Module):
    def __init__(self, N, device='cuda'):
        super().__init__()
        self.N = N
        self.device = device

        self.beta = nn.Parameter(0.2 * torch.ones(N))      # transmission rate
        self.sigma = nn.Parameter(0.2 * torch.ones(N))     # 1/latent period
        self.gamma = nn.Parameter(0.1 * torch.ones(N))     # recovery rate
        self.pi_logits = nn.Parameter(0.1 * torch.randn(N, N))

    def forward(self, x_hist):
        # x_hist: (B, window, N) → observed new infections
        B, T, N = x_hist.shape
        device = x_hist.device

        Pi = torch.softmax(self.pi_logits, dim=-1)  # (N,N) row-stochastic mobility

        # Initialize trajectories
        S = torch.ones(B, N, device=device) * 0.99
        E = torch.zeros(B, N, device=device)
        I = x_hist[:, 0].clamp(min=0)

        I_sim_list = [I]

        for t in range(1, T):
            lambda_t = torch.matmul(I, Pi)  # force of infection (B, N)
            dE = self.beta * S * lambda_t - self.sigma * E
            dI = self.sigma * E - self.gamma * I

            E = E + dE
            I = I + dI
            I = I.clamp(min=0)

            S = 1.0 - (E + I)
            S = S.clamp(min=0.01)

            I_sim_list.append(I)

        I_sim = torch.stack(I_sim_list, dim=1)  # (B, T, N)
        return I_sim, Pi


# === NGM Computation (for R0 loss) ===
def compute_seir_ngm(beta, sigma, gamma, Pi):
    N = beta.shape[0]
    device = beta.device
    I = torch.eye(N, device=device)
    Z = torch.zeros(N, N, device=device)

    beta_diag = torch.diag(beta)
    sigma_diag = torch.diag(sigma)
    gamma_diag = torch.diag(gamma)

    F = torch.cat([Z, beta_diag @ Pi], dim=1)  # (N, 2N)
    V = torch.cat([
        torch.cat([sigma_diag, Z], dim=1),
        torch.cat([-sigma_diag, gamma_diag], dim=1)
    ], dim=0)  # (2N, 2N)

    try:
        K = F @ torch.inverse(V)
        R0 = torch.svd(K)[1][0]  # spectral radius
    except:
        R0 = torch.tensor(1.0, device=device)
    return R0


# === FINAL MODEL — Compatible with main.py ===
class EpiSEIRCNNRNNRes_PINN(nn.Module):
    def __init__(self, args, Data, h):
        super(EpiSEIRCNNRNNRes_PINN, self).__init__()
        self.args = args
        self.h = h
        self.N = Data.m          # number of locations (9)
        self.window = args.window

        # === Data-driven modules ===
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

        # Output head
        self.output_head = nn.Linear(args.hidRNN + 64, h * self.N)

        # === Physics module ===
        self.physics = SEIRPhysicsPINN(self.N, device='cuda' if args.cuda else 'cpu')

    def forward(self, x):
        # x: (B, window, N)
        B = x.shape[0]

        # 1. CNN path
        x_cnn = x.unsqueeze(1)           # (B, 1, w, N)
        cnn_out = self.cnn(x_cnn)         # (B, 64, N)
        cnn_feat = cnn_out.mean(dim=-1)   # (B, 64)

        # 2. RNN path
        rnn_out = self.rnn(x)             # (B, hidRNN)

        # 3. Residual path
        res_out = self.residual(x)        # (B, h, N)

        # 4. Combine
        combined = torch.cat([cnn_feat, rnn_out], dim=1)
        dl_pred = self.output_head(combined).view(B, self.h, self.N)

        # 5. Physics simulation
        I_sim, Pi = self.physics(x)       # I_sim: (B, window, N)
        physics_pred = I_sim[:, -self.h:, :] if I_sim.size(1) >= self.h else I_sim[:, -1:, :].expand(-1, self.h, -1)

        # Final prediction: weighted fusion
        final_pred = 0.7 * dl_pred + 0.3 * physics_pred

        return final_pred, dl_pred, physics_pred, Pi