# models/EpiSEIRCNNRNNRes_PINN.py
# FINAL THESIS VERSION — SHAPE BUG FIXED + 8 OUTPUTS FOR COMPATIBILITY
# CNN output now (B, 64) + RNN (B, 50) = (B, 114) → matches Linear(114, 9)
# Based on Wu et al. (SIGIR 2018) CNN-RNN-Residual + Liu et al. (CIKM 2023) + your SEIR-PINN

import torch
import torch.nn as nn
import torch.nn.functional as F


# === CNN Module (Spatial) — From 2018 Paper ===
class CNNModule(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global avg to (B, 64, 1, 1)

    def forward(self, x):
        # x: (B, 1, window, N)
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)  # (B, 64, 1, 1)
        return x.view(x.size(0), -1)  # (B, 64) — FIXED SHAPE!


# === RNN Module (Temporal) — From 2018 Paper ===
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
        return out[:, -1, :]  # (B, hidden_size=50)


# === Residual Module — From 2018 Paper ===
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


# === SEIR Physics Module — Your Thesis Innovation ===
class SEIRPhysicsPINN(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        # Learnable epidemiological parameters
        self.beta = nn.Parameter(0.3 * torch.ones(N))   # Transmission rate
        self.sigma = nn.Parameter(0.2 * torch.ones(N))  # Incubation rate
        self.gamma = nn.Parameter(0.1 * torch.ones(N))  # Recovery rate
        # Learned mobility matrix logits
        self.pi_logits = nn.Parameter(torch.zeros(N, N))

    def forward(self, x_hist, horizon=1):
        B, T, N = x_hist.shape
        device = x_hist.device

        # Learned mobility matrix
        Pi = torch.softmax(self.pi_logits, dim=-1)  # (N, N)

        # Initial conditions from last observation
        I = x_hist[:, -1].clamp(min=1e-6)
        E = torch.zeros_like(I)
        S = torch.ones_like(I) * 0.98  # Assume mostly susceptible

        I_sim_list = []
        E_sim_list = []

        # Simulate forward for horizon steps
        for _ in range(horizon):
            # Force of infection with mobility
            force = torch.matmul(I, Pi.t())  # (B, N)
            lambda_t = self.beta.unsqueeze(0) * force

            # SEIR ODEs (Euler integration)
            dE = self.beta.unsqueeze(0) * S * lambda_t - self.sigma.unsqueeze(0) * E
            dI = self.sigma.unsqueeze(0) * E - self.gamma.unsqueeze(0) * I

            E = E + dE * 1.0  # dt=1 week
            I = I + dI * 1.0
            I = I.clamp(min=0)
            E = E.clamp(min=0)
            S = 1.0 - E - I
            S = S.clamp(min=0.01)

            I_sim_list.append(I)
            E_sim_list.append(E)

        I_sim = torch.stack(I_sim_list, dim=1)  # (B, h, N)
        E_sim = torch.stack(E_sim_list, dim=1)  # (B, h, N)

        return I_sim, E_sim, Pi


# === FINAL MODEL — CNN + RNN + Residual + SEIR-PINN Fusion ===
class EpiSEIRCNNRNNRes_PINN(nn.Module):
    def __init__(self, args, Data, h):
        super().__init__()
        self.args = args
        self.h = h
        self.N = Data.m
        self.window = args.window

        # CNN (spatial) — 2018 paper
        self.cnn = CNNModule(in_channels=1, out_channels=64)

        # RNN (temporal) — 2018 paper
        self.rnn = RNNModule(
            input_size=self.N,
            hidden_size=args.hidRNN,
            num_layers=2,
            dropout=args.dropout
        )

        # Residual links — 2018 paper
        self.residual = ResidualModule(
            window=args.window,
            N=self.N,
            horizon=h,
            hidden=64
        )

        # Data-driven fusion head
        self.output_head = nn.Linear(args.hidRNN + 64, h * self.N)  # 50 + 64 = 114 → 9

        # Physics module — your innovation
        self.physics = SEIRPhysicsPINN(self.N)

        # Learned fusion weight between DL and physics
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B = x.shape[0]

        # 1. CNN path (spatial features)
        cnn_feat = self.cnn(x.unsqueeze(1))  # (B, 64) — SHAPE FIXED!

        # 2. RNN path (temporal features)
        rnn_feat = self.rnn(x)  # (B, 50)

        # 3. Residual path (direct mapping)
        res_pred = self.residual(x)  # (B, h, N)

        # 4. Data-driven prediction (CNN + RNN fusion)
        dl_input = torch.cat([cnn_feat, rnn_feat], dim=1)  # (B, 114)
        dl_pred = self.output_head(dl_input).view(B, self.h, self.N)  # (B, 1, 9)

        # 5. Physics simulation (SEIR + learned mobility)
        I_sim, E_sim, Pi = self.physics(x, horizon=self.h)  # (B, 1, 9), (B, 1, 9), (9, 9)

        # 6. Learned fusion: DL + Physics
        alpha = torch.sigmoid(self.alpha)
        final_pred = alpha * dl_pred + (1 - alpha) * I_sim  # (B, 1, 9)

        # === RETURN EXACTLY 8 VALUES FOR train() COMPATIBILITY ===
        return (
            final_pred,           # 0: main fused prediction (B, h, N)
            dl_pred,              # 1: pure data-driven prediction (for epi loss)
            self.physics.beta,    # 2: learned transmission rate β  (N,)
            self.physics.sigma,   # 3: learned incubation rate σ  (N,)
            self.physics.gamma,   # 4: learned recovery rate γ    (N,)
            Pi,                   # 5: learned mobility matrix K  (N, N)
            E_sim,                # 6: simulated latent Exposed   (B, h, N)
            I_sim                 # 7: simulated Infected         (B, h, N)
        )


# NGM helper for loss (optional, from Liu et al. 2023)
def compute_seir_ngm(beta, sigma, gamma, Pi):
    N = beta.shape[0]
    device = beta.device
    eye = torch.eye(N, device=device)
    zeros = torch.zeros_like(eye)

    # Next-generation matrix for SEIR
    F = torch.cat([zeros, torch.diag(beta) @ Pi], dim=1)  # New infections
    V = torch.cat([
        torch.cat([torch.diag(sigma), zeros], dim=1),
        torch.cat([-torch.diag(sigma), torch.diag(gamma)], dim=1)
    ], dim=0)  # Transitions out

    K = F @ torch.inverse(V)
    R0 = torch.max(torch.abs(torch.linalg.eigvals(K)))
    return K, R0