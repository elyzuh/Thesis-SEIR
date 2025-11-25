# models/EpiSEIRCNNRNNRes_PINN.py
# Modified from Liu et al. (2023) CIKM
# Now: SEIR + PINN + NGM + Multi-patch Mobility
# Only this file is changed — everything else stays the same

import torch
import torch.nn as nn
import torch.nn.functional as F


# === Liu's CNN Module (Spatial) - KEEP ===
class CNNModule(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# === Liu's RNN Module (Temporal) - KEEP ===
class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x: (B, w, N) → flatten spatial
        x_flat = x.view(x.size(0), x.size(1), -1)
        _, hn = self.gru(x_flat)
        return hn.squeeze(0)  # (B, hidden)


# === Liu's Residual Module - KEEP ===
class ResidualModule(nn.Module):
    def __init__(self, w, N, h):
        super().__init__()
        self.fc = nn.Linear(w * N, h * N)

    def forward(self, x):
        return self.fc(x.flatten(1)).view(-1, h, N)


# === YOUR SEIR-PINN Physics Module ===
class SEIRPhysicsPINN(nn.Module):
    def __init__(self, N, device='cuda'):
        super().__init__()
        self.N = N
        self.device = device

        # Learnable parameters: β, σ, γ, mobility logits
        self.beta = nn.Parameter(torch.rand(N) * 0.3 + 0.1)
        self.sigma = nn.Parameter(torch.ones(N) * 0.2)   # 1/latency ~5 days
        self.gamma = nn.Parameter(torch.ones(N) * 0.1)   # 1/recovery ~10 days
        self.pi_logits = nn.Parameter(torch.randn(N, N) * 0.1)

    def forward(self, I_hist):
        B, w, N = I_hist.shape
        device = self.device

        # Mobility matrix (row-stochastic)
        Pi = torch.softmax(self.pi_logits, dim=1)  # (N, N)
        W = torch.diag(Pi.sum(1))
        A = W - Pi.t()  # net outflow

        # Initialize latent E
        E = torch.zeros_like(I_hist)
        E_traj = [E[:, 0]]
        I_traj = [I_hist[:, 0]]

        for t in range(1, w):
            Et = E_traj[-1]
            It = I_traj[-1]
            St = torch.ones_like(It)  # S/N ≈ 1

            # Cross-location force of infection
            lambda_t = torch.mm(It, Pi.t())  # (B, N)

            # SEIR ODEs (Euler step)
            dE_dt = self.beta * St * lambda_t - self.sigma * Et
            dI_dt = self.sigma * Et - self.gamma * It

            E_next = Et + dE_dt
            I_next = It + dI_dt

            E_traj.append(E_next)
            I_traj.append(I_next)

        E_sim = torch.stack(E_traj, dim=1)
        I_sim = torch.stack(I_traj, dim=1)
        return E_sim, I_sim, Pi


# === NGM for SEIR (2N x 2N) ===
def compute_seir_ngm(beta, sigma, gamma, Pi):
    N = beta.shape[0]
    I = torch.eye(N, device=beta.device)
    zeros = torch.zeros(N, N, device=beta.device)

    F = torch.cat([zeros, torch.diag(beta) @ Pi], dim=1)
    V11 = torch.diag(sigma)
    V22 = torch.diag(gamma) + (Pi.sum(1).diag() - Pi.t())
    V = torch.cat([torch.cat([V11, zeros], dim=1),
                   torch.cat([-V11, V22], dim=1)], dim=0)

    K = F @ torch.inverse(V)
    return K


# === FULL MODEL: Epi-SEIR-CNNRNN-Res-PINN ===
class EpiSEIRCNNRNNRes_PINN(nn.Module):
    def __init__(self, N, w, h, device='cuda'):
        super().__init__()
        self.N, self.w, self.h = N, w, h
        self.device = device

        # Liu's modules
        self.cnn = CNNModule()
        self.rnn = RNNModule(N)
        self.residual = ResidualModule(w, N, h)

        # Output head
        self.mlp = nn.Linear(w * N + 64, h * N)

        # Your physics
        self.physics = SEIRPhysicsPINN(N, device)

    def forward(self, x):
        B = x.shape[0]

        # === Data-driven path (Liu's) ===
        x_cnn = x.unsqueeze(1)  # (B,1,w,N)
        h_cnn = self.cnn(x_cnn).mean(dim=2).flatten(1)  # (B, 64*N)
        h_rnn = self.rnn(x)  # (B, 64)
        res = self.residual(x)  # (B, h, N)

        mlp_in = torch.cat([h_cnn, h_rnn], dim=1)
        I_pred_dl = self.mlp(mlp_in).view(B, self.h, self.N)

        # === Physics path (Yours) ===
        E_sim, I_sim, Pi = self.physics(x)
        I_pred_pinn = I_sim[:, -self.h:]  # last h steps

        # For loss: return both
        return I_pred_dl, I_pred_pinn, E_sim, Pi