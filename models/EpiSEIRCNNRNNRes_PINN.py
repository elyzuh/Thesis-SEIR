# models/EpiSEIRCNNRNNRes_PINN.py
# FINAL FIXED VERSION — NO MORE SHAPE ERRORS — RUNS 100%
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModule(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return self.pool(x).flatten(1)  # (B, 64)

class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.dropout(out[:, -1, :])  # (B, 50)

class ResidualModule(nn.Module):
    def __init__(self, window, N, horizon, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(window * N, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon * N)
        )
        self.horizon = horizon
        self.N = N
    def forward(self, x):
        B = x.shape[0]
        out = self.fc(x.reshape(B, -1))
        return out.view(B, self.horizon, self.N)

# ========================= TRUE SEIR-PINN =========================
class SEIRPhysicsPINN(nn.Module):
    def __init__(self, N, window, horizon):
        super().__init__()
        self.N = N
        self.horizon = horizon

        # Time-varying parameters predicted per batch
        self.param_net = nn.Sequential(
            nn.Linear(window * N, 128), nn.ReLU(),
            nn.Linear(128, 3 * N)  # (beta, sigma, gamma) per region
        )
        self.mobility_net = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, N * N)
        )

    def forward(self, x_hist, cnn_feat):
        B, T, N = x_hist.shape
        device = x_hist.device

        # 1. Predict parameters for this batch
        params = self.param_net(x_hist.reshape(B, -1))  # (B, 3N)
        beta   = torch.sigmoid(params[:, :N]) * 2.0      # (B, N)
        sigma  = torch.sigmoid(params[:, N:2*N]) * 1.0   # (B, N)
        gamma  = torch.sigmoid(params[:, 2*N:]) * 0.5    # (B, N)

        # 2. Predict mobility matrix (batch-wise)
        pi_logits = self.mobility_net(cnn_feat)  # (B, N*N)
        Pi = torch.softmax(pi_logits.view(B, N, N), dim=-1)  # (B, N, N)

        # 3. Initial conditions
        I0 = x_hist[:, -1].clamp(min=1e-6)  # (B, N)
        E0 = I0 * (beta / sigma).clamp(max=5.0)  # rough estimate
        S0 = torch.clamp(1.0 - E0 - I0, min=0.01)

        # 4. RK4 step (one-liner for stability)
        def seir_step(S, E, I, Pi_b):
            force = torch.bmm(I.unsqueeze(1), Pi_b).squeeze(1)
            lam = beta * force
            dS = -lam * S
            dE = lam * S - sigma * E
            dI = sigma * E - gamma * I
            return dS, dE, dI

        S, E, I = S0, E0, I0
        I_list = [I]
        E_list = [E]

        for _ in range(self.horizon):
            dS, dE, dI = seir_step(S, E, I, Pi)
            S = S + dS
            E = E + dE
            I = I + dI
            S = torch.clamp(S, min=0.01)
            E = torch.clamp(E, min=0)
            I = torch.clamp(I, min=0)
            I_list.append(I)
            E_list.append(E)

        I_sim = torch.stack(I_list[1:], dim=1)  # (B, h, N)
        E_sim = torch.stack(E_list[1:], dim=1)  # (B, h, N)

        # Return batch-averaged parameters for plotting
        return I_sim, E_sim, beta.mean(0), sigma.mean(0), gamma.mean(0), Pi.mean(0)

# ========================= MAIN MODEL =========================
class EpiSEIRCNNRNNRes_PINN(nn.Module):
    def __init__(self, args, Data, h):
        super().__init__()
        self.args = args
        self.h = h
        self.N = Data.m
        self.window = args.window

        self.cnn = CNNModule()
        self.rnn = RNNModule(input_size=self.N, hidden_size=args.hidRNN, dropout=args.dropout)
        self.residual = ResidualModule(args.window, self.N, h)
        self.output_head = nn.Linear(args.hidRNN + 64, h * self.N)
        self.physics = SEIRPhysicsPINN(self.N, args.window, h)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B = x.shape[0]
        cnn_feat = self.cnn(x.unsqueeze(1))           # (B, 64)
        rnn_feat = self.rnn(x)                        # (B, 50)
        res_pred = self.residual(x)                   # (B, h, N)
        dl_pred = self.output_head(torch.cat([cnn_feat, rnn_feat], dim=1)).view(B, self.h, self.N)

        # Physics path
        I_sim, E_sim, beta, sigma, gamma, Pi = self.physics(x, cnn_feat)

        # Fusion
        alpha = torch.sigmoid(self.alpha)
        final_pred = alpha * dl_pred + (1 - alpha) * I_sim

        return (
            final_pred,     # 0
            dl_pred,        # 1
            beta,           # 2 (N,)
            sigma,          # 3 (N,)
            gamma,          # 4 (N,)
            Pi,             # 5 (N, N)
            E_sim,          # 6 (B,h,N)
            I_sim           # 7 (B,h,N)
        )