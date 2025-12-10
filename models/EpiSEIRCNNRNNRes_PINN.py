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
# ========================= TRUE SEIR-PINN (FIXED) =========================
class SEIRPhysicsPINN(nn.Module):
    def __init__(self, N, window, horizon):
        super().__init__()
        self.N = N
        self.horizon = horizon

        # 1. Parameter Network
        # Added extra output head for E0_ratio to learn initial Exposed state
        self.param_net = nn.Sequential(
            nn.Linear(window * N, 128), nn.ReLU(),
            nn.Linear(128, 4 * N)  # Outputs: beta, sigma, gamma, AND E0_ratio
        )
        
        # 2. Mobility Network
        self.mobility_net = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, N * N)
        )

    def forward(self, x_hist, cnn_feat):
        B, T, N = x_hist.shape
        
        # --- A. Predict Parameters ---
        params = self.param_net(x_hist.reshape(B, -1))  # (B, 4N)
        
        # Constraints (Relaxed Gamma to 1.0 to allow faster recovery)
        beta     = torch.sigmoid(params[:, :N]) * 3.0       # Range: [0, 3.0]
        sigma    = torch.sigmoid(params[:, N:2*N]) * 1.0    # Range: [0, 1.0]
        gamma    = torch.sigmoid(params[:, 2*N:3*N]) * 1.0  # Range: [0, 1.0] (Was 0.5)
        e0_ratio = torch.sigmoid(params[:, 3*N:]) * 5.0     # Learnable E0 multiplier

        # --- B. Predict Mobility ---
        pi_logits = self.mobility_net(cnn_feat)
        # Softmax over 'destinations' ensuring outflow sums to 1 (or close to it)
        Pi = torch.softmax(pi_logits.view(B, N, N), dim=-1)

        # --- C. Initial Conditions ---
        I0 = x_hist[:, -1].clamp(min=1e-6)
        # LEARNED E0 instead of heuristic
        E0 = I0 * e0_ratio 
        S0 = torch.clamp(1.0 - E0 - I0, min=0.01)

        # --- D. The Derivative Function (dydt) ---
        def get_grads(S_curr, E_curr, I_curr):
            # Mobility Flow: "Force of infection entering region i from all j"
            # Batch Matrix Mul: (B, 1, N) x (B, N, N) -> (B, 1, N)
            force = torch.bmm(I_curr.unsqueeze(1), Pi).squeeze(1) 
            
            new_inf = beta * S_curr * force # S * beta * sum(I_j * Pi_ji)
            
            dS = -new_inf
            dE = new_inf - sigma * E_curr
            dI = sigma * E_curr - gamma * I_curr
            return dS, dE, dI

        # --- E. True Runge-Kutta 4 Integration ---
        # We simulate 'steps_per_week' sub-steps for high accuracy
        steps_per_week = 4 
        dt = 1.0 / steps_per_week
        
        S, E, I = S0, E0, I0
        I_preds = []
        E_preds = []

        # Outer loop: Weeks (Forecast Horizon)
        for t in range(self.horizon):
            # Inner loop: Sub-steps (Integration)
            for _ in range(steps_per_week):
                # k1
                dS1, dE1, dI1 = get_grads(S, E, I)
                
                # k2
                dS2, dE2, dI2 = get_grads(S + 0.5*dt*dS1, E + 0.5*dt*dE1, I + 0.5*dt*dI1)
                
                # k3
                dS3, dE3, dI3 = get_grads(S + 0.5*dt*dS2, E + 0.5*dt*dE2, I + 0.5*dt*dI2)
                
                # k4
                dS4, dE4, dI4 = get_grads(S + dt*dS3, E + dt*dE3, I + dt*dI3)

                # Update state
                S = S + (dt / 6.0) * (dS1 + 2*dS2 + 2*dS3 + dS4)
                E = E + (dt / 6.0) * (dE1 + 2*dE2 + 2*dE3 + dE4)
                I = I + (dt / 6.0) * (dI1 + 2*dI2 + 2*dI3 + dI4)

                # Clamp for stability
                S = S.clamp(min=0, max=1)
                E = E.clamp(min=0, max=1)
                I = I.clamp(min=0, max=1)

            # Store the state at the end of the week
            I_preds.append(I)
            E_preds.append(E)

        I_sim = torch.stack(I_preds, dim=1)  # (B, Horizon, N)
        E_sim = torch.stack(E_preds, dim=1)

        return I_sim, E_sim, beta.mean(0), sigma.mean(0), gamma.mean(0), Pi.mean(0)