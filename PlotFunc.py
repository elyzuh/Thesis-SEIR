# PlotFunc.py
# Thesis-ready visualization for Epi-SEIR-CNNRNN-Res-PINN
# Supports: Input+Pred+GT, Latent E(t), β/σ/γ/R₀, NGM, Mobility
# Saves + Shows all plots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import os


# ========================================
# 1. Plot Input + Prediction + Ground Truth
# ========================================
def PlotTrends(InputRealLocationTimeData, RealLocationTimeData, PredictedLocationTimeData, save_dir, horizon):
    """
    InputReal: (loc, sample, time_in)
    Real: (loc, sample, horizon)
    Pred: (loc, sample, horizon)
    """
    os.makedirs(save_dir, exist_ok=True)
    LocationNumber = RealLocationTimeData.shape[0]
    InputTimeLength = InputRealLocationTimeData.shape[2]
    NumberOfSamples = RealLocationTimeData.shape[1]

    for i in range(LocationNumber):
        fig, ax = plt.subplots(figsize=(15, 6))
        count = 0
        for n in range(NumberOfSamples):
            x_in = np.arange(count, count + InputTimeLength)
            y_in = InputRealLocationTimeData[i, n]

            x_gt = np.array([InputTimeLength + count - 1, InputTimeLength + count + horizon - 1])
            y_gt = np.array([InputRealLocationTimeData[i, n, -1], RealLocationTimeData[i, n, 0]])

            x_pred = x_gt.copy()
            y_pred = np.array([InputRealLocationTimeData[i, n, -1], PredictedLocationTimeData[i, n, 0]])

            if count == 0:
                ax.plot(x_in, y_in, 'o-', color='navy', label='Input (History)', markersize=4, alpha=0.8)
                ax.plot(x_gt, y_gt, 's--', color='tab:blue', label='Ground Truth', markersize=5)
                ax.plot(x_pred, y_pred, 'd--', color='tab:red', label='Prediction', markersize=5)
            else:
                ax.plot(x_in, y_in, 'o-', color='navy', markersize=4, alpha=0.8)
                ax.plot(x_gt, y_gt, 's--', color='tab:blue', markersize=5)
                ax.plot(x_pred, y_pred, 'd--', color='tab:red', markersize=5)

            ax.axvline(x_in[-1], color='gray', lw=0.5, alpha=0.7)
            count += 1

        ax.set_xlim(-1, InputTimeLength + NumberOfSamples + horizon - 1)
        ax.set_title(f"Location {i+1} - Input + Forecast")
        ax.set_xlabel("Week")
        ax.set_ylabel("Influenza Activity Level")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = f"{save_dir}Input_Prediction_Groundtruth_Loc{i+1}.pdf"
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()


# ========================================
# 2. Plot Prediction vs Ground Truth (Test Set)
# ========================================
def PlotPredictionTrends(RealLocationTimeData, PredictedLocationTimeData, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    LocationNumber, TimeLength = RealLocationTimeData.shape
    x = np.arange(TimeLength)

    for i in range(LocationNumber):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, RealLocationTimeData[i], 'o-', color='tab:blue', label='Ground Truth', markersize=4)
        ax.plot(x, PredictedLocationTimeData[i], 's--', color='tab:red', label='Prediction', markersize=4)
        for t in x[::4]:
            ax.axvline(t, color='gray', lw=0.3, alpha=0.5)
        ax.set_xlim(-1, TimeLength)
        ax.set_title(f"Location {i+1} - Test Set Forecast")
        ax.set_xlabel("Week")
        ax.set_ylabel("Influenza Activity Level")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = f"{save_dir}Prediction_Groundtruth_Loc{i+1}.pdf"
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()


# ========================================
# 3. Plot Latent Exposed E(t)
# ========================================
def PlotLatentE(EList, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    LocationNumber, TimeLength = EList.shape
    x = np.arange(TimeLength)

    for i in range(LocationNumber):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, EList[i], '^-', color='tab:green', label='Inferred E(t)', markersize=4)
        for t in x[::4]:
            ax.axvline(t, color='gray', lw=0.3, alpha=0.5)
        ax.set_title(f"Latent Exposed E(t) - Location {i+1}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Exposed (E)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = f"{save_dir}LatentE_Loc{i+1}.pdf"
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()


# ========================================
# 4. Plot Parameters: β, σ, γ, R₀
# ========================================
def PlotParameters(BetaList, SigmaList, GammaList, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    LocationNumber, TimeLength = BetaList.shape
    x = np.arange(TimeLength)

    param_names = ['β (Transmission)', 'σ (Incubation)', 'γ (Recovery)']
    colors = ['tab:red', 'tab:purple', 'tab:cyan']
    data = [BetaList, SigmaList, GammaList]

    for i in range(LocationNumber):
        fig, ax = plt.subplots(figsize=(15, 6))
        ax2 = ax.twinx()
        lines = []
        labels = []

        for param, name, color in zip(data, param_names, colors):
            line = ax.plot(x, param[i], 'o-', color=color, label=name.split()[0], markersize=4, alpha=0.8)[0]
            lines.append(line)
            labels.append(name.split()[0])

        # R₀ = β / γ
        R0 = BetaList[i] / (GammaList[i] + 1e-8)
        line_r0 = ax2.plot(x, R0, 's--', color='black', label='$R_0$', markersize=4)[0]
        lines.append(line_r0)
        labels.append('$R_0$')

        for t in x[::4]:
            ax.axvline(t, color='gray', lw=0.3, alpha=0.5)

        ax.set_xlim(-1, TimeLength)
        ax.set_title(f"Epidemiological Parameters - Location {i+1}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Rate")
        ax2.set_ylabel("$R_0$")

        ax.legend(lines[:3], labels[:3], loc='upper left')
        ax2.legend([line_r0], [labels[3]], loc='upper right')
        ax.grid(True, alpha=0.3)

        fname = f"{save_dir}Parameters_Loc{i+1}.pdf"
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()


# ========================================
# 5. Plot Single Matrix (NGM or Mobility)
# ========================================
def PlotEachMatrix(Matrix, Type, SaveName, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    TimeLength = Matrix.shape[0]
    cmap = 'Spectral_r' if "NGM" in Type or "Mobility" in Type else 'RdBu_r'

    for t in range(min(4, TimeLength)):
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            Matrix[t], cmap=cmap, annot=False, square=True,
            cbar=True, xticklabels=False, yticklabels=False,
            ax=ax, linewidths=0.1, linecolor='gray'
        )
        ax.set_title(f"{Type} - Week {t+1}")
        fname = f"{save_dir}{SaveName}_t{t+1}.pdf"
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        plt.show()
        plt.close()


# ========================================
# 6. Plot All Matrices in Grid
# ========================================
def PlotAllMatrices(Matrix, Type, SaveName, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    TimeLength = Matrix.shape[0]
    cmap = 'Spectral_r' if "NGM" in Type or "Mobility" in Type else 'RdBu_r'

    vmin, vmax = np.min(Matrix), np.max(Matrix)
    cols = math.ceil(math.sqrt(TimeLength))
    rows = math.ceil(TimeLength / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4.5), constrained_layout=True)

    for t in range(TimeLength):
        r, c = t // cols, t % cols
        ax = axes[r, c] if rows > 1 else axes[c]
        sns.heatmap(
            Matrix[t], cmap=cmap, vmin=vmin, vmax=vmax,
            square=True, cbar=False, xticklabels=False, yticklabels=False,
            ax=ax, linewidths=0.1
        )
        ax.set_title(f"Week {t+1}")

    # Remove empty subplots
    for t in range(TimeLength, rows * cols):
        r, c = t // cols, t % cols
        fig.delaxes(axes[r, c] if rows > 1 else axes[c])

    plt.suptitle(Type, fontsize=16)
    fname = f"{save_dir}{SaveName}_AllTime.pdf"
    plt.savefig(fname, transparent=True, bbox_inches='tight')
    plt.show()
    plt.close()