# utils.py
# FINAL WORKING VERSION — 2025 Thesis SEIR-PINN (US HHS data compatible)
# Fixed: data format, scaling, adjacency loading, and tuple conversion

import numpy as np
import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler


class Data_utility(object):
    def __init__(self, args):
        self.data_path = args.data
        self.train_ratio = args.train
        self.valid_ratio = args.valid
        self.horizon = args.horizon
        self.window = args.window
        self.normalize = args.normalize
        self.sim_mat = args.sim_mat  # legacy

        # ======================= Load main data =======================
        data = pd.read_csv(self.data_path).values
        self.rawdat = data[:, 1:].astype(np.float32)  # drop date column
        self.T, self.m = self.rawdat.shape
        self.dat = np.zeros_like(self.rawdat)

        # ======================= ADJACENCY MATRIX =======================
        if args.model == "EpiSEIRCNNRNNRes_PINN":
            adj_file = 'data/us_hhs/ind_mat2.txt'
            if not os.path.exists(adj_file):
                raise FileNotFoundError(f"Adjacency matrix not found: {adj_file}")
            self.adj = np.loadtxt(adj_file, delimiter=',')
            print(f"Loaded binary adjacency matrix from {adj_file} (shape: {self.adj.shape})")
        else:
            if args.sim_mat is None:
                raise ValueError("sim_mat is required for non-PINN models")
            self.adj = np.loadtxt(args.sim_mat, delimiter=',')
            print(f"Loaded similarity matrix from {args.sim_mat}")

        # ======================= SCALING =======================
        train_len = int(self.T * self.train_ratio)
        train_data = self.rawdat[:train_len]

        if self.normalize == 2:
            max_val = np.max(train_data, axis=0)
            max_val[max_val == 0] = 1.0
            self.dat = self.rawdat / max_val
            self.scale = torch.from_numpy(max_val).float()
        elif self.normalize == 1:
            self.scale_mean = train_data.mean(axis=0)
            self.scale_std = train_data.std(axis=0) + 1e-8
            self.dat = (self.rawdat - self.scale_mean) / self.scale_std
            self.scale = torch.from_numpy(self.scale_std).float()
        else:
            self.dat = self.rawdat.copy()
            self.scale = torch.ones(self.m)

        # ======================= Train/Valid/Test Split =======================
        train_len = int(self.T * self.train_ratio)
        valid_len = int(self.T * self.valid_ratio)
        train_end = train_len
        valid_end = train_len + valid_len

        # Create raw numpy arrays first
        train_inputs, train_targets = self._create_dataset(0, train_end)
        valid_inputs, valid_targets = self._create_dataset(train_end, valid_end)
        test_inputs,  test_targets  = self._create_dataset(valid_end, self.T)

        # CRITICAL: Convert to list of (x, y) tensor tuples (this fixes the unpack error)
        self.train = [(torch.FloatTensor(x), torch.FloatTensor(y))
                      for x, y in zip(train_inputs, train_targets)]
        self.valid = [(torch.FloatTensor(x), torch.FloatTensor(y))
                      for x, y in zip(valid_inputs, valid_targets)]
        self.test  = [(torch.FloatTensor(x), torch.FloatTensor(y))
                      for x, y in zip(test_inputs, test_targets)]

        print(f"Dataset ready → Train:{len(self.train)} | Valid:{len(self.valid)} | Test:{len(self.test)} samples")

        # Move scale to GPU if available
        self.scale = self.scale.float()
        if torch.cuda.is_available():
            self.scale = self.scale.cuda()

        # Pre-compute RSE denominator from training targets
        train_np = np.stack([y.numpy() for _, y in self.train])  # (N, horizon, m)
        self.rse_denominator = np.sqrt(np.mean(train_np**2, axis=(0, 1))) + 1e-8
        self.rse_denominator = torch.from_numpy(self.rse_denominator).float()
        if torch.cuda.is_available():
            self.rse_denominator = self.rse_denominator.cuda()

    def _create_dataset(self, start, end):
        inputs, targets = [], []
        L = end - self.window - self.horizon + 1
        for i in range(start, start + max(L, 0)):
            x = self.dat[i:i + self.window]
            y = self.dat[i + self.window:i + self.window + self.horizon]
            inputs.append(x)
            targets.append(y)
        return np.array(inputs), np.array(targets)

    def get_batches(self, data, batch_size, shuffle=True):
        if shuffle:
            indices = np.random.permutation(len(data))
        else:
            indices = np.arange(len(data))
        for i in range(0, len(data), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_x = torch.stack([data[j][0] for j in batch_idx])
            batch_y = torch.stack([data[j][1] for j in batch_idx])
            yield batch_x, batch_y

    def _relative_error(self, true, pred):
        return np.sqrt(np.mean((true - pred) ** 2, axis=(0,1))) / (self.rse_denominator.cpu().numpy() + 1e-8)

    def _relative_absolute_error(self, true, pred):
        return np.mean(np.abs(true - pred), axis=(0,1)) / (np.mean(np.abs(true), axis=(0,1)) + 1e-8)