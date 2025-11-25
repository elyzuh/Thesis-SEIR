# utils.py — FINAL 100% WORKING VERSION FOR US HHS (9 regions) + SEIR-PINN
# Fixes: inhomogeneous arrays, unpacking error, scaling, adjacency loading

import numpy as np
import pandas as pd
import torch
import os

class Data_utility(object):
    def __init__(self, args):
        self.data_path = args.data
        self.train_ratio = args.train
        self.valid_ratio = args.valid
        self.horizon = args.horizon
        self.window = args.window
        self.normalize = args.normalize

        # ======================= Load main data =======================
        data = pd.read_csv(self.data_path).values
        self.rawdat = data[:, 1:].astype(np.float32)  # drop date
        self.T, self.m = self.rawdat.shape            # (weeks, regions)

        # ======================= Adjacency matrix (HHS) =======================
        adj_file = 'data/us_hhs/ind_mat2.txt'
        if not os.path.exists(adj_file):
            raise FileNotFoundError(f"Missing adjacency matrix: {adj_file}")
        self.adj = np.loadtxt(adj_file, delimiter=',')
        print(f"Loaded binary adjacency matrix from {adj_file} (shape: {self.adj.shape})")

        # ======================= Normalization (normalize=2 = divide by train max) =======================
        train_len = int(self.T * self.train_ratio)
        train_data = self.rawdat[:train_len]

        if self.normalize == 2:
            max_val = np.max(train_data, axis=0)
            max_val[max_val == 0] = 1.0
            self.dat = self.rawdat / max_val[None, :]           # broadcast
            self.scale = torch.from_numpy(max_val).float()
        else:
            self.dat = self.rawdat.copy()
            self.scale = torch.ones(self.m)

        # ======================= Create datasets safely =======================
        def create_dataset(start_idx, end_idx):
            seq_len_needed = self.window + self.horizon
            if end_idx - start_idx < seq_len_needed:
                return [], []  # no valid sequences

            inputs, targets = [], []
            for i in range(start_idx, end_idx - seq_len_needed + 1):
                x = self.dat[i:i + self.window]                    # (window, m)
                y = self.dat[i + self.window:i + self.window + self.horizon]  # (horizon, m)
                inputs.append(x)
                targets.append(y)
            return inputs, targets

        # Split points
        train_end = int(self.T * self.train_ratio)
        valid_end = train_end + int(self.T * self.valid_ratio)

        train_x, train_y = create_dataset(0, train_end)
        valid_x, valid_y = create_dataset(train_end, valid_end)
        test_x,  test_y  = create_dataset(valid_end, self.T)

        # Convert to list of torch tensors (this fixes the unpacking error)
        self.train = [(torch.FloatTensor(x), torch.FloatTensor(y)) for x, y in zip(train_x, train_y)]
        self.valid = [(torch.FloatTensor(x), torch.FloatTensor(y)) for x, y in zip(valid_x, valid_y)]
        self.test  = [(torch.FloatTensor(x), torch.FloatTensor(y)) for x, y in zip(test_x,  test_y)]

        print(f"Data loaded: {self.m} regions × {self.T} weeks")
        print(f"Dataset ready → Train: {len(self.train)} | Valid: {len(self.valid)} | Test: {len(self.test)} samples")

        # Move scale to GPU if available
        if torch.cuda.is_available():
            self.scale = self.scale.cuda()

        # RSE denominator (from training targets only)
        if len(self.train) > 0:
            train_targets = torch.stack([y for _, y in self.train])
            self.rse_denominator = torch.sqrt(torch.mean(train_targets**2, dim=(0,1))) + 1e-8
            if torch.cuda.is_available():
                self.rse_denominator = self.rse_denominator.cuda()
        else:
            self.rse_denominator = torch.ones(self.m)

    # Fixed batch generator
    def get_batches(self, data_list, batch_size, shuffle=True):
        if shuffle:
            indices = np.random.permutation(len(data_list))
        else:
            indices = np.arange(len(data_list))

        for i in range(0, len(data_list), batch_size):
            batch = [data_list[j] for j in indices[i:i + batch_size]]
            if len(batch) == 0:
                continue
            batch_x = torch.stack([x for x, y in batch])
            batch_y = torch.stack([y for x, y in batch])
            yield batch_x, batch_y

    def _relative_error(self, true, pred):
        return np.sqrt(np.mean((true - pred)**2, axis=(0,1))) / (self.rse_denominator.cpu().numpy() + 1e-8)

    def _relative_absolute_error(self, true, pred):
        return np.mean(np.abs(true - pred), axis=(0,1)) / (np.mean(np.abs(true), axis=(0,1)) + 1e-8)