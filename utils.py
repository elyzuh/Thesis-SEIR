# utils.py
# Modified for SEIR-PINN-NGM compatibility
# Keeps all original functionality + supports binary adjacency

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os


class Data_utility(object):
    def __init__(self, args):
        self.data_path = args.data
        self.train_ratio = args.train
        self.valid_ratio = args.valid
        self.horizon = args.horizon
        self.window = args.window
        self.normalize = args.normalize
        self.sim_mat = args.sim_mat  # only for CNNRNN_Res_epi

        # Load main data
        data = pd.read_csv(self.data_path).values  # (T, N+1) → drop date
        self.rawdat = data[:, 1:]  # (T, N)
        self.dat = np.zeros_like(self.rawdat)

        # === ADJACENCY MATRIX LOADING ===
        if args.model == "EpiSEIRCNNRNNRes_PINN":
            # Use binary adjacency from GenerateAdjacentMatrix.py
            adj_file = 'data/us_hhs/ind_mat2.txt'
            if not os.path.exists(adj_file):
                raise FileNotFoundError(f"Adjacency matrix not found: {adj_file}")
            self.adj = np.loadtxt(adj_file, delimiter=',')
            print(f"Loaded binary adjacency matrix from {adj_file} (shape: {self.adj.shape})")
        else:
            # Original behavior: use sim_mat for CNNRNN_Res_epi
            if args.sim_mat is None:
                raise ValueError("sim_mat is required for CNNRNN_Res_epi")
            self.adj = np.loadtxt(args.sim_mat, delimiter=',')
            print(f"Loaded similarity matrix from {args.sim_mat}")

        self.m = self.rawdat.shape[1]  # number of locations
        self.T = self.rawdat.shape[0]  # time steps

        # Normalization
        # normalize == 2: divide each location by its max -> maps to [0,1]
        # normalize == 1: standard normalization (zero mean, unit std)
        # normalize == 0: no normalization
        if self.normalize == 2:
            self.scale = np.max(self.rawdat, axis=0)
            self.scale[self.scale == 0] = 1
            self.dat = self.rawdat / self.scale
        elif self.normalize == 1:
            self.scale = np.std(self.rawdat, axis=0)
            self.scale[self.scale == 0] = 1
            self.dat = (self.rawdat - np.mean(self.rawdat, axis=0)) / self.scale
        else:
            self.dat = self.rawdat

        # Split train/valid/test
        train_len = int(self.T * self.train_ratio)
        valid_len = int(self.T * self.valid_ratio)
        self.train = self._create_dataset(0, train_len)
        self.valid = self._create_dataset(train_len, train_len + valid_len)
        self.test = self._create_dataset(train_len + valid_len, self.T)

        # Scale for loss
        self.scale = torch.from_numpy(self.scale).float()
        self.rse = self._relative_error(self.train[1], np.zeros_like(self.train[1]))
        self.rae = self._relative_absolute_error(self.train[1], np.zeros_like(self.train[1]))

    def _create_dataset(self, start, end):
        inputs, targets = [], []
        for i in range(start, end - self.window - self.horizon + 1):
            inputs.append(self.dat[i:i + self.window])
            targets.append(self.dat[i + self.window:i + self.window + self.horizon])
        return np.array(inputs), np.array(targets)  # (samples, window, N), (samples, horizon, N)


    # yields mini-batches from a (inputs, targets) dataset for training/validation.
    def get_batches(self, data, batch_size, shuffle=True):
        inputs, targets = data
        length = len(inputs)
        if shuffle:
            index = np.random.permutation(length)
        else:
            index = np.arange(length)
        start_idx = 0
        while start_idx < length:
            end_idx = min(start_idx + batch_size, length)
            excerpt = index[start_idx:end_idx]
            yield torch.FloatTensor(inputs[excerpt]), torch.FloatTensor(targets[excerpt])
            start_idx = end_idx

    def _relative_error(self, true, pred):
        return np.sqrt(np.mean((true - pred) ** 2)) / (np.mean(true) + 1e-8)

    def _relative_absolute_error(self, true, pred):
        return np.mean(np.abs(true - pred)) / (np.mean(np.abs(true)) + 1e-8)