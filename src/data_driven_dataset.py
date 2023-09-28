import os.path

import yaml
import numpy as np
import torch
import h5py

from torch.utils.data.dataset import Dataset


class PDEDataset(Dataset):
    def __init__(self, sol_path, data_config_path, mode, t_len_pct=1.0, use_previous_t=True, data_size=-1):
        self.use_previous_t = use_previous_t
        self.mode = mode
        # Read the configuration of the data to fetch dx and dt
        with open(data_config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            self.delta_t = config['data']['delta_t']
            self.delta_x = config['data']['delta_x']
            self.t_len = config['data']['t_len']
            self.x_len = config['data']['x_len']

        # Data of size Signals x T x Variable len
        if os.path.isfile(sol_path):
            sol_data = torch.load(sol_path)
        elif os.path.isfile(f"{os.path.splitext(sol_path)[0]}.hdf5"):
            sol_path = f"{os.path.splitext(sol_path)[0]}.hdf5"
            with h5py.File(sol_path, 'r') as f:
                sol_data = f['sol_dataset'][:]
        else:
            raise FileNotFoundError()

        # Divide to train and test
        train_ratio = 0.9
        buffer = int(train_ratio * sol_data.shape[0])
        self.max_t_context = int(t_len_pct * sol_data.shape[1])

        if mode == 'train':
            if data_size == -1 or data_size > buffer:
                sol_data = sol_data[:buffer, :self.max_t_context]
            else:
                sol_data = sol_data[:data_size, :self.max_t_context]
        elif mode == 'test':
            self.gt_sol_data = sol_data[buffer:]
            sol_data = sol_data[buffer:, :self.max_t_context]
        elif mode == 'val':
            num_samples = sol_data.shape[0] - buffer
            self.gt_sol_data = sol_data[:num_samples]
            sol_data = sol_data[:num_samples, :self.max_t_context]
        else:
            raise ValueError('Unsupported training mode used.')

        self.sol_data = sol_data

        if mode == 'train':
            print(f'Train data created with {self.sol_data.shape[0]} train samples, and {self.max_t_context} time steps.')

    def __len__(self):
        return self.sol_data.shape[0]

    def __getitem__(self, idx):
        f_data = torch.FloatTensor(self.sol_data[idx])
        if self.mode == 'train':
            return f_data
        else:
            gt_f_data = torch.FloatTensor(self.gt_sol_data[idx])
            return f_data, gt_f_data
