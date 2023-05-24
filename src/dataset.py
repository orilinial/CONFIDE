import os.path

import yaml
import numpy as np
import torch
import h5py

from torch.utils.data.dataset import Dataset


class PDEDataset(Dataset):
    def __init__(self, sol_path, params_path, x_path, t_path, data_config_path, mode, data_size_pct=1.0, t_len_pct=1.0, use_previous_t=True, data_size=-1, noise_coeff=0.0):
        self.use_previous_t = use_previous_t

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

        # x (PDE variable) and t (time) arrays. They are repeated to the shape of Signals x T x Variable len
        self.x = torch.load(x_path)[0]
        self.t = torch.load(t_path)[0]
        self.row_size = self.x.shape[0]

        # Parameters GT dataset of size Signals x T x Variable len x Parameters amount
        params = torch.load(params_path)

        # Divide to train and test
        train_ratio = 0.9
        buffer = int(train_ratio * sol_data.shape[0])
        self.max_t_context = int(t_len_pct * self.t.shape[0])
        self.context_shape = (self.max_t_context, *sol_data.shape[2:])

        if type(params[0]) == dict:
            if len(self.x.shape) == 3:
                params = np.expand_dims([[sig[key] for key in sig] for sig in params], axis=(1, 2, 3)).repeat(self.t.shape[0], axis=1).repeat(self.x.shape[0], axis=2).repeat(self.x.shape[0], axis=3)
            elif len(self.x.shape) == 1:
                params = np.expand_dims([[sig[key] for key in sig] for sig in params], axis=(1, 2)).repeat(self.t.shape[0], axis=1).repeat(self.x.shape[0], axis=2)
        else:
            params = np.expand_dims(params, axis=1).repeat(self.t.shape[0], axis=1)

        if mode == 'train':
            if data_size == -1 or data_size > buffer:
                sol_data = sol_data[:buffer, :self.max_t_context]
                params = params[:buffer, :self.max_t_context]
            else:
                sol_data = sol_data[:data_size, :self.max_t_context]
                params = params[:data_size, :self.max_t_context]
        elif mode == 'test':
            sol_data = sol_data[buffer:, :]
            params = params[buffer:, :]
        else:
            raise ValueError('Unsupported training mode used.')

        noise = np.random.randn(*sol_data.shape)
        sol_data = sol_data + noise_coeff * noise

        self.sol_data = sol_data
        self.context_data = self.sol_data[:, :self.max_t_context]
        self.params = params[:, :-1]

        if mode == 'train':
            print(f'Train data created with {self.sol_data.shape[0]} train samples, and {self.max_t_context} time steps.')

        # Create a mapping from idx to (signal, t)
        self.image_to_idx_dict = {}
        idx = 0
        start_t = 1 if self.use_previous_t else 0
        for signal in range(self.sol_data.shape[0]):
            for t in range(start_t, self.sol_data.shape[1]-1):
                self.image_to_idx_dict[idx] = (signal, t)
                idx += 1

    def __len__(self):
        if self.use_previous_t:
            return self.sol_data.shape[0] * (self.sol_data.shape[1] - 2)
        else:
            return self.sol_data.shape[0] * (self.sol_data.shape[1] - 1)

    def __getitem__(self, idx):
        signal_idx, t_idx = self.image_to_idx_dict[idx]
        if self.use_previous_t:
            f_data = torch.FloatTensor(self.sol_data[signal_idx, t_idx-1:t_idx+2])
        else:
            f_data = torch.FloatTensor(self.sol_data[signal_idx, t_idx:t_idx+2])

        t_data = torch.FloatTensor(self.t[t_idx:t_idx+1])
        params = torch.FloatTensor(self.params[signal_idx, t_idx])
        context = torch.FloatTensor(self.context_data[signal_idx])

        return f_data, t_data, params, context
