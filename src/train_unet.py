import os
import argparse

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from unet_models.two_dim import Unet as UnetTwoDim
from unet_models.one_dim import Unet as UnetOneDim

from src.configs.train_defaults import get_cfg_defaults
from src.data_driven_dataset import PDEDataset

from utils.utils import get_date_time_str, save_config, set_seed, print_model_size


def get_model(pde_type):
    if pde_type == 'fn2d':
        return UnetTwoDim
    else:
        return UnetOneDim


def eval_model(config, model, dataloader, epoch, show_fig, path, device):
    test_loss_array = []
    with torch.no_grad():
        for f_data, _ in dataloader:
            if config.pde_type != 'fn2d':
                f_data = f_data.unsqueeze(2).unsqueeze(4)

            inputs = f_data[:, :-1].to(device)
            outputs = f_data[:, -1].unsqueeze(1).to(device)
            pred = model(inputs)
            test_error = ((pred - outputs) ** 2).mean()
            test_loss_array.append(test_error)

    return torch.FloatTensor(test_loss_array).mean()


def train(config):
    device = torch.device('cpu') if ((not torch.cuda.is_available()) or config.system.cpu) else torch.device('cuda')
    set_seed(config.system.seed)
    path = config.results_path
    if config.create_timestamp:
        path = os.path.join(path, get_date_time_str())

    os.makedirs(path, exist_ok=True)

    # ----------------------------------------------------------------
    # ---------------------- Create datasets -------------------------
    # ----------------------------------------------------------------
    data_path = os.path.join(config.data.path, 'sol_dataset.pkl')
    data_config_path = os.path.join(config.data.path, 'config.yaml')

    train_dataset = PDEDataset(data_path, data_config_path, mode='train', t_len_pct=config.data.t_len_pct)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
                                  num_workers=config.system.num_workers)

    val_dataset = PDEDataset(data_path, data_config_path, mode='val', t_len_pct=config.data.t_len_pct)
    val_dataloader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)

    test_dataset = PDEDataset(data_path, data_config_path, mode='test', t_len_pct=config.data.t_len_pct)
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)

    # ----------------------------------------------------------------
    # --------------- Create model and optimizer ---------------------
    # ----------------------------------------------------------------
    if config.pde_type == 'fn2d':
        n_input_scalar_components = 2
    else:
        n_input_scalar_components = 1

    model = get_model(config.pde_type)(n_input_scalar_components=n_input_scalar_components,
                                       n_input_vector_components=0,
                                       n_output_scalar_components=n_input_scalar_components,
                                       n_output_vector_components=0,
                                       time_history=train_dataset.max_t_context - 1,
                                       time_future=1,
                                       hidden_channels=16,
                                       activation='relu')

    print_model_size(model, 'Unet')

    # print(f'Model Unet has {print_model_size(model)} params.')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    for epoch in trange(config.train.num_epochs):
        losses = []
        for f_data in train_dataloader:
            if config.pde_type != 'fn2d':
                f_data = f_data.unsqueeze(2).unsqueeze(4)
            optimizer.zero_grad()
            inputs = f_data[:, :-1].to(device)
            target = f_data[:, -1].to(device).unsqueeze(1)
            pred = model(inputs)
            pred_loss = ((pred - target) ** 2).mean()

            pred_loss.backward()
            optimizer.step()

            losses.append(pred_loss.item())

        test_pde_loss = eval_model(config, model, test_dataloader, epoch + 1, False, path, device)
        print(f'Error on epoch [%d/%d]: train loss = %.6f, test_loss = %.6f' %
              (epoch + 1, config.train.num_epochs, np.mean(losses), test_pde_loss))

    torch.save(model.state_dict(), os.path.join(path, 'model_checkpoint.pkl'))
    save_config(config, os.path.join(path, 'config.yaml'))
    print(f'Train complete. Model and outputs saved in {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--config-list', nargs="+", default=None)

    args = parser.parse_args()

    config = get_cfg_defaults(args.config_file, args.config_list)

    train(config)
