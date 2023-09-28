import os
import argparse

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import trange

from src.configs.train_defaults import get_cfg_defaults
from src.neural_ode_models.one_dim import normal_kl, log_normal_pdf
from src.neural_ode_models.one_dim import NeuralODE as NeuralODE1D
from src.neural_ode_models.two_dim import NeuralODE as NeuralODE2D
from src.data_driven_dataset import PDEDataset
from utils.utils import get_date_time_str, save_config, set_seed, print_model_size


def get_model(pde_type):
    if pde_type == 'fn2d':
        return NeuralODE2D
    else:
        return NeuralODE1D


def eval_model(config, model, dataloader, epoch, show_fig, path, device):
    test_loss_array = []

    with torch.no_grad():
        for f_data, gt_f_data in dataloader:
            t = torch.arange(0.0, dataloader.dataset.t_len, dataloader.dataset.delta_t).to(device)  #[:dataloader.dataset.self.max_t_context]
            pred_f, _, _ = model(f_data.to(device), t, eval=True)
            test_error = ((pred_f - gt_f_data.to(device)) ** 2).mean()
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
    input_dim = train_dataset.sol_data.shape[1:]
    model = get_model(config.pde_type)(input_dim).to(device)
    print_model_size(model, 'Latent-ODE')
    quit()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    # Evaluate pretrained model to have a baseline
    test_pde_loss = eval_model(config, model, val_dataloader, 'Pretraining', False, path, device)
    print(f'Test error pretraining. PDE loss = %.6f' % test_pde_loss)

    # ----------------------------------------------------------------
    # --------------------- Start training ---------------------------
    # ----------------------------------------------------------------
    for epoch in trange(config.train.num_epochs):
        losses = []
        for f_data in train_dataloader:
            optimizer.zero_grad()
            f_data = f_data.to(device)

            t = torch.arange(0.0, train_dataset.t_len, train_dataset.delta_t)[:train_dataset.max_t_context].to(device)
            pred_f, qz0_mean, qz0_logvar = model(f_data, t)

            noise_std = torch.zeros(pred_f.size()).to(device) + 0.3
            noise_logvar = 2. * torch.log(noise_std).to(device)
            logpx = log_normal_pdf(f_data, pred_f, noise_logvar).mean()#.sum(list(range(1, len(f_data.shape))))
            # pred_loss = ((f_data - pred_f) ** 2).mean()
            pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1).mean()

            # loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss = -logpx + 1e-2 * analytic_kl

            # loss = pred_loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

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
