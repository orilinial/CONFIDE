import os
import argparse
import sys
import inspect

import torch
import pde
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import trange

from configs.train_defaults import get_cfg_defaults
from dataset import PDEDataset

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import utils


@torch.no_grad()
def solve_pde(initial_conditions, delta_t, delta_x, t_len, x_len, params, pde_type):
    grid = pde.CartesianGrid([[0.0, x_len]], x_len // delta_x)
    state = pde.ScalarField(grid, data=initial_conditions)
    bc = [{'value': initial_conditions[0]}, {'value': initial_conditions[-1]}]
    eq = get_pde(pde_type)(bc, params)

    storage = pde.MemoryStorage()
    eq.solve(state, t_range=t_len, dt=delta_t/2.0, tracker=storage.tracker(delta_t))

    return np.array(storage.data)


def get_pde(pde_type):
    if pde_type == 'const':
        return ConstPDE
    else:
        raise ValueError(f'pde type should be [const] but got {pde_type}')
    

class ConstPDE(pde.PDEBase):
    def __init__(self, bc, params):
        super(ConstPDE, self).__init__()
        self.bc = bc
        self.params = params

    def evolution_rate(self, state, t=0):
        a = self.params[0]
        b = self.params[1]
        c = self.params[2]

        dx_dt = a * state.laplace(self.bc) + b * state.gradient(self.bc).to_scalar() + c
        return dx_dt


def calc_const_pde_loss(f_data, x_data, delta_t, delta_x, params, reduce=True):
    a = params[..., 0].unsqueeze(1).expand(params.shape[0], f_data.shape[2])
    b = params[..., 1].unsqueeze(1).expand(params.shape[0], f_data.shape[2])
    c = params[..., 2].unsqueeze(1).expand(params.shape[0], f_data.shape[2])

    # using the EXPLICIT scheme: two rows of the solution for two consecutive time steps
    df_dt = torch.gradient(f_data, dim=1)[0][:, 1] / delta_t
    df_dx = torch.gradient(f_data[:, 1], dim=1)[0] / delta_x
    d2f_dx2 = torch.gradient(df_dx, dim=1)[0] / delta_x

    pde_loss = ((df_dt - a * d2f_dx2 - b * df_dx - c) ** 2)
    if reduce:
        pde_loss = pde_loss.mean()

    return pde_loss


def get_pde_loss_func(pde_type):
    if pde_type == 'const':
        return calc_const_pde_loss
    else:
        raise ValueError(f'pde type should be [const] but got {pde_type}')


def infer(config):
    utils.utils.set_seed(config.system.seed)
    path = config.results_path
    if config.create_timestamp:
        path = os.path.join(path, utils.utils.get_date_time_str())

    os.makedirs(path, exist_ok=True)

    # ----------------------------------------------------------------
    # ---------------------- Create datasets -------------------------
    # ----------------------------------------------------------------
    data_path = os.path.join(config.data.path, 'sol_dataset.pkl')
    params_path = os.path.join(config.data.path, 'parameters_dataset.pkl')
    x_path = os.path.join(config.data.path, 'x_dataset.pkl')
    t_path = os.path.join(config.data.path, 't_dataset.pkl')
    data_config_path = os.path.join(config.data.path, 'config.yaml')

    test_dataset = PDEDataset(data_path, params_path, x_path, t_path, data_config_path,
                              mode='test', t_len_pct=config.data.t_len_pct, noise_coeff=config.data.noise)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # ----------------------------------------------------------------
    # ------------ Initialize parameters and optimizer ---------------
    # ----------------------------------------------------------------
    pde_params = torch.nn.Parameter(torch.ones((test_dataset.sol_data.shape[0], test_dataset.params.shape[-1])))
    optimizer = torch.optim.Adam([pde_params], lr=config.train.lr)

    # ----------------------------------------------------------------
    # --------------------- Start Inferring --------------------------
    # ----------------------------------------------------------------
    loss_func = calc_const_pde_loss

    for epoch in trange(config.train.num_epochs):
        context_data = test_dataset.context_data
        losses = []
        for t in range(1, context_data.shape[1]-1):
            f_data = torch.FloatTensor(context_data[:, t-1:t+2])
            loss = loss_func(f_data, torch.FloatTensor(test_dataset.x), test_dataset.delta_t, test_dataset.delta_x, pde_params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        with torch.no_grad():
            param_error = ((pde_params - test_dataset.params[:, 0, 0]) ** 2).mean()

        if epoch % 50 == 0:
            print(f'DI Test error on epoch [%d/%d] PDE loss = %.6f, params error = %.6f' % (
                epoch + 1, config.train.num_epochs, np.mean(losses), param_error))

    torch.save(pde_params, os.path.join(path, 'pde_params.pkl'))
    utils.utils.save_config(config, os.path.join(path, 'config.yaml'))
    print(f'Inference complete. Output saved in {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--config-list', nargs="+", default=None)

    args = parser.parse_args()

    config = get_cfg_defaults(args.config_file, args.config_list)

    infer(config)
