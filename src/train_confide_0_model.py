import os
import argparse
import sys
import inspect

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import trange

from configs.train_defaults import get_cfg_defaults
from confide_0_models.one_d_model import ConfideZero as ConfideZeroModel1D
from confide_0_models.two_d_model import ConfideZero as ConfideZeroModel2D
from dataset import PDEDataset

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import utils


def eval_model(config, model, dataloader, epoch, show_fig, path, device):
    test_pde_loss_array = []
    ae_recon_loss_array = []

    with torch.no_grad():
        for batch in dataloader:
            f_data = batch[0].to(device)
            t = batch[1].to(device)
            context = batch[3].to(device)

            recon_context, pred_rhs = model(t, f_data[:, 1], context)
            test_pde_loss = model.loss_func(f_data, dataloader.dataset.delta_t, pred_rhs)
            ae_recon_loss = calc_ae_recon_loss(context, recon_context)

            test_pde_loss_array.append(test_pde_loss)
            ae_recon_loss_array.append(ae_recon_loss)

    return torch.FloatTensor(test_pde_loss_array).mean(), torch.FloatTensor(ae_recon_loss_array).mean()


def calc_ae_recon_loss(context, pred_context, loss_type='l2'):
    if loss_type == 'l1':
        loss = torch.abs(context - pred_context).mean()
    elif loss_type == 'l2':
        loss = ((context - pred_context) ** 2).mean()
    else:
        raise ValueError(f'loss_type can be only l1 / l2, but got {loss_type}')
    return loss


def get_model(model_type):
    if model_type == 'fn2d':
        return ConfideZeroModel2D
    else:
        return ConfideZeroModel1D


def train(config):
    device = torch.device('cpu') if ((not torch.cuda.is_available()) or config.system.cpu) else torch.device('cuda')
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

    train_dataset = PDEDataset(data_path, params_path, x_path, t_path, data_config_path, mode='train',
                               t_len_pct=config.data.t_len_pct, data_size=config.data.size)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
                                  num_workers=config.system.num_workers)

    test_dataset = PDEDataset(data_path, params_path, x_path, t_path, data_config_path,
                              mode='test', t_len_pct=config.data.t_len_pct)
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)

    # ----------------------------------------------------------------
    # --------------- Create model and optimizer ---------------------
    # ----------------------------------------------------------------
    input_dim = train_dataset.context_shape

    model = get_model(config.pde_type)(input_dim, train_dataset.x.shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    # Evaluate pretrained model to have a baseline
    test_pde_loss, ae_recon_loss = eval_model(config, model, test_dataloader, 'Pretraining', False, path, device)
    print(f'Test error pretraining. PDE loss = %.6f, ae loss = %.6f' % (test_pde_loss, ae_recon_loss))

    # ----------------------------------------------------------------
    # --------------------- Start training ---------------------------
    # ----------------------------------------------------------------
    for epoch in trange(config.train.num_epochs):
        losses = []
        for batch in train_dataloader:
            f_data = batch[0].to(device)
            t = batch[1].to(device)
            context = batch[3].to(device)

            recon_context, pred_rhs = model(t, f_data[:, 1], context)

            pde_loss = model.loss_func(f_data, train_dataset.delta_t, pred_rhs)
            ae_recon_loss = calc_ae_recon_loss(context, recon_context)

            loss = config.train.pde_loss_coeff * pde_loss + config.train.ae_loss_coeff * ae_recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        test_pde_loss, ae_recon_loss = eval_model(config, model, test_dataloader, epoch + 1, config.show_fig, path, device)
        print(f'Test error on epoch [%d/%d] PDE loss = %.6f, ae loss = %.6f' % (
            epoch + 1, config.train.num_epochs, test_pde_loss, ae_recon_loss))

    torch.save(model.state_dict(), os.path.join(path, 'model_checkpoint.pkl'))
    utils.utils.save_config(config, os.path.join(path, 'config.yaml'))
    print(f'Train complete. Model and outputs saved in {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--config-list', nargs="+", default=None)

    args = parser.parse_args()

    config = get_cfg_defaults(args.config_file, args.config_list)

    train(config)
