import os
import argparse
import sys
import inspect

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import trange

from configs.train_defaults import get_cfg_defaults
from pde_models.pdexplain_model import PDExplain
from dataset import PDEDataset

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import utils


def transform_gt_params(gt_params, pde_type, f_data):
    if pde_type == 'burgers':
        # params are a and -u
        new_gt_params = torch.cat((gt_params, -1.0 * f_data[:, 1].unsqueeze(2)), dim=2)

    elif pde_type == 'fn2d':
        # params are k and Rv
        u = f_data[:, 1, 0]
        v = f_data[:, 1, 1]
        k = gt_params[..., 2]
        Rv_gt = u - v
        new_gt_params = torch.cat((k.unsqueeze(3), Rv_gt.unsqueeze(3)), dim=3)

    elif pde_type == 'fn2d_u':
        # params are Ru and Rv
        u = f_data[:, 1, 0]
        v = f_data[:, 1, 1]
        k = gt_params[..., 2]
        Ru_gt = u - u ** 3 - k - v
        Rv_gt = u - v
        new_gt_params = torch.cat((Ru_gt.unsqueeze(3), Rv_gt.unsqueeze(3)), dim=3)

    else:
        new_gt_params = gt_params

    return new_gt_params


def eval_model(config, model, dataloader, device):
    test_pde_loss_array = []
    test_pde_loss_array_gt = []
    ae_recon_loss_array = []
    params_error_array = []

    all_gt_params = []
    all_pred_params = []
    with torch.no_grad():
        for batch in dataloader:
            f_data = batch[0].to(device)
            t = batch[1].to(device)
            gt_params = transform_gt_params(batch[2].to(device), config.pde_type, f_data).to(device)
            context = batch[3].to(device)

            recon_context, pred_params = model(t, f_data[:, 1], context)
            test_pde_loss = model.loss_func(f_data, dataloader.dataset.delta_t, dataloader.dataset.delta_x, pred_params)
            test_pde_loss_gt = model.loss_func(f_data, dataloader.dataset.delta_t, dataloader.dataset.delta_x, gt_params)
            ae_recon_loss = calc_ae_recon_loss(context, recon_context)
            params_error = ((gt_params.to(device) - pred_params) ** 2).mean()

            test_pde_loss_array.append(test_pde_loss)
            test_pde_loss_array_gt.append(test_pde_loss_gt)
            ae_recon_loss_array.append(ae_recon_loss)
            params_error_array.append(params_error)

            all_gt_params.append(gt_params.cpu().numpy())
            all_pred_params.append(pred_params.cpu().numpy())

    return torch.FloatTensor(test_pde_loss_array).mean(), torch.FloatTensor(ae_recon_loss_array).mean(), torch.FloatTensor(params_error_array).mean()


def calc_ae_recon_loss(context, pred_context, loss_type='l2'):
    if loss_type == 'l1':
        loss = torch.abs(context - pred_context).mean()
    elif loss_type == 'l2':
        loss = ((context - pred_context) ** 2).mean()
    else:
        raise ValueError(f'loss_type can be only l1 / l2, but got {loss_type}')
    return loss


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
                               t_len_pct=config.data.t_len_pct, data_size=config.data.size,
                               noise_coeff=config.data.noise)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
                                  num_workers=config.system.num_workers)

    test_dataset = PDEDataset(data_path, params_path, x_path, t_path, data_config_path,
                              mode='test', t_len_pct=config.data.t_len_pct, noise_coeff=config.data.noise)
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)

    # ----------------------------------------------------------------
    # --------------- Create model and optimizer ---------------------
    # ----------------------------------------------------------------
    input_dim = train_dataset.context_shape
    model = PDExplain(input_dim, config.pde_type, config.model.use_ic_in_decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)

    # Evaluate pretrained model to have a baseline
    test_pde_loss, ae_recon_loss, params_error = eval_model(config, model, test_dataloader, device)
    print(f'Test error pretraining. PDE loss = %.6f, ae loss = %.6f, Params error = %.3f' % (
    test_pde_loss, ae_recon_loss, params_error))

    # Create a model for saving the best model found (on val set)
    best_model = PDExplain(input_dim, config.pde_type, config.model.use_ic_in_decoder).to(device)
    best_loss = np.inf

    # ----------------------------------------------------------------
    # --------------------- Start training ---------------------------
    # ----------------------------------------------------------------
    for epoch in trange(config.train.num_epochs):
        losses = []
        train_params_loss = []
        for batch in train_dataloader:
            f_data = batch[0].to(device)
            t = batch[1].to(device)
            gt_params = transform_gt_params(batch[2].to(device), config.pde_type, f_data).to(device)
            context = batch[3].to(device)

            recon_context, pred_params = model(t, f_data[:, 1], context)

            batch_params_error = ((gt_params - pred_params) ** 2).mean()
            train_params_loss.append(batch_params_error.item())

            pde_loss = model.loss_func(f_data, train_dataset.delta_t, train_dataset.delta_x, pred_params)
            ae_recon_loss = calc_ae_recon_loss(context, recon_context)

            loss = config.train.pde_loss_coeff * pde_loss + config.train.ae_loss_coeff * ae_recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(pde_loss.item())

        test_pde_loss, ae_recon_loss, params_error = eval_model(config, model, test_dataloader, device)
        print(
            f'Test error on epoch [%d/%d] PDE loss = %.6f, ae loss = %.6f, test params error = %.6f, train params error = %.6f' % (
                epoch + 1, config.train.num_epochs, test_pde_loss, ae_recon_loss, params_error,
                np.mean(train_params_loss)))

        if np.mean(losses) < best_loss:
            best_loss = np.mean(losses)
            best_model.load_state_dict(model.state_dict())
            print(f'Found a new best model')

    test_pde_loss, ae_recon_loss, params_error = eval_model(config, best_model, test_dataloader, device)
    print(f'Best model test error: PDE loss = %.6f, ae loss = %.6f, test params error = %.3f' % (
        test_pde_loss, ae_recon_loss, params_error))

    torch.save(best_model.state_dict(), os.path.join(path, 'model_checkpoint.pkl'))
    utils.utils.save_config(config, os.path.join(path, 'config.yaml'))
    print(f'Train complete. Model and outputs saved in {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--config-list', nargs="+", default=None)

    args = parser.parse_args()

    config = get_cfg_defaults(args.config_file, args.config_list)

    train(config)
