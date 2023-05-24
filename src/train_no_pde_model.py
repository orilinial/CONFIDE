import os
import argparse
import sys
import inspect

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import trange

from configs.train_defaults import get_cfg_defaults
from no_pde_models.one_d_model import OneDimModel
from no_pde_models.two_d_model import TwoDimModel
from dataset import PDEDataset

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import utils


@torch.no_grad()
def solve_pde(initial_conditions, model, context, t_array, device='cpu'):
    """
    This function outputs a PDE solution on times included in t_array given initial conditions an autoregressive model,
    and a context.
    :param initial_conditions: numpy array (dim=1) of initial conditions to the PDE solutions (i.e., f(x, t=0))
    :param model: Trained model that outputs the solution to the PDE at a given time t
    :param context: a latent compact tensor (dim=1) describing the given solution
    :param t_array: an iterable array (numpy array or list usually) of desired times to solve the PDE for
    :param device: which device the model is on (cpu / cuda)
    :return: a numpy array of the PDE solution with shape = T x X.
    """
    res = np.zeros((t_array.shape[0], *initial_conditions.shape))
    res[0] = initial_conditions
    context_tensor = torch.FloatTensor(context).unsqueeze(0).to(device)

    current_f_tensor = torch.FloatTensor(res[0]).unsqueeze(0).to(device)
    for i in range(t_array.shape[0] - 1):
        current_t_tensor = torch.FloatTensor([t_array[i]]).unsqueeze(0).to(device)
        _, next_f_tensor = model(current_t_tensor, current_f_tensor, context_tensor)
        current_f_tensor = next_f_tensor
        res[i+1] = next_f_tensor.cpu().numpy()
    return res


def eval_model(config, model, dataloader, epoch, show_fig, path, device):
    test_loss_array = []
    ae_recon_loss_array = []

    with torch.no_grad():
        for batch in dataloader:
            f_data = batch[0].to(device)
            t = batch[1].to(device)
            context = batch[3].to(device)

            f_input = f_data[:, 0].to(device)
            target = f_data[:, 1].to(device)

            recon_context, res = model(t, f_input, context)

            test_error = ((target - res) ** 2).mean()
            ae_recon_loss = calc_ae_recon_loss(context, recon_context)

            test_loss_array.append(test_error)
            ae_recon_loss_array.append(ae_recon_loss)

    return torch.FloatTensor(test_loss_array).mean(), torch.FloatTensor(ae_recon_loss_array).mean()


def calc_pde_loss(f_data, delta_t, delta_x, res):
    a = res[..., 0]
    b = res[..., 1]
    c = res[..., 2]

    # using the EXPLICIT scheme: two rows of the solution for two consecutive time steps
    df_dt = torch.gradient(f_data, dim=1)[0][:, 1] / delta_t
    df_dx = torch.gradient(f_data[:, 1], dim=1)[0] / delta_x
    d2f_dx2 = torch.gradient(df_dx, dim=1)[0] / delta_x

    prediction_loss = ((df_dt - a * d2f_dx2 - b * df_dx - c) ** 2).mean()

    loss = prediction_loss
    return loss


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
        return TwoDimModel
    else:
        return OneDimModel


def train(config):
    device = torch.device('cpu') if ((not torch.cuda.is_available()) or config.system.cpu) else torch.device('cuda')
    utils.set_seed(config.system.seed)
    path = config.results_path
    if config.create_timestamp:
        path = os.path.join(path, utils.get_date_time_str())

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
                               t_len_pct=config.data.t_len_pct)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
                                  num_workers=config.system.num_workers)

    test_dataset = PDEDataset(data_path, params_path, x_path, t_path, data_config_path,
                              mode='test', t_len_pct=config.data.t_len_pct)
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)

    # ----------------------------------------------------------------
    # --------------- Create model and optimizer ---------------------
    # ----------------------------------------------------------------
    sol_dim = train_dataset.sol_data.shape[2]
    input_dim = train_dataset.context_shape

    output_dim = sol_dim
    model = get_model(config.pde_type)(input_dim, output_dim, input_dim[1]).to(device)
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

            f_input = f_data[:, 0]
            target = f_data[:, 1]

            recon_context, res = model(t, f_input, context)

            pde_loss = ((target - res) ** 2).mean()
            ae_recon_loss = calc_ae_recon_loss(context, recon_context)
            loss = config.train.pde_loss_coeff * pde_loss + config.train.ae_loss_coeff * ae_recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        test_pde_loss, ae_recon_loss = eval_model(config, model, test_dataloader, epoch + 1, False, path, device)
        print(f'Test error on epoch [%d/%d] PDE loss = %.6f, ae loss = %.6f' % (
            epoch + 1, config.train.num_epochs, test_pde_loss, ae_recon_loss))

    torch.save(model.state_dict(), os.path.join(path, 'model_checkpoint.pkl'))
    utils.save_config(config, os.path.join(path, 'config.yaml'))
    print(f'Train complete. Model and outputs saved in {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--config-list', nargs="+", default=None)

    args = parser.parse_args()

    config = get_cfg_defaults(args.config_file, args.config_list)

    train(config)
