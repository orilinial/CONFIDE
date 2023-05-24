import os
import argparse
import sys
import inspect

import numpy as np
import torch
from tqdm import trange
from joblib import Parallel, delayed
import h5py

from configs.create_data_defaults import get_cfg_defaults
import samplers

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import utils


def get_pde(pde_type):
    if pde_type == 'const':
        return samplers.ConstEquationSampler
    elif pde_type == 'burgers':
        return samplers.BurgersEquationSampler
    elif pde_type == 'fn2d':
        return samplers.FNSampler
    else:
        raise ValueError('Wrong type of PDE. Supported types: [const / burgers]')


def make_dataset(config):
    utils.utils.set_seed(config.system.seed)

    if config.system.parallel:
        def create_single_signal(i):
            good_sample = False
            while not good_sample:
                pde = get_pde(config.pde.type)(config)
                pde_sol, x, t, parameters = pde.sample_pde_sol(config.show_fig)

                if config.pde.type == 'fn2d':
                    # No bad samples
                    break

                # Testing that the solution makes sense.
                # If not this is probably an error of the PDE solver,
                # In this case do not add this sample to the dataset and sample a new one.
                if pde_sol[:, 0].max() > 0.5 or pde_sol[:, 0].min() < -0.5:
                    continue

                if pde_sol[:, -1].max() > 0.5 or pde_sol[:, -1].min() < -0.5:
                    continue

                if pde_sol.max() > 4 or pde_sol.min() < -4:
                    continue

                good_sample = True

            return pde_sol, x, t, parameters

        res = Parallel(n_jobs=-1, verbose=10)(delayed(create_single_signal)(i) for i in trange(config.data.size))
        pde_sol, x, t, parameters = zip(*res)
        sol_dataset = np.stack(pde_sol, axis=0)
        x_dataset = np.stack(x, axis=0)
        t_dataset = np.stack(t, axis=0)
        parameters_dataset = np.stack(parameters, axis=0)

    else:
        parameters_dataset = []
        sol_dataset = []
        x_dataset = []
        t_dataset = []

        for i in trange(config.data.size):
            good_sample = False
            while not good_sample:
                pde = get_pde(config.pde.type)(config)
                pde_sol, x, t, parameters = pde.sample_pde_sol(config.show_fig)

                if config.pde.type == 'fn2d':
                    # No bad samples
                    break

                # Testing that the solution makes sense.
                # If not this is probably an error of the PDE solver,
                # In this case do not add this sample to the dataset and sample a new one.
                if pde_sol[:, 0].max() > 0.5 or pde_sol[:, 0].min() < -0.5:
                    print('BC(x=0) Error: solution to the PDE is unstable or has very high values.')
                    continue

                if pde_sol[:, -1].max() > 0.5 or pde_sol[:, -1].min() < -0.5:
                    print('BC(x=-1) Error: solution to the PDE is unstable or has very high values.')
                    continue

                if pde_sol.max() > 7 or pde_sol.min() < -7:
                    print('Value Error: solution to the PDE is unstable or has very high values.')
                    continue

                good_sample = True

            x_dataset.append(x)
            t_dataset.append(t)
            sol_dataset.append(pde_sol)
            parameters_dataset.append(parameters)

        x_dataset = np.stack(x_dataset)
        t_dataset = np.stack(t_dataset)
        sol_dataset = np.stack(sol_dataset)
        parameters_dataset = np.stack(parameters_dataset)

    if config.create_timestamp:
        path = os.path.join(config.data.path, utils.utils.get_date_time_str())
    else:
        path = config.data.path

    os.makedirs(path)
    utils.utils.save_config(config, os.path.join(path, 'config.yaml'))

    if config.system.h5py:
        with h5py.File(os.path.join(path, 'sol_dataset.hdf5'), "w") as f:
            f.create_dataset("sol_dataset", data=sol_dataset, compression='gzip')
    else:
        torch.save(sol_dataset, os.path.join(path, 'sol_dataset.pkl'))
        torch.save(x_dataset, os.path.join(path, 'x_dataset.pkl'))
        torch.save(t_dataset, os.path.join(path, 't_dataset.pkl'))
        torch.save(parameters_dataset, os.path.join(path, 'parameters_dataset.pkl'))

    print(f'Dataset created and saved to {path}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--config-file', type=str, default=None)
    parser.add_argument('--config-list', nargs="+", default=None)
    args = parser.parse_args()
    config = get_cfg_defaults(args.config_file, args.config_list)
    make_dataset(config)
