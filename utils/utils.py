import yaml
import random

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from datetime import datetime
from yacs.config import CfgNode as CN
import torch


def set_seed(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True


def get_date_time_str(add_hash=True):
    now = datetime.now()
    return_str = 'date_%s_time_%s' % (now.strftime('%d_%m_%Y'), now.strftime('%H_%M'))
    if add_hash:
        return_str = '%s_hash_%s' % (return_str, now.strftime('%f'))
    return return_str


def save_config(config, path):
    def convert_config_to_dict(cfg_node, key_list):
        if not isinstance(cfg_node, CN):
            return cfg_node

        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_config_to_dict(v, key_list + [k])
        return cfg_dict

    with open(path, 'w') as f:
        yaml.dump(convert_config_to_dict(config, []), f, default_flow_style=False)


def show_3d_fig(x, t, y, title=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, T = np.meshgrid(x, t)
    ax.plot_surface(X, T, y, cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('t')
    if title is not None:
        plt.title(title)
    plt.show()
