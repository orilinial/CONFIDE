from yacs.config import CfgNode as CN

_C = CN()

_C.show_fig = True
_C.pde_type = ''    # Possible types [const / burgers / fn2d / fn2d_u]

_C.model = CN()
_C.model.latent_dim = 32
_C.model.use_ic_in_decoder = True

_C.create_timestamp = False
_C.results_path = 'src/results/'

_C.data = CN()
_C.data.path = ''
_C.data.t_len_pct = 1.0
_C.data.size = -1
_C.data.noise = 0.0

_C.train = CN()
_C.train.batch_size = 1024
_C.train.lr = 0.001
_C.train.num_epochs = 50
_C.train.ae_loss_coeff = 1.0
_C.train.pde_loss_coeff = 1.0

_C.system = CN()
_C.system.cpu = False
_C.system.seed = 2
_C.system.num_workers = 32


def get_cfg_defaults(config_file=None, config_list=None):
    cfg = _C.clone()
    if config_file is not None:
        cfg.merge_from_file(config_file)
    if config_list is not None:
        cfg.merge_from_list(config_list)
    cfg.freeze()
    return cfg
