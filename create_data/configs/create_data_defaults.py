from yacs.config import CfgNode as CN

_C = CN()

_C.show_fig = False
_C.create_timestamp = False

_C.system = CN()
_C.system.seed = 2
_C.system.parallel = True
_C.system.h5py = False

_C.data = CN()
_C.data.size = 10000
_C.data.x_len = 20.0
_C.data.t_len = 5.0
_C.data.delta_t = 0.05
_C.data.delta_x = 0.5
_C.data.path = 'data/'
_C.data.noise_coeff = 0.0

_C.data.model_path = ''
_C.data.load_model = True

_C.pde = CN()
_C.pde.type = ''            # Acceptable values 'const' / 'burgers' / 'fn2d'


def get_cfg_defaults(config_file=None, config_list=None):
    cfg = _C.clone()
    if config_file is not None:
        cfg.merge_from_file(config_file)
    if config_list is not None:
        cfg.merge_from_list(config_list)
    cfg.freeze()
    return cfg
