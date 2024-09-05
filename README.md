# CONFIDE: Contextual Finite Differences Modelling of PDEs

This repository is an implementation of the [CONFIDE paper](https://arxiv.org/abs/2303.15827), which is part of the proceedings of KDD 2024.

Example results for signal prediction using (from left to right): (1) Ground truth, (2) CONFIDE, (3) FNO, and (4) UNET (other baselines are worse).

<p align="center">
<img src="https://github.com/orilinial/CONFIDE/blob/master/pde_vid_2_all.gif" width="800">
</p>

### Data creation
To create the datasets used in the paper run:
* Constant coefficient PDE:  `python create_data/create_data.py --config-file create_data/configs/const_pde_default.yaml`
* Burgers' PDE: `python create_data/create_data.py --config-file create_data/configs/burgers_default.yaml`
* FitzHugh-Nagumo PDE:  `python create_data/create_data.py --config-file create_data/configs/fn2d_default.yaml`

The data would be created using default arguments. To view / modify them check the file `create_data/configs/create_data_defaults.py` file, and the corresponding modifications in the YAML files.

### Training
To train the CONFIDE model run: `python src/train_confide_model.py --config-file CONFIG_PATH`
where `CONFIG_PATH` should be changed to the specific required experiment. For example, use `src/configs/burgers_pde/confide.yaml` for the Burgers' equation.

To train baselines:

* CONFIDE-0: `python src/train_confide_0_model.py --config-file CONFIG_PATH`.
* FNO: `python train_fno.py --config-file CONFIG_PATH`.
* UNET: `python train_unet.py --config-file CONFIG_PATH`.
* Latent-ODE (Neural ODE): `python train_latent_ode_model.py --config-file CONFIG_PATH`.
  
### Requirements:
Described in the `requirementes.txt` file. Joblib is used for data creation and neuralop is used speficially for FNO.

### Citing:
Please cite this project when using it:
```
@inproceedings{linial2024confide,
  title={CONFIDE: Contextual Finite Difference Modelling of PDEs},
  author={Linial, Ori and Avner, Orly and Di Castro, Dotan},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1839--1850},
  year={2024}
}
```
