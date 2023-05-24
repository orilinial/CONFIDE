import torch
import torch.nn as nn
import numpy as np
import pde


class FCAE(nn.Module):
    def __init__(self, input_dim):
        super(FCAE, self).__init__()
        self.latent_dim = 32
        ic_dim = np.prod(input_dim[1:])
        if len(input_dim) > 2:
            self.input_dim = input_dim[0] * input_dim[1] * input_dim[2]
        else:
            self.input_dim = input_dim[0] * input_dim[1]
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, self.latent_dim))

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim + ic_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, self.input_dim))

    def forward(self, x):
        if len(x.shape) > 3:
            input_x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        else:
            input_x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        z = self.encoder_fc(input_x)
        input_z = torch.cat((z, x[:, 0]), dim=1)
        out = self.decoder_fc(input_z)

        if len(x.shape) > 3:
            out = out.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        else:
            out = out.view(x.shape[0], x.shape[1], x.shape[2])
        return out, z


class ContextToRHS(nn.Module):
    def __init__(self, x_dim, latent_dim, hidden_dim=1024):
        input_dim = x_dim + latent_dim
        super(ContextToRHS, self).__init__()
        output_dim = x_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim)
                                 )

    def forward(self, t, f_data, context):
        input_data = torch.cat((f_data, context), dim=1)
        rhs = self.net(input_data)
        return rhs


class RHSModel(nn.Module):
    def __init__(self, input_dim, x_dim):
        super(RHSModel, self).__init__()
        self.ae_model = FCAE(input_dim)
        self.context_to_params_model = ContextToRHS(x_dim, self.ae_model.latent_dim)
        self.loss_func = calc_pde_loss

    def forward(self, t, f_data, sol_context):
        recon_sol, context = self.ae_model(sol_context)
        rhs = self.context_to_params_model(t, f_data, context)
        return recon_sol, rhs


def calc_pde_loss(f_data, delta_t, res, reduce=True):
    # using the EXPLICIT scheme: two rows of the solution for two consecutive time steps
    df_dt = torch.gradient(f_data, dim=1)[0][:, 1] / delta_t

    pde_loss = ((df_dt - res) ** 2)
    if reduce:
        pde_loss = pde_loss.mean()

    return pde_loss


def solve_pde(initial_conditions, delta_t, delta_x, t_len, x_len, model, context):
    grid = pde.CartesianGrid([[0.0, x_len]], x_len // delta_x)
    state = pde.ScalarField(grid, data=initial_conditions)
    bc = [{'value': initial_conditions[0]}, {'value': initial_conditions[-1]}]
    eq = OneDimPDE(bc, model, context)

    storage = pde.MemoryStorage()
    eq.solve(state, t_range=t_len, dt=delta_t/2.0, tracker=storage.tracker(delta_t))

    return np.array(storage.data)


class OneDimPDE(pde.PDEBase):
    def __init__(self, bc, model, context):
        super(OneDimPDE, self).__init__()
        self.bc = bc
        self.model = model
        self.context = torch.FloatTensor(context).unsqueeze(0)

    def evolution_rate(self, state, t=0):
        model_t = torch.FloatTensor([t]).unsqueeze(0)
        state_tensor = torch.FloatTensor(state.data).unsqueeze(0)
        _, dx_dt = self.model(model_t, state_tensor, self.context)
        dx_dt = dx_dt.squeeze().numpy()
        if type(state) == pde.ScalarField:
            dx_dt = pde.ScalarField(state.grid, data=dx_dt)
        if type(state) == pde.VectorField:
            dx_dt = pde.VectorField(state.grid, data=dx_dt)
        return dx_dt
