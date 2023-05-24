import math

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


class ConvAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        # In the (x,y),(u,v) case: input_dim = (50, 2, 32, 32)
        super(ConvAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim[1], 16, 3, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 16, 3, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            # nn.Conv2d(16, 16, 3, stride=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, stride=1)
        )

        ### Linear net
        self.fc_net = FCAE(input_dim=(input_dim[0], 16 * 2 * 2))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 3, stride=2),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(16, input_dim[1], 2, stride=3),  # b, 1, 28, 28
        )

    def forward(self, x):
        # Input shape: B, T, U, X, Y; where U is the PDE variable dim, and X, Y are the 2D spatial variables
        # Conv - Encode
        x_flat = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        z = self.encoder(x_flat)  # Shape = BT x 16 x 2 x 2

        # FC net
        z_flat = z.view(x.shape[0], x.shape[1], z.shape[1] * z.shape[2] * z.shape[3])  # Shape = B x T x LatentDim
        out, latent = self.fc_net(z_flat)
        out = out.view(*z.shape)

        # Conv - decode
        out = self.decoder(out)
        out = out.view(*x.shape)
        return out, latent


class ContextToRHS(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim=1024):
        super(ContextToRHS, self).__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim)
                                 )

        self.output_net = nn.Sequential(nn.Linear(hidden_dim + math.prod(self.input_dim[1:]), hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, math.prod(self.input_dim[1:])))


    def forward(self, t, f, context):
        embedding = self.net(context)
        input_data = torch.cat((f.view(f.shape[0], -1), embedding), dim=1)
        rhs = self.output_net(input_data).view(f.shape[0], *self.input_dim[1:])
        return rhs


class RHSModel(nn.Module):
    def __init__(self, input_dim, x_dim):
        super(RHSModel, self).__init__()
        self.ae_model = ConvAE(input_dim)
        self.context_to_params_model = ContextToRHS(self.ae_model.latent_dim, input_dim)
        self.loss_func = calc_pde_loss

    def forward(self, t, f_data, sol_context):
        recon_sol, context = self.ae_model(sol_context)
        rhs = self.context_to_params_model(t, f_data, context)
        return recon_sol, rhs


def calc_pde_loss(f_data, delta_t, res, reduce=True):
    # using the EXPLICIT scheme: two rows of the solution for two consecutive time steps
    # df_dt = torch.gradient(f_data, dim=1)[0][:, 1] / delta_t

    u = f_data[:, :, 0]
    v = f_data[:, :, 1]

    du_dt = torch.gradient(u, dim=1, spacing=delta_t)[0][:, 1]
    dv_dt = torch.gradient(v, dim=1, spacing=delta_t)[0][:, 1]

    df_dt = torch.stack((du_dt, dv_dt), dim=1)

    pde_loss = ((df_dt - res) ** 2)
    if reduce:
        pde_loss = pde_loss.mean()

    return pde_loss


def solve_pde(delta_t, delta_x, t_len, x_len, model, context):
    x_low = -x_len
    y_low = -x_len
    x_high = x_len
    y_high = x_len

    delta_y = delta_x
    x_len = x_high - x_low
    y_len = y_high - y_low
    grid = pde.CartesianGrid([(x_low, x_high), (y_low, y_high)], [x_len // delta_x, y_len // delta_y])

    state = pde.FieldCollection([
        pde.ScalarField(grid, data=context[0, 0]),
        pde.ScalarField(grid, data=context[0, 1])
    ])

    eq = TwoDimPDE(model, context)

    storage = pde.MemoryStorage()
    eq.solve(state, t_range=t_len, dt=delta_t / 10.0, tracker=storage.tracker(delta_t))


    return np.array(storage.data)


class TwoDimPDE(pde.PDEBase):
    def __init__(self, model, context):
        super(TwoDimPDE, self).__init__()
        self.bc = 'auto_periodic_neumann'
        self.model = model
        self.context = torch.FloatTensor(context).unsqueeze(0)

    def evolution_rate(self, state, t=0):
        model_t = torch.FloatTensor([t]).unsqueeze(0)
        state_tensor = torch.FloatTensor(state.data).unsqueeze(0)
        _, dx_dt = self.model(model_t, state_tensor, self.context)
        dx_dt = dx_dt.squeeze().numpy()
        if type(state) == pde.ScalarField:
            dx_dt = pde.ScalarField(state.grid, data=dx_dt)
        if type(state) == pde.FieldCollection:
            dx_dt = pde.FieldCollection([pde.ScalarField(state.grid, data=dx_dt[i]) for i in range(dx_dt.shape[0])])
        return dx_dt
