import math

import torch
import torch.nn as nn
import numpy as np


class FCAE(nn.Module):
    def __init__(self, input_dim):
        super(FCAE, self).__init__()
        self.latent_dim = 32
        ic_dim = np.prod(input_dim[1:])
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
        input_x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        z = self.encoder_fc(input_x)
        input_z = torch.cat((z, x[:, 0]), dim=1)
        out = self.decoder_fc(input_z)
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


class ContextToSol(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim=200):
        super(ContextToSol, self).__init__()
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
        f_t_1 = self.output_net(input_data).view(f.shape[0], *self.input_dim[1:])
        return f_t_1


class TwoDimModel(nn.Module):
    def __init__(self, input_dim, output_dim, x_dim):
        super(TwoDimModel, self).__init__()
        self.ae_model = ConvAE(input_dim)
        self.context_to_sol_model = ContextToSol(self.ae_model.latent_dim, input_dim)

    def forward(self, t, f, sol_context):
        recon_sol, context = self.ae_model(sol_context)
        f_t_1 = self.context_to_sol_model(t, f, context)
        return recon_sol, f_t_1

    # def forward_multiple_t(self, t, f, sol_context):
    #     recon_sol, context = self.ae_model(sol_context)
    #     context = context.expand(t.shape[0], context.shape[1])
    #     params = self.context_to_params_model(t, context)
    #     return recon_sol, params
