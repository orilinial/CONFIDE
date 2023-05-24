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


class ContextToSol(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=200):
        super(ContextToSol, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim),
                                 )

    def forward(self, t, f, context):
        input_data = torch.cat((t, f, context), dim=1)
        out = self.net(input_data)
        return out


class OneDimModel(nn.Module):
    def __init__(self, input_dim, output_dim, x_dim):
        super(OneDimModel, self).__init__()
        self.ae_model = FCAE(input_dim)
        context_to_sol_input_dim = x_dim + 1 + self.ae_model.latent_dim
        self.context_to_sol_model = ContextToSol(context_to_sol_input_dim, output_dim)

    def forward(self, t, f, sol_context):
        recon_sol, context = self.ae_model(sol_context)
        f_t_1 = self.context_to_sol_model(t, f, context)
        return recon_sol, f_t_1

    # def forward_multiple_t(self, t, f, sol_context):
    #     recon_sol, context = self.ae_model(sol_context)
    #     context = context.expand(t.shape[0], context.shape[1])
    #     params = self.context_to_params_model(t, context)
    #     return recon_sol, params
