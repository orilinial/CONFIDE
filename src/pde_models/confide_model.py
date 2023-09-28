import torch
import torch.nn as nn
import numpy as np



class FCAE(nn.Module):
    def __init__(self, input_dim, use_ic_in_decoder=True):
        super(FCAE, self).__init__()
        self.use_ic_in_decoder = use_ic_in_decoder
        self.latent_dim = 32
        ic_dim = np.prod(input_dim[1:])
        self.input_dim = np.prod(input_dim)

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

        decoder_input_dim = self.latent_dim + ic_dim if use_ic_in_decoder else self.latent_dim
        self.decoder_fc = nn.Sequential(
            nn.Linear(decoder_input_dim, 128),
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
        input_x = x.view(x.shape[0], -1)
        z = self.encoder_fc(input_x)

        if self.use_ic_in_decoder:
            ic = x[:, 0].view(x.shape[0], -1)
            input_z = torch.cat((z, ic), dim=1)
        else:
            input_z = z

        out = self.decoder_fc(input_z)

        out = out.view(*x.shape)
        return out, z


class ConvAE(nn.Module):
    def __init__(self, input_dim, use_ic_in_decoder=True):
        super(ConvAE, self).__init__()
        self.use_ic_in_decoder = use_ic_in_decoder

        self.latent_dim = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim[1], 16, 3, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 16, 3, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
        )

        self.fc_net = FCAE(input_dim=(input_dim[0], 16 * 2 * 2), use_ic_in_decoder=use_ic_in_decoder)

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
        z = self.encoder(x_flat)

        # FC net
        z_flat = z.view(x.shape[0], x.shape[1], z.shape[1] * z.shape[2] * z.shape[3])  # Shape = B x T x LatentDim
        out, latent = self.fc_net(z_flat)
        out = out.view(*z.shape)

        # Conv - decode
        out = self.decoder(out)
        out = out.view(*x.shape)
        return out, latent


class Confide(nn.Module):
    def __init__(self, input_dim, pde_type, use_ic_in_decoder=True):
        super(Confide, self).__init__()
        self.ae_model = get_ae_model(pde_type)(input_dim, use_ic_in_decoder)
        self.context_to_params_model = get_context_to_params_model(pde_type)(self.ae_model.latent_dim, input_dim)
        self.loss_func = get_pde_loss(pde_type)

    def forward(self, t, f, sol_context):
        recon_sol, context = self.ae_model(sol_context)
        params = self.context_to_params_model(t, f, context)
        return recon_sol, params

    def forward_multiple_t(self, t, f, sol_context):
        recon_sol, context = self.ae_model(sol_context)
        context = context.expand(t.shape[0], context.shape[1])
        params = self.context_to_params_model(t, f, context)
        return recon_sol, params


def get_ae_model(pde_type):
    if pde_type == 'const':
        return FCAE
    elif pde_type == 'burgers':
        return FCAE
    elif pde_type == 'fn2d':
        return ConvAE
    elif pde_type == 'fn2d_u':
        return ConvAE
    else:
        raise ValueError('pde_type should be [const/burgers/fn2d/fn2d_u]')


def get_context_to_params_model(pde_type):
    if pde_type == 'const':
        from .const_pde_model import ContextToParams
    elif pde_type == 'burgers':
        from .burgers_model import ContextToParams
    elif pde_type == 'fn2d_u':
        from .fn2d_u_model import ContextToParams
    elif pde_type == 'fn2d':
        from .fn2d_model import ContextToParams
    else:
        raise ValueError('pde_type should be [const/burger/fn2d/fn2d_u]')

    return ContextToParams



def get_pde_loss(pde_type):
    if pde_type == 'const':
        from .const_pde_model import calc_const_pde_loss as calc_pde_loss
    elif pde_type == 'burgers':
        from .burgers_model import calc_burgers_pde_loss as calc_pde_loss
    elif pde_type == 'fn2d_u':
        from .fn2d_u_model import calc_fn2d_pde_loss as calc_pde_loss
    elif pde_type == 'fn2d':
        from .fn2d_model import calc_fn2d_pde_loss as calc_pde_loss
    else:
        raise ValueError('pde_type should be [const/burger/fn2d/fn2d_u]')

    return calc_pde_loss
