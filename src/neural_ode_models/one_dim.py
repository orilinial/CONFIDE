import math

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np

class NeuralODEfunc(nn.Module):
    def __init__(self, latent_dim, nhidden=200):
        super(NeuralODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out

class RecognitionNet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(RecognitionNet, self).__init__()
        self.fc_input_net = nn.Sequential(
            nn.Linear(input_dim[1], 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim))

    def forward(self, x):
        out = self.fc_input_net(x)
        return out


class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim, nhidden=32):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.i2h = nn.Linear(latent_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=1024):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, math.prod(input_dim[1:]))
                                 )

    def forward(self, z):
        out = self.net(z).view((z.shape[0], z.shape[1], *self.input_dim[1:]))
        return out


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class NeuralODE(nn.Module):
    def __init__(self, input_dim):
        super(NeuralODE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = 32
        self.func = NeuralODEfunc(self.latent_dim)
        self.rec = RecognitionNet(self.input_dim, self.latent_dim)
        self.rec_rnn = RecognitionRNN(self.latent_dim)
        self.dec = Decoder(input_dim, self.latent_dim)

    def forward(self, x, t_tensor, eval=False):
        x = self.rec(x)
        h = torch.zeros((x.shape[0], self.rec_rnn.nhidden)).to(x.device)

        for t in reversed(range(x.shape[1])):
            obs = x[:, t, :]
            out, h = self.rec_rnn.forward(obs, h)

        qz0_mean, qz0_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        if not eval:
            epsilon = torch.randn(qz0_mean.size()).to(x.device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        else:
            z0 = qz0_mean
        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z0, t_tensor, method='dopri5').permute(1, 0, 2)
        pred_x = self.dec(pred_z)
        return pred_x, qz0_mean, qz0_logvar
