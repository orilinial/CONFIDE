import math

import torch
from torch import nn


class ContextToParams(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim=1024):
        super(ContextToParams, self).__init__()
        output_dim = 1
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

        self.k_linear = nn.Linear(hidden_dim, output_dim)

        self.Rv_net = nn.Sequential(nn.Linear(hidden_dim + math.prod(self.input_dim[1:]), hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, math.prod(self.input_dim[2:])))

    def forward(self, t, f, context):
        input_data = context
        embedding = self.net(input_data)

        Rv = self.Rv_net(torch.cat((embedding, f.view(f.shape[0], -1)), dim=1)).view(f.shape[0], *self.input_dim[2:])
        k = self.k_linear(embedding) * 1e-3
        k = k.unsqueeze(1).expand(embedding.shape[0], *self.input_dim[2:])

        res = torch.stack((k, Rv), dim=3)
        return res


def calc_fn2d_pde_loss(f_data, delta_t, delta_x, res, reduce=True):
    u = f_data[:, :, 0]
    v = f_data[:, :, 1]

    du_dt = torch.gradient(u, dim=1, spacing=delta_t)[0][:, 1]
    du_dx = torch.gradient(u[:, 1], dim=1, spacing=delta_x, edge_order=2)[0]
    d2u_dx2 = torch.gradient(du_dx, dim=1, spacing=delta_x, edge_order=2)[0]
    du_dy = torch.gradient(u[:, 1], dim=2, spacing=delta_x, edge_order=2)[0]
    d2u_dy2 = torch.gradient(du_dy, dim=2, spacing=delta_x, edge_order=2)[0]
    laplacian_u = d2u_dx2 + d2u_dy2

    dv_dt = torch.gradient(v, dim=1, spacing=delta_t)[0][:, 1]  # / delta_t
    dv_dx = torch.gradient(v[:, 1], dim=1, spacing=delta_x, edge_order=2)[0]
    d2v_dx2 = torch.gradient(dv_dx, dim=1, spacing=delta_x, edge_order=2)[0]
    dv_dy = torch.gradient(v[:, 1], dim=2, spacing=delta_x, edge_order=2)[0]
    d2v_dy2 = torch.gradient(dv_dy, dim=2, spacing=delta_x, edge_order=2)[0]
    laplacian_v = d2v_dy2 + d2v_dx2

    a = 1e-3
    b = 5e-3

    k = res[..., 0]
    Rv = res[..., 1]
    Ru = u[:, 1] - u[:, 1] ** 3 - k - v[:, 1]

    du_dt_rhs = a * laplacian_u + Ru
    dv_dt_rhs = b * laplacian_v + Rv

    pde_loss_u = (du_dt - du_dt_rhs) ** 2
    pde_loss_v = (dv_dt - dv_dt_rhs) ** 2
    pde_loss = pde_loss_u + pde_loss_v
    if reduce:
        pde_loss = pde_loss.mean()

    return pde_loss
