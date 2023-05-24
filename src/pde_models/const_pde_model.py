import torch
from torch import nn


class ContextToParams(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim=1024):
        super(ContextToParams, self).__init__()
        output_dim = 1
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
                                 )

        self.a_linear = nn.Linear(hidden_dim, output_dim)
        self.b_linear = nn.Linear(hidden_dim, output_dim)
        self.c_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, t, f, context):
        input_data = context
        embedding = self.net(input_data)

        a = self.a_linear(embedding)
        b = self.b_linear(embedding)
        c = self.c_linear(embedding)

        a = a.expand(embedding.shape[0], 40)
        b = b.expand(embedding.shape[0], 40)
        c = c.expand(embedding.shape[0], 40)

        res = torch.stack((a, b, c), dim=2)
        return res


def calc_const_pde_loss(f_data, delta_t, delta_x, res, reduce=True):
    a = res[..., 0]
    b = res[..., 1]
    c = res[..., 2]

    df_dt = torch.gradient(f_data, dim=1)[0][:, 1] / delta_t
    df_dx = torch.gradient(f_data[:, 1], dim=1)[0] / delta_x
    d2f_dx2 = torch.gradient(df_dx, dim=1)[0] / delta_x

    rhs = a * d2f_dx2 + b * df_dx + c
    pde_loss = ((df_dt - rhs) ** 2)
    if reduce:
        pde_loss = pde_loss.mean()

    return pde_loss

