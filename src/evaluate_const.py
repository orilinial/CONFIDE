import numpy as np
import torch
import pde


def solve_pde(initial_conditions, delta_t, delta_x, t_len, x_len, model, context, pde_type):
    grid = pde.CartesianGrid([[0.0, x_len]], x_len // delta_x)
    state = pde.ScalarField(grid, data=initial_conditions)
    bc = [{'value': initial_conditions[0]}, {'value': initial_conditions[-1]}]

    eq = get_pde_class(pde_type)(bc, grid.axes_coords[0], model, context)

    storage = pde.MemoryStorage()
    eq.solve(state, t_range=t_len, dt=delta_t/2.0, tracker=storage.tracker(delta_t))

    times = storage.times
    t = torch.FloatTensor(times[:-1]).unsqueeze(1)
    f = torch.FloatTensor(np.array(storage.data[:-1]))
    _, parameters = model.forward_multiple_t(t, f, torch.FloatTensor(context).unsqueeze(0))
    parameters = parameters.squeeze().numpy()

    return np.array(storage.data), parameters


def get_pde_class(pde_type):
    if pde_type == 'const':
        return ConstPDE
    elif pde_type == 'burgers':
        return BurgersPDE
    else:
        raise ValueError(f'pde_type should be [const/burgers] but got {pde_type}.')


class ConstPDE(pde.PDEBase):
    def __init__(self, bc, x, model, context):
        super(ConstPDE, self).__init__()
        self.bc = bc
        self.x = torch.FloatTensor(x).unsqueeze(1)
        self.context = torch.FloatTensor(context).unsqueeze(0)
        self.model = model
        self.j = 0

    def evolution_rate(self, state, t=0):
        model_t = torch.FloatTensor([t]).unsqueeze(0)
        _, parameters = self.model(model_t, self.context)
        a = parameters[..., 0].squeeze().numpy()
        b = parameters[..., 1].squeeze().numpy()
        c = parameters[..., 2].squeeze().numpy()

        dx_dt = a * state.laplace(self.bc) + b * state.gradient(self.bc).to_scalar() + c
        return dx_dt


class BurgersPDE(pde.PDEBase):
    def __init__(self, bc, x, model, context):
        super(BurgersPDE, self).__init__()
        self.bc = bc
        self.x = torch.FloatTensor(x).unsqueeze(1)
        self.context = torch.FloatTensor(context).unsqueeze(0)
        self.model = model
        self.j = 0

    def evolution_rate(self, state, t=0):
        t_tensor = torch.FloatTensor([t]).unsqueeze(0)
        state_tensor = torch.FloatTensor(state.data).unsqueeze(0)
        _, parameters = self.model(t_tensor, state_tensor, self.context)
        a = parameters[..., 0].squeeze().numpy()
        b = parameters[..., 1].squeeze().numpy()

        dx_dt = a * state.laplace(self.bc) + b * state.gradient(self.bc).to_scalar()
        return dx_dt
