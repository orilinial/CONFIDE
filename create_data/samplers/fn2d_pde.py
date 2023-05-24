import sys
import os
import inspect

import pde
import numpy as np

from .default import Sampler

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from utils.gp import sample_gp_2d


class FNSampler(Sampler):
    """
    FitzHugh-Nagumo PDE:
        Klassen & Troy 1984,
        Stationary wave solutions of a system of reaction-diffusion
        equations derived from the fitzhugh–nagumo equations
    """
    def __init__(self, config):
        super(FNSampler, self).__init__(config.data.x_len, config.data.t_len, config.data.delta_x, config.data.delta_t)
        self.config = config

    def get_initial_conditions(self, grid):
        x, y = grid.axes_coords
        initial_conditions = sample_gp_2d(x, y, size=2)

        u = pde.ScalarField(grid, initial_conditions[0])
        v = pde.ScalarField(grid, initial_conditions[1])
        state = pde.FieldCollection([u, v])

        return state

    def get_pde(self):
        parameters = {'a': 1e-3, # * np.random.uniform(0, 1),
                      'b': 5e-3,
                      'k': np.random.uniform(0, 1)}

        eq = FNEquation(parameters)
        return eq, parameters

    def sample_pde_sol(self, show_fig=False):
        """
        This function samples a solution of the heat-equation PDE:
        du_dt = a * d^2u_dx^2 + b * du_dx + c
        :param show_fig: whether to demonstrate the solution of the PDE or not. If True, then the current run shows the
        figure and stops the current run.
        :return:
        1. The PDE solution as a matrix of T time steps X Input variable size.
        2. Input variable array.
        3. Time steps array.
        4. Numpy array of parameters used to solve the PDE
           (e.g., du_dt = a * d^2u_dx^2 + b * du_dx + c -> returns [a, b, c]
        """

        # Create grid to solve PDE on
        x_low = -self.x_len
        y_low = -self.x_len
        x_high = self.x_len
        y_high = self.x_len

        delta_x = delta_y = self.delta_x
        x_len = x_high - x_low
        y_len = y_high - y_low
        grid = pde.CartesianGrid([(x_low, x_high), (y_low, y_high)], [x_len // delta_x, y_len // delta_y])

        # Get the initial conditions: f(x, t=0)
        state = self.get_initial_conditions(grid)
        eq, parameters = self.get_pde()

        storage = pde.MemoryStorage()
        eq.solve(state, t_range=self.t_len, dt=self.delta_t / 10.0, tracker=storage.tracker(self.delta_t))

        return np.asarray(storage.data)[:-1], grid.cell_coords, np.arange(0.0, self.t_len, self.delta_t), parameters


class FNEquation(pde.PDEBase):
    def __init__(self, params):
        super(FNEquation, self).__init__()
        self.bc = "auto_periodic_neumann"
        self.parameters = params

    def evolution_rate(self, state, t=0):
        u, v = state  # membrane potential and recovery variable
        a = self.parameters['a']
        b = self.parameters['b']
        k = self.parameters['k']

        Ru = u - u**3 - k - v
        Rv = u - v

        du_dt = a * u.laplace(bc=self.bc) + Ru
        dv_dt = b * v.laplace(bc=self.bc) + Rv

        return pde.FieldCollection([du_dt, dv_dt])
