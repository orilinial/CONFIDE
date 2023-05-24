import sys
import os
import inspect

import pde
import numpy as np

from .default import Sampler

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)
from utils.gp import sample_gp


class BurgersEquationSampler(Sampler):
    def __init__(self, config):
        super(BurgersEquationSampler, self).__init__(config.data.x_len, config.data.t_len, config.data.delta_x,
                                                  config.data.delta_t)
        self.config = config

    def get_initial_conditions(self, x):
        y = np.array([0.0, 0.0])
        _, res = sample_gp(x, y)
        bc_low = 0.0
        bc_high = 0.0

        return res, bc_low, bc_high

    def get_pde(self, x, bc):
        parameters = {'a': np.random.uniform(1.0, 2.0)}

        eq = BurgersPDE(bc, parameters)
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
        x_low = 0.0
        x_high = self.config.data.x_len
        grid = pde.CartesianGrid([(x_low, x_high)], (x_high - x_low) // self.delta_x)

        # Get the initial conditions: f(x, t=0)
        initial_conditions, bc_low, bc_high = self.get_initial_conditions(grid.axes_coords[0])
        state = pde.ScalarField(grid, data=initial_conditions)
        bc = self.get_boundary_conditions(low=bc_low, high=bc_high)
        eq, parameters = self.get_pde(grid.axes_coords[0], bc)

        storage = pde.MemoryStorage()
        eq.solve(state, t_range=self.t_len, dt=self.delta_t / 10.0, tracker=storage.tracker(self.delta_t))

        if show_fig:
            pde.plot_kymograph(storage)
            quit()

        return np.asarray(storage.data)[:-1], grid.axes_coords[0], np.arange(0.0, self.t_len, self.delta_t), parameters


class BurgersPDE(pde.PDEBase):
    def __init__(self, bc, params):
        super(BurgersPDE, self).__init__()
        self.bc = bc
        self.parameters = params

    def evolution_rate(self, state, t=0):
        dx_dt = self.parameters['a'] * state.laplace(self.bc) - state * state.gradient(self.bc).to_scalar()
        return dx_dt
