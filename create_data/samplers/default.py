from abc import abstractmethod, ABC


class Sampler(ABC):
    def __init__(self, x_len, t_len, delta_x, delta_t):
        self.x_len = x_len
        self.t_len = t_len
        self.delta_x = delta_x
        self.delta_t = delta_t

    @staticmethod
    def get_boundary_conditions(low=0.0, high=0.0):
        bc_x_left = {'value': low}
        bc_x_right = {'value': high}
        bc_x = [bc_x_left, bc_x_right]
        return bc_x