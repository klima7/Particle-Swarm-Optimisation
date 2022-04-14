import numpy as np


class PSOSolver:

    def __init__(self, n_particles=100, w=1, v=1, c1=1, c2=1):
        self.n_particles = n_particles
        self.w = np.broadcast_to(w, (n_particles,))
        self.v = np.broadcast_to(v, (n_particles,))
        self.c1 = np.broadcast_to(c1, (n_particles,))
        self.c2 = np.broadcast_to(c2, (n_particles,))

    def solve(self, fun, domain):
        domain = np.array(domain)
        n_dim = domain.shape[0]

        pos = self._random_positions(domain)
        vel = self._random_velocities(n_dim)
        lv = np.repeat(np.inf, self.n_particles)
        lp = np.zeros((self.n_particles, n_dim))
        gv = np.inf
        gp = np.zeros((n_dim,))

        for i in range(100):
            lv, lp, gv, gp = self._get_minimums(lv, lp, gv, gp, fun, pos)

        return 0

    def _random_positions(self, domain):
        dims_pos = []
        for constraint in domain:
            start, end = constraint
            dim_pos = np.random.rand(self.n_particles) * (end - start) + start
            dims_pos.append(dim_pos)
        pos = np.column_stack(dims_pos)
        return pos

    def _random_velocities(self, n_dims):
        directions = np.random.rand(self.n_particles, n_dims)
        lengths = np.sqrt(directions[:, 0]**2 + directions[:, 1]**2)
        scale_factors = self.v / lengths
        scale_factors_transformed = np.repeat(np.expand_dims(scale_factors, axis=1), 2, axis=1)
        scaled_directions = directions * scale_factors_transformed
        return scaled_directions

    def _get_minimums(self, lv, lp, gv, gp, fun, pos):
        # calling function
        values = self._get_function_values(fun, pos)

        # local minimums
        where_better = values < lv
        lv[where_better] = values[where_better]
        lp[where_better] = pos[where_better]

        # global minimum
        min_index = np.argmin(values)
        min_value = values[min_index]
        if min_value < gv:
            gv = min_value
            gp = pos[min_index]

        return lv, lp, gv, gp

    @staticmethod
    def _get_function_values(fun, pos):
        return np.array([fun(args) for args in pos])

