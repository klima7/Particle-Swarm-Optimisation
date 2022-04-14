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
        x = self._random_positions(domain)
        v = self._random_velocities(domain.ndim)
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
