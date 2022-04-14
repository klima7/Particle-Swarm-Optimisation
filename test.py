import numpy as np
from swarm import PSOSolver


def fun(vec):
    x, y = vec
    term1 = (1.5 - x - x * y) ** 2
    term2 = (2.25 - x + x * (y ** 2)) ** 2
    term3 = (2.625 - x + x * (y ** 3)) ** 2
    result = term1 + term2 + term3
    return result


if __name__ == '__main__':
    np.random.seed(42)
    solver = PSOSolver()
    domain = [[-4.5, 4.5], [-4.5, 4.5]]
    solution = solver.solve(fun, domain)
    print(f'Solution: {solution}')
