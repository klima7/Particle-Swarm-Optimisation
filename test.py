import numpy as np
import matplotlib.pyplot as plt
from swarm import PSOSolver


def fun(vec):
    x, y = vec
    term1 = (1.5 - x - x * y) ** 2
    term2 = (2.25 - x + x * (y ** 2)) ** 2
    term3 = (2.625 - x + x * (y ** 3)) ** 2
    result = term1 + term2 + term3
    return result


if __name__ == '__main__':

    fig, ax = plt.subplots()
    ax.axis([-4.5, 4.5, -4.5, 4.5])
    sc = ax.scatter([], [], marker='o')
    ax.set_title('Particles')

    def update_plot(best_arg, best_value, pos):
        x = [p[0] for p in pos]
        y = [p[1] for p in pos]
        sc.set_offsets(np.c_[x, y])
        plt.draw()
        plt.pause(0.05)

    np.random.seed(42)
    solver = PSOSolver()
    domain = [[-4.5, 4.5], [-4.5, 4.5]]
    arg, val = solver.solve(fun, domain, callback=update_plot)
    print(f'Arguments: {arg}')
    print(f'Value    : {val}')
