import numpy as np
import matplotlib.pyplot as plt
from swarm import PSOSolver


def function(vec):
    x, y = vec
    term1 = (1.5 - x - x * y) ** 2
    term2 = (2.25 - x + x * (y ** 2)) ** 2
    term3 = (2.625 - x + x * (y ** 3)) ** 2
    result = term1 + term2 + term3
    return result


def test(solver):

    # function and domain
    fun = function
    domain = np.array([[-4.5, 4.5], [-4.5, 4.5]])

    # plot visualization
    fig, ax = plt.subplots()
    ax.axis(domain.flatten())
    plot_background(ax, fun, domain)
    sc = ax.scatter([], [], marker='o')
    ax.set_title('Particles')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    annotation = plt.annotate(
        "minimum",
        xy=(-4, -4),
        ha="center",
        arrowprops=dict(facecolor='black', shrink=0.1, width=2),
        fontsize=12,
    )

    # update visualization
    def update_plot(best_arg, best_value, pos):
        x = [p[0] for p in pos]
        y = [p[1] for p in pos]
        sc.set_offsets(np.c_[x, y])
        annotation.xy = best_arg
        annotation.set_position(best_arg - np.array([2, -2]))
        plt.draw()
        plt.pause(0.05)

    # solve
    np.random.seed(42)
    arg, val = solver.solve(fun, domain, callback=update_plot)
    print(f'Args: {arg}; Value: {val}')


def plot_background(ax, fun, domain):
    resolution = 100
    space = np.meshgrid(np.linspace(domain[0, 0], domain[0, 1], resolution), np.linspace(domain[1, 0], domain[1, 1], resolution))
    space_stacked = np.dstack([*space]).reshape(-1, 2)
    values = np.array([fun(arg) for arg in space_stacked]).reshape(resolution, resolution)
    ax.contourf(space[0], space[1], values, levels=20, alpha=0.2)


if __name__ == '__main__':
    solver = PSOSolver()
    test(solver)
