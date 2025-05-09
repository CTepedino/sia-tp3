import sys

import numpy as np
from matplotlib import pyplot as plt


def step(x):
    return 1 if x >= 0 else -1

if __name__ == "__main__":

    points = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    #and
    #expected_outs = [-1, -1, -1, 1]

    #xor
    expected_outs = [-1, 1, 1, -1]

    weights_file = "../weights.txt"
    if len(sys.argv) > 1:
        weights_file = sys.argv[1]

    with open(weights_file, "r") as f:
        w1 = float(f.readline().strip())
        w2 = float(f.readline().strip())
        b = float(f.readline().strip())

    x_vals = np.linspace(-2, 2, 100)
    if w2 != 0:
        y_vals = -(w1 / w2) * x_vals - (b / w2)
        plt.plot(x_vals, y_vals, 'b')

    for (x1, x2), y in zip(points, expected_outs):
        color = 'green' if y==1 else 'red'
        plt.scatter(x1, x2, s=50, color=color)

    plt.grid(True)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.show()