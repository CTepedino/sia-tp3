import sys

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from perceptrons.StepPerceptron import StepPerceptron


def plot_decision_boundary(X, y, w, epoch, iteration, save_path):
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    plt.clf()
    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], c='b', marker='o', label='Clase +1' if i == 0 else "")
        else:
            plt.scatter(X[i, 0], X[i, 1], c='r', marker='x', label='Clase -1' if i == 0 else "")

    x_vals = np.linspace(-2, 2, 100)

    if w[2] != 0:
        y_vals = -(w[1] * x_vals + w[0]) / w[2]
        plt.plot(x_vals, y_vals, 'k-')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(f'Epoch {epoch + 1}, Iteración {iteration + 1}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    learning_rate = config["learning_rate"]
    epochs = config["epochs"]

    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_results = "./exer1/results/result_ex1_and_" + timestamp
    os.makedirs(save_results)

    inputs = [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ]

    expected_outputs = [-1, -1, -1, 1]

    perceptron = StepPerceptron(2, learning_rate)

    perceptron.train(inputs, expected_outputs, epochs)

    plot_decision_boundary(inputs, expected_outputs, perceptron.weights, epochs, epochs, save_results)

    for x, y in zip(inputs, expected_outputs):
        output = perceptron.test(x)
        print(f"IN: {x}, EXPECTED: {y}, PREDICTED: {output}")



