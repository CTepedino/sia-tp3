import sys

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from perceptrons.StepPerceptron import StepPerceptron

def differences(xi):
    return int(xi[0] != xi[1])

def transform_dataset(X):
    return [[differences(xi)] for xi in X]

def plot_decision_boundary(X, y, w, epoch, iteration, save_path):
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    plt.clf()
    X_transf = np.array(transform_dataset(X))

    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X_transf[i], 1, c='b', marker='o', label='Clase +1' if i == 0 else "")
        else:
            plt.scatter(X_transf[i], 1, c='r', marker='x', label='Clase -1' if i == 0 else "")

    if w[1] != 0:
        boundary_x = -w[0]/w[1]
        plt.plot([boundary_x, boundary_x], [0, 2], 'k--')

    plt.xlim(-0.5, 2.5)
    plt.ylim(0.5, 1.5)
    plt.title(f'Epoch {epoch + 1}, Iteración {iteration + 1}')
    plt.xlabel('Cantidad de diferencias entre x1 y x2')
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
    save_results = "./exer1/results/result_ex1_xor_"+timestamp
    os.makedirs(save_results)

    inputs = [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ]

    expected_outputs = [-1, 1, 1, -1]

    transformed_inputs = transform_dataset(inputs)

    perceptron = StepPerceptron(1, learning_rate)

    perceptron.train(transformed_inputs, expected_outputs, epochs)

    plot_decision_boundary(inputs , expected_outputs, perceptron.weights, epochs, epochs, save_results)

    for x, tx, y in zip(inputs, transformed_inputs, expected_outputs):
        output = perceptron.test(tx)
        print(f"IN: {x}, EXPECTED: {y}, PREDICTED: {output}")

