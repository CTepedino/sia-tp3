import csv
import sys
import json

from ref.singleLayerPerceptron import LinearPerceptron, NonLinearPerceptron
from ref.activatorFunctions import non_linear_functions

import numpy as np

def load_dataset(dataset):
    data = np.loadtxt(dataset, delimiter=",", skiprows=1)
    inputs = data[:, :-1]
    expected_outputs = data[:, -1]
    return inputs, expected_outputs



if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    inputs, expected_outputs = load_dataset(config["dataset"])

    #normalizo inputs
    inputs_mean = inputs.mean(axis=0)
    inputs_std = inputs.std(axis=0)
    inputs = (inputs - inputs_mean) / inputs_std

    lr = config["learning_rate"]
    match config["perceptron"]:
        case "linear":
            perceptron = LinearPerceptron(len(inputs[0]), lr)
        case "non_linear_sigmoid":
            #normalizo outputs a 0, 1
            expected_outputs = (expected_outputs - np.min(expected_outputs)) / (np.max(expected_outputs) - np.min(expected_outputs))

            function, derivative = non_linear_functions["sigmoid"]
            perceptron = NonLinearPerceptron(len(inputs[0]), lr, function, derivative)
        case "non_linear_tanh":
            #normalizo outputs a -1, 1
            expected_outputs = 2 * (expected_outputs - np.min(expected_outputs)) / (np.max(expected_outputs) - np.min(expected_outputs)) - 1

            function, derivative = non_linear_functions["tanh"]
            perceptron = NonLinearPerceptron(len(inputs[0]), lr, function, derivative)
        case _:
            perceptron = None

    perceptron.train(inputs, expected_outputs, config["epochs"])

    print(f"Best epoch: {perceptron.error_min_epoch} - error: {perceptron.error_min}")



