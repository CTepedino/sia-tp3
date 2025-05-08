import csv
import sys
import json

from ref.normalizers import standarize_input
from ref.singleLayerPerceptron import LinearPerceptron, SigmoidPerceptron, TanhPerceptron
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

    inputs = standarize_input(inputs)

    lr = config["learning_rate"]
    match config["perceptron"]:
        case "linear":
            perceptron = LinearPerceptron(len(inputs[0]), lr)
        case "non_linear_sigmoid":
            function, derivative = non_linear_functions["sigmoid"]
            perceptron = SigmoidPerceptron(len(inputs[0]), lr)
        case "non_linear_tanh":
            function, derivative = non_linear_functions["tanh"]
            perceptron = TanhPerceptron(len(inputs[0]), lr)
        case _:
            perceptron = None

    perceptron.train(inputs, expected_outputs, config["epochs"])

    print(f"Best epoch: {perceptron.error_min_epoch} - error: {perceptron.error_min}")



