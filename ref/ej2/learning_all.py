import json
import sys

import numpy as np

from ref.normalizers import standarize_input, normalize_output_sigmoid, normalize_output_tanh
from ref.singleLayerPerceptron import LinearPerceptron, SigmoidPerceptron, TanhPerceptron


def load_dataset(dataset):
    data = np.loadtxt(dataset, delimiter=",", skiprows=1)
    inputs = data[:, :-1]
    expected_outputs = data[:, -1]
    return inputs, expected_outputs


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    inputs, expected_outputs = load_dataset(config["dataset"])

    seed = config["seed"]
    learning_rates = config["learning_rates"]
    epochs = config["epochs"]

    out_file = "results.csv"

    inputs = standarize_input(inputs)
    sigmoid_expected_outputs = normalize_output_sigmoid(expected_outputs)
    tanh_expected_outputs = normalize_output_tanh(expected_outputs)

    with open(out_file, "w") as f:
        for lr in learning_rates:
            linear = LinearPerceptron(len(inputs[0]), lr, seed=seed)
            linear.train(inputs, expected_outputs, epochs)
            f.write(f"linear,{lr},{linear.error_min_epoch},{linear.error_min}\n")

            sigmoid = SigmoidPerceptron(len(inputs[0]), lr, seed=seed)
            sigmoid.train(inputs, sigmoid_expected_outputs, epochs)
            f.write(f"sigmoid,{lr},{sigmoid.error_min_epoch},{sigmoid.error_min}\n")

            tanh = TanhPerceptron(len(inputs[0]), lr, seed=seed)
            tanh.train(inputs, tanh_expected_outputs, epochs)
            f.write(f"tanh,{lr},{tanh.error_min_epoch},{tanh.error_min}\n")







