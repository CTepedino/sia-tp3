import json
import sys

from ref.Perceptron import perceptrons

if __name__ == "__main__":

    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    training_inputs = config["training_inputs"]
    training_outputs = config["training_outputs"]



    perceptron = perceptrons[config["perceptron"]](len(training_inputs[0]), config["learning_rate"])
    perceptron.train(training_inputs, training_outputs, config["epochs"])

    test_inputs = []
    if "test_inputs" in config:
        test_inputs = config["test_inputs"]
    test_outputs = []
    if "test_outputs" in config:
        test_outputs = config["test_outputs"]

    for x, y in zip(test_inputs, test_outputs):
        out = perceptron.test(x)
        print(f"TEST: {x} - RESULT - {out} - EXPECTED - {y}")


