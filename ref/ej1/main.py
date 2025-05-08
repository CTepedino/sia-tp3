import sys
import json
from ref.perceptrons.StepPerceptron import StepPerceptron

if __name__ == "__main__":

    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    training_inputs = config["training_inputs"]
    training_outputs = config["training_outputs"]

    perceptron = StepPerceptron(
        input_size=len(training_inputs[0]),
        learning_rate=config["learning_rate"]
    )

    perceptron.train(training_inputs, training_outputs, config["epochs"])

    test_inputs = []
    test_outputs = []
    if "test_inputs" in config and "test_outputs" in config:
        test_inputs = config["test_inputs"]
        test_outputs = config["test_outputs"]

    for x, y in zip(test_inputs, test_outputs):
        out = perceptron.test(x)
        print(f"TEST: {x} - RESULT - {out} - EXPECTED - {y}")

    out_path = "weights.txt"
    if len(sys.argv) > 2:
        out_path = sys.argv[2]

    with open(out_path, "w") as f:
        for w in perceptron.weights:
            f.write(f"{w}\n")
