import csv
import sys
import json
import os
from perceptrons.LinearPerceptron import LinearPerceptron
from perceptrons.SigmoidPerceptron import SigmoidPerceptron
from perceptrons.TanhPerceptron import TanhPerceptron


def load_dataset(dataset):
    with open(dataset, "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = [list(map(float, row)) for row in reader]
    inputs = [row[:-1] for row in data]
    outputs = [row[-1] for row in data]
    return inputs, outputs

perceptrons = {
    "linear": LinearPerceptron,
    "sigmoid": SigmoidPerceptron,
    "tanh": TanhPerceptron
}

if __name__ == "__main__":

    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    inputs, outputs = load_dataset("./TP3-ej2-conjunto.csv")

    seed = None
    if "seed" in config:
        seed = config["seed"]

    learning_rate = config["learning_rate"]
    epochs = config["epochs"]

    perceptron = perceptrons[config["perceptron"]](len(inputs[0]), learning_rate, seed = seed)
    perceptron.train(inputs, outputs, epochs)

    print(f"Min error: {perceptron.error_min}")
    str_dir = "./results/"
    os.makedirs(str_dir, exist_ok=True)
    with open(f"{str_dir}result.csv", "a") as f:
        f.write(f"{config['perceptron']},{learning_rate},{perceptron.error_min_epoch},{perceptron.error_min}\n")