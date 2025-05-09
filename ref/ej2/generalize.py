import csv
import json
import sys
import random

from ref.perceptrons.LinearPerceptron import LinearPerceptron
from ref.perceptrons.SigmoidPerceptron import SigmoidPerceptron
from ref.perceptrons.TanhPerceptron import TanhPerceptron


def load_dataset(dataset):
    with open(dataset, "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = [list(map(float, row)) for row in reader]
    inputs = [row[:-1] for row in data]
    outputs = [row[-1] for row in data]
    return inputs, outputs

def k_fold_split(inputs, outputs, k, seed=None):
    if seed is not None:
        random.seed(seed)

    data = list(zip(inputs, outputs))
    random.shuffle(data)

    fold_size = len(data) // k
    folds = [data[i*fold_size : (i+1)*fold_size] for i in range(k-1)]
    folds.append(data[(k-1)*fold_size:])

    return folds

perceptrons = {
    "linear": LinearPerceptron,
    "sigmoid": SigmoidPerceptron,
    "tanh": TanhPerceptron
}

def mse(expected, predicted):
    return 0.5 * ((expected - predicted) ** 2)

if __name__ == "__main__":
    inputs, outputs = load_dataset("TP3-ej2-conjunto.csv")

    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    seed = None
    if "seed" in config:
        seed = config["seed"]

    k = config["partition_count"]

    learning_rate = config["learning_rate"]
    epochs = config["epochs"]

    folds = k_fold_split(inputs, outputs, k, seed)
    errors = []

    with open("generalizations.csv", "a") as f:
        train_error = 0

        for i in range(k):
            test_fold = folds[i]
            train_folds = [sample for j, fold in enumerate(folds) if j != i for sample in fold]

            train_inputs = [x for x, y in train_folds]
            train_outputs = [y for x, y in train_folds]
            test_inputs = [x for x, y in test_fold]
            test_outputs = [y for x, y in test_fold]



            perceptron = perceptrons[config["perceptron"]](len(inputs[0]), learning_rate, seed=seed)
            perceptron.train(train_inputs, train_outputs, epochs)
            print(f"Training error: {perceptron.error_min}")
            train_error += perceptron.error_min

            fold_error = 0
            for x, y in zip(test_inputs, test_outputs):
                predicted = perceptron.test(x)
                fold_error += perceptron.error(y, predicted)
            fold_error /= len(test_fold)

            errors.append(fold_error)
            print(f"Fold {i + 1}: MSE = {fold_error:.4f}")


        train_error = train_error / k
        test_error = sum(errors) / k
        print(f"\nAverage MSE over {k} folds: {test_error:.4f}")

        f.write(f"{config['perceptron']},{learning_rate},{epochs},{k},{train_error},{test_error}")


