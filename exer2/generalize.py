import csv
import json
import sys
import random
import os
import numpy as np

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
    inputs, outputs = load_dataset("./exer2/TP3-ej2-conjunto.csv")

    output_max = max(outputs)
    output_min = min(outputs)

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
    str_dir = "./results/"
    os.makedirs(str_dir, exist_ok=True)

    with open(f"{str_dir}generalizations.csv", "a") as f:
        train_sum = 0
        test_sum = 0

        for _ in range(100):
            fold_errors = []
            train_error = 0

            for i in range(k):
                test_fold = folds[i]
                train_folds = [sample for j, fold in enumerate(folds) if j != i for sample in fold]

                train_inputs = [x for x, y in train_folds]
                train_outputs = [y for x, y in train_folds]
                test_inputs = [x for x, y in test_fold]
                test_outputs = [y for x, y in test_fold]



                perceptron = perceptrons[config["perceptron"]](len(inputs[0]), learning_rate, seed=seed)
                perceptron.set_bounds(output_min, output_max)
                perceptron.train(train_inputs, train_outputs, epochs)
                print(f"Training error: {perceptron.error_min}")
                train_error += perceptron.error_min

                fold_error = 0
                for x, y in zip(test_inputs, test_outputs):
                    predicted = perceptron.test(x)
                    fold_error += mse(y, predicted)
                    print(f"{x} - real: {y} - predicted: {predicted}")
                fold_error /= len(test_fold)

                fold_errors.append(fold_error)
                print(f"Fold {i + 1}: MSE = {fold_error:.4f}")


            train_error = train_error / k
            test_error = sum(fold_errors) / k

            train_sum += train_error

            errors.append(test_error)


            print(f"\nAverage MSE over {k} folds: {test_error:.4f}")



        std_mse = np.std(errors, ddof=1)
        test_mean = np.mean(errors)


        f.write(f"{config['perceptron']},{learning_rate},{epochs},{k},{train_sum/100},{test_mean},{std_mse}\n")


