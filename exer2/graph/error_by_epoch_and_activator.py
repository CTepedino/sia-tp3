import csv
import matplotlib.pyplot as plt

def load_dataset(filename):
    epochs = []
    errors = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            epochs.append(int(line[0]))
            errors.append(float(line[1]))
    return epochs, errors

if __name__ == "__main__":
    epochs, errors = load_dataset("../results/linear_avg_err_by_epoch")

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, errors, label="linear")

    epochs, errors = load_dataset("../results/sigmoid_avg_err_by_epoch")
    plt.plot(epochs, errors, label="sigmoid")

    epochs, errors = load_dataset("../results/tanh_avg_err_by_epoch")
    plt.plot(epochs, errors, label="tanh")

    plt.xlabel("Epoch")
    plt.ylabel("Error promedio")

    plt.grid(True)
    plt.tight_layout()

    plt.ylim(0, 50)
    plt.xlim(0, 25000)

    plt.legend()
    plt.show()
