import csv
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(filename):
    results = {
        "tanh": {"error": [], "k": []},
        "sigmoid": {"error": [], "k": []}
    }

    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if line[0] == "tanh":
                results["tanh"]["k"].append(float(line[2]))
                results["tanh"]["error"].append(float(line[5]))
                results["sigmoid"]["k"].append(float(line[2]))
                results["sigmoid"]["error"].append(float(line[4]))
    return results

# if __name__ == "__main__":
#
#     results = load_dataset("generalization.csv")
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(results["tanh"]["k"], results["tanh"]["error"], label="tanh")
#     plt.plot(results["sigmoid"]["k"], results["sigmoid"]["error"], label="sigmoid")
#
#     plt.xlabel("K")
#     plt.ylabel("Error promedio de testeo")
#
#     plt.grid(True)
#     plt.tight_layout()
#
#     plt.legend()
#     plt.show()

if __name__ == "__main__":

    results = load_dataset("generalization.csv")

    # Extract and zip together to sort by k
    data = list(zip(results["tanh"]["k"], results["sigmoid"]["error"], results["tanh"]["error"]))
    data.sort(key=lambda x: -x[0])  # Sort by k

    # Unzip the sorted data
    k_values, tanh_errors, sigmoid_errors = zip(*data)

    x = np.arange(len(k_values))  # the label locations
    width = 0.35  # width of the bars

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, tanh_errors, width, label="entrenamiento")
    plt.bar(x + width/2, sigmoid_errors, width, label="testeo")

    plt.xlabel("Learning rate")
    plt.ylabel("Error promedio")
    plt.xticks(x, k_values)  # Set x-tick labels to sorted k values
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.legend()
    plt.show()