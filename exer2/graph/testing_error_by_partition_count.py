import csv
import matplotlib.pyplot as plt

def load_dataset(filename):
    results = {
        "tanh": {"error": [], "k": []},
        "sigmoid": {"error": [], "k": []}
    }

    with open(filename, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            results[line[0]]["k"].append(int(line[3]))
            results[line[0]]["error"].append(float(line[5]))
    return results

if __name__ == "__main__":

    results = load_dataset("generalization.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(results["tanh"]["k"], results["tanh"]["error"], label="tanh")
    plt.plot(results["sigmoid"]["k"], results["sigmoid"]["error"], label="sigmoid")

    plt.xlabel("K")
    plt.ylabel("Error promedio de testeo")

    plt.grid(True)
    plt.tight_layout()

    plt.legend()
    plt.show()