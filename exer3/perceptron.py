import copy
import random
import os
from datetime import datetime
import numpy as np

from activatorFunctions import step, identity


def binary_error(y, output):
    return 0 if y == output else 1

def mse_error(y, output):
    return 0.5 * ((y-output)**2)

class SingleLayerPerceptron:
    def __init__(self, input_size, learning_rate, activator_function, error_function, activator_derivative = lambda x: 1):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate

        self.activator_function = activator_function
        self.error_function = error_function
        self.activator_derivative = activator_derivative

        self.error_min = None
        self.best_weights = None

    def train(self, training_set, expected_outputs, epochs):
        for epoch in range(epochs):
            error = 0
            for x, y in zip(training_set, expected_outputs):
                x_with_bias = x + [1]
                h = sum(w * x_i for w, x_i in zip(self.weights, x_with_bias))

                output = self.activator_function(h)
                self.weights = [w + self.learning_rate * (y-output) * self.activator_derivative(h) * x_i for w, x_i in zip(self.weights, x_with_bias)]
                error += self.error_function(y, output)

                # print(f"input: {x[:-1]}, out: {output}, expected: {y}")

            average_error = error/len(training_set)
            # print(f"epoch {epoch + 1} average error - {average_error}")
            if self.error_min is None or average_error < self.error_min:
                self.error_min = average_error
                self.best_weights = self.weights

        self.weights = self.best_weights


    def test(self, x):
        x_with_bias = x + [1]
        return self.activator_function(sum(w * x_i for w, x_i in zip(self.weights, x_with_bias)))

class SingleLayerStepPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate):
        super().__init__(input_size, learning_rate, step, binary_error)

class SingleLayerLinearPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate):
        super().__init__(input_size, learning_rate, identity, mse_error)


class SingleLayerNonLinearPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate, non_linear_function, non_linear_derivative):
        super().__init__(input_size, learning_rate, non_linear_function, mse_error, non_linear_derivative)


class MultiLayerPerceptron:
    def __init__(self, layers, learning_rate, activator_function, activator_derivative = lambda x: 1):
        self.layers = layers
        self.learning_rate = learning_rate
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        results_directory = "results/results_ex3_mlp_" + timestamp
        os.makedirs(results_directory, exist_ok=True)
        self.results_directory = results_directory 
        self.results_file = os.path.join(results_directory, "results.txt")

        self.activator_function = activator_function
        self.activator_derivative = activator_derivative

        # Inicialización de Xavier/Glorot para los pesos
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (layers[i] + layers[i + 1]))
            layer_weights = np.random.normal(0, scale, (layers[i + 1], layers[i]))
            layer_biases = np.random.normal(0, scale, layers[i + 1])
            
            self.weights.append(layer_weights.tolist())
            self.biases.append(layer_biases.tolist())

        self.error_min = None
        self.best_weights = None
        self.best_biases = None
        self.patience = 10  # Para early stopping
        self.patience_counter = 0

    def forward_propagation(self, input_data):
        activations = [np.array(input_data)]
        hs = []

        for layer_index in range(len(self.weights)):
            prev_activation = activations[-1]
            layer_weights = np.array(self.weights[layer_index])
            layer_biases = np.array(self.biases[layer_index])

            h = np.dot(layer_weights, prev_activation) + layer_biases
            a = np.array([self.activator_function(x) for x in h])

            hs.append(h)
            activations.append(a)

        return hs, activations

    def back_propagation(self, expected_output, hs, activations):
        deltas = [None] * len(self.weights)
        expected_output = np.array(expected_output)

        # Delta de la última capa
        last_layer = len(self.weights) - 1
        output_error = activations[-1] - expected_output
        deltas[last_layer] = output_error * np.array([self.activator_derivative(z) for z in hs[-1]])

        # Backpropagation
        for l in range(len(self.weights) - 2, -1, -1):
            weights_next = np.array(self.weights[l + 1])
            delta_next = deltas[l + 1]
            h_current = hs[l]
            
            delta = np.dot(weights_next.T, delta_next) * np.array([self.activator_derivative(z) for z in h_current])
            deltas[l] = delta

        # Actualización de pesos y biases
        for l in range(len(self.weights)):
            delta = deltas[l]
            activation = activations[l]
            
            # Actualizar pesos
            weight_gradients = np.outer(delta, activation)
            self.weights[l] = np.array(self.weights[l]) - self.learning_rate * weight_gradients
            
            # Actualizar biases
            self.biases[l] = np.array(self.biases[l]) - self.learning_rate * delta

    def train(self, training_set, expected_outputs, epochs):
        best_error = float('inf')
        
        for epoch in range(epochs):
            error = 0
            np.random.seed(epoch)  # Para reproducibilidad
            indices = np.random.permutation(len(training_set))
            
            for idx in indices:
                x = training_set[idx]
                y = expected_outputs[idx]
                
                hs, activations = self.forward_propagation(x)
                output = activations[-1]
                self.back_propagation(y, hs, activations)

                error += 0.5 * np.sum((np.array(y) - output) ** 2)

            average_error = error / len(training_set)
            
            # Early stopping
            if average_error < best_error:
                best_error = average_error
                self.best_weights = copy.deepcopy(self.weights)
                self.best_biases = copy.deepcopy(self.biases)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping en época {epoch + 1}")
                    break

            # Escribir en archivo
            with open(self.results_file, "a") as f:
                log_line = f"epoch {epoch + 1} average error - {average_error}\n"
                f.write(log_line)
                if (epoch + 1) % 100 == 0:
                    print(log_line.strip())

        self.weights = copy.deepcopy(self.best_weights)
        self.biases = copy.deepcopy(self.best_biases)

    def test(self, input_data):
        hs, activations = self.forward_propagation(input_data)
        return activations[-1].tolist()

perceptrons = {
    "step": SingleLayerStepPerceptron,
    "linear": SingleLayerLinearPerceptron,
    "non_linear": SingleLayerNonLinearPerceptron,
    "multilayer": MultiLayerPerceptron
}

