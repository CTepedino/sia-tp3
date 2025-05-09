import copy
import random
from abc import ABC, abstractmethod

class MultiLayerPerceptron(ABC):
    def __init__(self, layers, learning_rate, activator_function, activator_derivative = lambda x: 1):
        self.layers = layers
        self.learning_rate = learning_rate

        self.activator_function = activator_function
        self.activator_derivative = activator_derivative

        self.weights = [
            [[random.uniform(-1, 1) for _ in range(layers[i])] for _ in range(layers[i + 1])]
            for i in range(len(layers) - 1)
        ]

        self.biases = [
            [random.uniform(-1, 1) for _ in range(layers[i + 1])]
            for i in range(len(layers) - 1)
        ]

        self.error_min = None
        self.best_weights = None
        self.best_biases = None

    def forward_propagation(self, input_data):
        activations = [input_data]
        hs = []

        for layer_index in range(len(self.weights)):
            prev_activation = activations[-1]
            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]

            layer_h = []
            layer_activation = []

            for neuron_weights, bias in zip(layer_weights, layer_biases):
                h = sum(w * x for w, x in zip(neuron_weights, prev_activation)) + bias
                a = self.activator_function(h)
                layer_h.append(h)
                layer_activation.append(a)

            hs.append(layer_h)
            activations.append(layer_activation)

        return hs, activations

    @abstractmethod
    def back_propagation(self, expected_output, hs, activations):
        pass

    def train(self, training_set, expected_outputs, epochs):
        for epoch in range(epochs):
            error = 0

            for x, y in zip(training_set, expected_outputs):
                hs, activations = self.forward_propagation(x)
                output = activations[-1]
                self.back_propagation(y, hs, activations)

                error += 0.5 * sum((yt - yp) ** 2 for yt, yp in zip(y, output)) / len(y)

            average_error = error/len(training_set)
            print(f"epoch {epoch + 1} average error - {average_error}")
            if self.error_min is None or average_error < self.error_min:
                self.error_min = average_error
                self.best_weights = copy.deepcopy(self.weights)
                self.best_biases = copy.deepcopy(self.biases)

        self.weights = copy.deepcopy(self.best_weights)
        self.biases = copy.deepcopy(self.best_biases)

    def test(self, input_data):
        hs, activations = self.forward_propagation(input_data)
        return activations[-1]


class DescendingGradientMLP(MultiLayerPerceptron):
    def back_propagation(self, expected_output, hs, activations):
        deltas = [[] for _ in range(len(self.weights))]

        last_layer = len(self.weights) - 1
        deltas[last_layer] = [
            (output - target) * self.activator_derivative(z)
            for output, target, z in zip(activations[-1], expected_output, hs[-1])
        ]

        for l in range(len(self.weights) -2, -1, -1):
            deltas[l] = []
            for i in range(len(self.weights[l])):
                downstream_sum = sum(
                    deltas[l + 1][k] * self.weights[l + 1][k][i]
                    for k in range(len(self.weights[l + 1]))
                )
                delta = downstream_sum * self.activator_derivative(hs[l][i])
                deltas[l].append(delta)

        for l in range(len(self.weights)):
            for i in range(len(self.weights[l])):
                for j in range(len(self.weights[l][i])):
                    gradient = deltas[l][i] * activations[l][j]
                    self.weights[l][i][j] -= self.learning_rate * gradient
                self.biases[l][i] -= self.learning_rate * deltas[l][i]

class MomentumMLP(MultiLayerPerceptron):
    def back_propagation(self, expected_output, hs, activations):
        pass

class AdamMLP(MultiLayerPerceptron):
    def back_propagation(self, expected_output, hs, activations):
        pass

perceptrons = {
    "desdending_gradient": DescendingGradientMLP,
    "momentum": MomentumMLP,
    "adam": AdamMLP
}