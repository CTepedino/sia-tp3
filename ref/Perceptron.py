import copy
import random

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

                print(f"input: {x}, out: {output}, expected: {y}")

            average_error = error/len(training_set)
            print(f"epoch {epoch + 1} average error - {average_error}")
            if self.error_min is None or average_error < self.error_min:
                self.error_min = average_error
                self.best_weights = self.weights

        self.weights = self.best_weights


    def test(self, x):
        x_with_bias = x + [1]
        return self.activator_function(sum(w * x_i for w, x_i in zip(self.weights, x_with_bias)))


def step(x):
    return 1 if x >= 0 else -1

def binary_error(y, output):
    return 0 if y == output else 1

def mse_error(y, output):
    return 0.5 * ((y-output)**2)

class SingleLayerStepPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate):
        super().__init__(input_size, learning_rate, step, binary_error)

def identity(x):
    return x

def identity_error_function(y, output):
    return 0.5 * ((y - output) ** 2)

class SingleLayerLinearPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate):
        super().__init__(input_size, learning_rate, identity, mse_error)


class SingleLayerNonLinearPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate, non_linear_function, non_linear_derivative):
        def non_linear_error(y, output):
            return 0.5 * ((y - output) ** 2)
        super().__init__(input_size, learning_rate, non_linear_function, mse_error, non_linear_derivative)


class MultiLayerPerceptron:
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

    def forwardPropagation(self, input_data):
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

    def backPropagation(self, expected_output, hs, activations):
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

    def train(self, training_set, expected_outputs, epochs):
        for epoch in range(epochs):
            error = 0

            for x, y in zip(training_set, expected_outputs):
                hs, activations = self.forwardPropagation(x)
                output = activations[-1]
                self.backPropagation(y, hs, activations)

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
        hs, activations = self.forwardPropagation(input_data)
        return activations[-1]

perceptrons = {
    "step": SingleLayerStepPerceptron,
    "linear": SingleLayerLinearPerceptron,
    "non_linear": SingleLayerNonLinearPerceptron,
    "multilayer": MultiLayerPerceptron
}

