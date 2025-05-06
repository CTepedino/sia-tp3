class SingleLayerPerceptron:
    def __init__(self, input_size, learning_rate, activator_function, error_function, weight_update_factor = lambda x: 1):
        self.weights = [0.0 for _ in range(input_size + 1)]
        self.learning_rate = learning_rate

        self.activator_function = activator_function
        self.error_function = error_function
        self.weight_update_factor = weight_update_factor

        self.error_min = None
        self.best_weights = None

    def train(self, training_set, expected_outputs, epochs):
        for epoch in range(epochs):
            error = 0
            for x, y in zip(training_set, expected_outputs):
                x.append(1)
                h = sum(w * x_i for w, x_i in zip(self.weights, x))

                output = self.activator_function(h)
                self.weights = [w + self.learning_rate * (y-output) * self.weight_update_factor(h) * x_i for w, x_i in zip(self.weights, x)]
                error += self.error_function(y, output)

                print(f"input: {x[:-1]}, out: {output}, expected: {y}")

            average_error = error/len(training_set)
            print(f"epoch {epoch + 1} average error - {average_error}")
            if self.error_min is None or average_error < self.error_min:
                self.error_min = average_error
                self.best_weights = self.weights

        self.weights = self.best_weights


    def test(self, x):
        x.append(1)
        return self.activator_function(sum(w * x_i for w, x_i in zip(self.weights, x)))


def step(x):
    return 1 if x >= 0 else -1

def step_error_function(y, output):
    return 0.5 * abs(y - output)

class SingleLayerStepPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate):
        super().__init__(input_size, learning_rate, step, step_error_function)

def identity(x):
    return x

def identity_error_function(y, output):
    return 0.5 * ((y - output) ** 2)

class SingleLayerLinearPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate):
        super().__init__(input_size, learning_rate, identity, identity_error_function)


class SingleLayerNonLinearPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate, non_linear_function, non_linear_derivative):
        def non_linear_error(y, output):
            return 0.5 * ((y - output) ** 2)
        super().__init__(input_size, learning_rate, non_linear_function, non_linear_error, non_linear_derivative)



class MultiLayerPerceptron:
    def __init__(self, layers, learning_rate, activator_function, error_function, weight_update_factor = lambda x: 1):
        self.layers = layers
        self.learning_rate = learning_rate

        self.activator_function = activator_function
        self.error_function = error_function
        self.weight_update_factor = weight_update_factor

        self.weights = [
            [[0.0 for _ in range(layers[i])] for _ in range(layers[i + 1])]
            for i in range(len(layers) - 1)
        ]

        self.biases = [
            [0.0 for _ in range(layers[i + 1])]
            for i in range(len(layers) - 1)
        ]

        self.error_min = None
        self.best_weights = None
        self.best_biases = None

    def forwardPropagation(self, training_set):
        activations = [training_set]

        for layer_index in range(len(self.weights)):
            prev_activation = activations[-1]
            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]
            layer_output = []

            for neuron_weights, bias in zip(layer_weights, layer_biases):
                h = sum(w * x_i for w, x_i in zip(neuron_weights, prev_activation)) + bias
                layer_output.append(self.activator_function(h))

        return activations[-1]


    def train(self, training_set, expected_outputs, epochs):
        for epoch in range(epochs):
            error = 0

            self.forwardPropagation(training_set)



        self.weights = self.best_weights


perceptrons = {
    "step": SingleLayerStepPerceptron,
    "linear": SingleLayerLinearPerceptron,
    "non_linear": SingleLayerNonLinearPerceptron
}



