class SingleLayerPerceptron:
    def __init__(self, input_size, learning_rate, activator_function, error_function, weight_update_factor = lambda x: 1):
        self.weights = [0.0 for _ in range(input_size)]
        self.learning_rate = learning_rate
        self.bias = 0

        self.activator_function = activator_function
        self.error_function = error_function
        self.weight_update_factor = weight_update_factor

        self.error_min = None
        self.best_weights = None
        self.best_bias = None

    def train(self, training_set, expected_outputs, epochs):
        for epoch in range(epochs):
            error = 0
            for x, y in zip(training_set, expected_outputs):
                h = sum(w * x_i for w, x_i in zip(self.weights, x)) + self.bias

                output = self.activator_function(h)

                self.weights = [w + self.learning_rate * (y-output) * self.weight_update_factor(h) * x_i for w, x_i in zip(self.weights, x)]
                self.bias += self.learning_rate * (y-output) * self.weight_update_factor(h)
                error += self.error_function(y, output)

                print(f"input: {x}, out: {output}, expected: {y}")

            average_error = error/len(training_set)
            print(f"epoch {epoch + 1} average error - {average_error}")
            if self.error_min is None or average_error < self.error_min:
                self.error_min = average_error
                self.best_weights = self.weights
                self.best_bias = self.bias

        self.weights = self.best_weights
        self.bias = self.best_bias



    def test(self, x):
        return self.activator_function(sum(w * x_i for w, x_i in zip(self.weights, x)) + self.bias)


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

perceptrons = {
    "step": SingleLayerStepPerceptron,
    "linear": SingleLayerLinearPerceptron,
    "non_linear": SingleLayerNonLinearPerceptron
}



