import random

from ref.activatorFunctions import step, identity, hyperbolic_tangent, hyperbolic_tangent_derivative, sigmoid, sigmoid_derivative
from abc import ABC

from ref.normalizers import normalize_output_sigmoid, denormalize_output_sigmoid, normalize_output_tanh, \
    denormalize_output_tanh


def binary_error(y, output):
    return 0 if y == output else 1

def mse_error(y, output):
    return 0.5 * ((y-output)**2)

class SingleLayerPerceptron(ABC):
    def __init__(self, input_size, learning_rate, activator_function, error_function, activator_derivative = lambda x: 1, output_normalizer = lambda x: (x, min(x), max(x)), output_denormalizer = lambda x, x_min, x_max: x, seed = None):
        if seed is not None:
            random.seed(seed)

        self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate

        self.activator_function = activator_function
        self.error_function = error_function
        self.activator_derivative = activator_derivative

        self.output_normalizer = output_normalizer
        self.output_denormalizer = output_denormalizer

        self.error_min = None
        self.best_weights = None
        self.error_min_epoch = None

    def train(self, training_set, expected_outputs, epochs):

        normalized_outputs, min_output, max_output = self.output_normalizer(expected_outputs)

        for epoch in range(epochs):
            error = 0
            for x, y in zip(training_set, normalized_outputs):
                x_with_bias = x + [1]
                h = sum(w * x_i for w, x_i in zip(self.weights, x_with_bias))

                output = self.activator_function(h)
                self.weights = [w + self.learning_rate * (y-output) * self.activator_derivative(h) * x_i for w, x_i in zip(self.weights, x_with_bias)]
                error += self.error_function(self.output_denormalizer(y, min_output, max_output), self.output_denormalizer(output, min_output, max_output))

                #print(f"input: {x}, out: {output}, expected: {y}")

            average_error = error/len(training_set)
            #print(f"epoch {epoch + 1} average error - {average_error}")
            if self.error_min is None or average_error < self.error_min:
                self.error_min = average_error
                self.best_weights = self.weights
                self.error_min_epoch = epoch + 1

        self.weights = self.best_weights


    def test(self, x):
        x_with_bias = x + [1]
        return self.activator_function(sum(w * x_i for w, x_i in zip(self.weights, x_with_bias)))

class StepPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate, seed = None):
        super().__init__(input_size, learning_rate, step, binary_error, seed = seed)

class LinearPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate, seed = None):
        super().__init__(input_size, learning_rate, identity, mse_error, seed = seed)


class NonLinearPerceptron(SingleLayerPerceptron):
    def __init__(self, input_size, learning_rate, non_linear_function, non_linear_derivative, output_normalizer, output_denormalizer, seed = None):
        super().__init__(input_size, learning_rate, non_linear_function, mse_error, non_linear_derivative, output_normalizer, output_denormalizer, seed = seed)

class SigmoidPerceptron(NonLinearPerceptron):
    def __init__(self, input_size, learning_rate, seed=None):
        super().__init__(input_size, learning_rate, sigmoid, sigmoid_derivative, normalize_output_sigmoid, denormalize_output_sigmoid, seed = seed)


class TanhPerceptron(NonLinearPerceptron):
    def __init__(self, input_size, learning_rate, seed=None):
        super().__init__(input_size, learning_rate, hyperbolic_tangent, hyperbolic_tangent_derivative, normalize_output_tanh,
                         denormalize_output_tanh, seed=seed)
