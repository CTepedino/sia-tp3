from perceptrons.SingleLayerPerceptron import SingleLayerPerceptron
from abc import ABC, abstractmethod

class NonLinearPerceptron(SingleLayerPerceptron, ABC):
    def __init__(self, input_size, learning_rate, seed = None):
        super().__init__(input_size, learning_rate, seed = seed)
        self.min_output = None
        self.max_output = None

    def set_bounds(self, min, max):
        self.min_output = min
        self.max_output = max

    @abstractmethod
    def activator(self, h):
        pass

    @abstractmethod
    def activator_derivative(self, h):
        pass

    @abstractmethod
    def normalize_output(self, output):
        pass

    def normalize_outputs(self, outputs):
        if self.min_output is None and self.max_output is None:
            self.min_output = min(outputs)
            self.max_output = max(outputs)
        return [self.normalize_output(output) for output in outputs]

    @abstractmethod
    def denormalize_output(self, normalized_output):
        pass

    def weight_update(self, w, x, y, output, h):
        return w + self.learning_rate * (y - output) * x * self.activator_derivative(h)

    def error(self, y, output):
        denormalized_y = self.denormalize_output(y)
        denormalized_output = self.denormalize_output(output)
        return 0.5 * ((denormalized_y - denormalized_output) ** 2)

    def train(self, training_set, expected_outputs, epochs):

        super().train(training_set, self.normalize_outputs(expected_outputs), epochs)

    def test(self, x):
        return self.denormalize_output(super().test(x))