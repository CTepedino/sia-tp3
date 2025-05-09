from ref.perceptrons.NonLinearPerceptron import NonLinearPerceptron
import math

class TanhPerceptron(NonLinearPerceptron):
    def __init__(self, input_size, learning_rate, seed = None):
        super().__init__(input_size, learning_rate, seed = seed)
        self.min_output = None
        self.max_output = None

    def activator(self, h):
        return math.tanh(h)

    def activator_derivative(self, h):
        return 1 - math.tanh(h)**2

    def normalize_output(self, output):
        return (output - self.min_output) / (self.max_output - self.min_output) - 1

    def denormalize_output(self, normalized_output):
        return ((normalized_output + 1)/2) * (self.max_output - self.min_output) + self.min_output

