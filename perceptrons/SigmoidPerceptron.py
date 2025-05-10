from perceptrons.NonLinearPerceptron import NonLinearPerceptron
import math

class SigmoidPerceptron(NonLinearPerceptron):

    def activator(self, h):
        return 1 / (1 + math.exp(-h))

    def activator_derivative(self, h):
        s = self.activator(h)
        return s * (1 - s)

    def normalize_output(self, output):
        return (output - self.min_output) / (self.max_output - self.min_output)

    def denormalize_output(self, normalized_output):
        return normalized_output * (self.max_output - self.min_output) + self.min_output

