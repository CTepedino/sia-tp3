from ref.perceptrons.SingleLayerPerceptron import SingleLayerPerceptron


class StepPerceptron(SingleLayerPerceptron):
    def activator(self, h):
        return 1 if h >= 0 else -1

    def error(self, y, output):
        return 0 if y == output else 1