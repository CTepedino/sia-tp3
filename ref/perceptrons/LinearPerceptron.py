from ref.perceptrons.SingleLayerPerceptron import SingleLayerPerceptron


class LinearPerceptron(SingleLayerPerceptron):
    def activator(self, h):
        return h

    def error(self, y, output):
        return 0.5 * ((y - output) ** 2)

