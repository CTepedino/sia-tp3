from abc import ABC, abstractmethod
import random

class SingleLayerPerceptron(ABC):
    def __init__(self, input_size, learning_rate, seed = None):
        if seed is not None:
            random.seed(seed)

        self.weights = [random.uniform(-1, 1) for _ in range(input_size + 1)]
        self.learning_rate = learning_rate

        self.error_min = None
        self.best_weights = None
        self.error_min_epoch = None

    @abstractmethod
    def activator(self, h):
        pass

    @abstractmethod
    def error(self, y, output):
        pass

    def weight_update(self, w, x, y, output, h):
        return w + self.learning_rate * (y-output) * x

    def train(self, training_set, expected_outputs, epochs):


        #f = open("avg_err_by_epoch", "w")
        for epoch in range(epochs):
            combined = list(zip(training_set, expected_outputs))
            random.shuffle(combined)
            training_set[:], expected_outputs[:] = zip(*combined)

            error = 0
            for x, y in zip(training_set, expected_outputs):
                x_with_bias = x + [1]
                h = sum(w * x_i for w, x_i in zip(self.weights, x_with_bias))

                output = self.activator(h)
                self.weights = [self.weight_update(w, x_i, y, output, h) for w, x_i in zip(self.weights, x_with_bias)]
                error += self.error(y, output)
                #print(f"input: {x}, out: {output}, expected: {y}")

            average_error = error/len(training_set)
            #print(f"epoch {epoch + 1} average error - {average_error}")
            #f.write(f"{epoch+1},{average_error}\n")
            if self.error_min is None or average_error < self.error_min:
                self.error_min = average_error
                self.best_weights = self.weights
                self.error_min_epoch = epoch + 1

        self.weights = self.best_weights

        #f.close()


    def test(self, x):
        x_with_bias = x + [1]
        return self.activator(sum(w * x_i for w, x_i in zip(self.weights, x_with_bias)))
