import random
import sys

import matplotlib.pyplot as plt


class LinearModel:
    """ A model (linear equation) for linear regression problems.
    """

    def __init__(self):
        # Initially set the bias and weight to be 0.
        self._bias = 0
        self._weight = 0

        # Initialize the epoch to 0, the previous loss to a large number.
        self._epoch = 0
        self._previous_loss = sys.maxsize

        # For plotting, set the last line to None.
        self._last_line = None

    def __call__(self, feature):
        """ Override the call function to serve as a forward pass through the model.

        :param feature: value to find a prediction for
        :return: inference for the provided feature
        """
        return self._bias + self._weight*feature

    def __str__(self):
        return "{0} + w_1 * {1}".format(self._bias, self._weight)

    def __repr__(self):
        return "LinearModel({0}, {1})".format(self._bias, self._weight)

    @property
    def bias(self):
        return self._bias

    @property
    def weight(self):
        return self._weight

    @staticmethod
    def from_bias_and_weight(bias, weight):
        """ Creates a LinearModel with the provided bias and weight as initial values.

        :param bias: bias of the linear model
        :param weight: weight of the linear model
        :return: new LinearModel with the provided arguments
        """
        new_model = LinearModel()
        new_model._bias = bias
        new_model._weight = weight
        return new_model

    def predict(self, features, batch_size=None):
        """ Creates a list of predictions given a provided list of features.

        :param features: list of numbers to create predictions on
        :param batch_size: size of the batch which is used in predictions
        :return: list of numbers which are the resulting predictions
        """
        return [self(feature) for feature in features[:batch_size]]

    def train(self, training_set, learning_rate=.01, delta=.05, batch_size=None):
        """ Given a set of points, mutate this model using gradient descent at the provided learning rate to find
            an optimal linear equation to fit the points.

        :param training_set: TrainingSet of points to create a model for
        :param learning_rate: rate at which we take steps in gradient descent
        :param delta: threshold which determines when to stop training
        :param batch_size: size of the batch that is used in predictions and gradient descent
        :return: None
        """
        # Increment the number of epochs this model has been through and display only the most recent line.
        self._epoch += 1
        if self._last_line is not None:
            self._last_line.remove()
        plt.title('epoch {0}'.format(self._epoch))
        self._last_line, = plt.plot(range(1, 10), self.predict(range(1, 10)))

        # Pause to show updates.
        plt.pause(.005)

        # Make predictions and find the error rate.
        predictions = self.predict(training_set.features, batch_size)
        loss = mse(training_set.labels, predictions, batch_size)
        print("epoch {0}: {1} ---> loss: {2}".format(self._epoch, self, loss))

        # Determine whether training should continue or not.
        if abs(loss-self._previous_loss) > delta:
            self._previous_loss = loss

            # Calculate the gradient and adjust the model, then train again.
            gradient = l2_gradient(training_set.labels, training_set.features, self, batch_size)
            self._bias -= learning_rate * gradient[0]
            self._weight -= learning_rate * gradient[1]
            self.train(training_set, learning_rate, delta, batch_size)
        else:
            print("\nconverged at: ", self)


class TrainingSet:
    """ A collection of points to be used for training a LinearModel.
    """

    def __init__(self, features, labels):
        self._features = features
        self._labels = labels

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @staticmethod
    def create_training_set(n):
        """ Creates a training set of size n using points of x and y values ranging [1, 9].

        :param n: number of points to generate
        :return: new instance of TrainingSet
        """
        return TrainingSet([random.randint(1, 9) for _ in range(n)], [random.randint(1, 9) for _ in range(n)])


def mse(labels, predictions, batch_size=None):
    """ Calculates the mean squared error (MSE) given lists: one of labels and one of predictions.

    :param labels: list of numbers which are examples in a set
    :param predictions: list of numbers which are values predicted from some model
    :param batch_size: size of the batch to find the MSE of
    :return: number which is mean squared error of the set
    """
    return (1/len(labels[:batch_size]))*sum([(y-y_prime)**2
                                             for y, y_prime in zip(labels[:batch_size], predictions[:batch_size])])


def l2_gradient(labels, features, model, batch_size=None):
    """ Calculates the gradient of an L2 loss function using MSE.

    :param labels: list of provided labels
    :param features: list of provided features
    :param model: LinearModel to use for gradient descent
    :param batch_size: size of the batch to use
    :return: tuple representing gradient of function in the order (y, y')
    """
    del_b = (-2/len(labels[:batch_size]))*sum([y-model(x) for y, x in zip(labels[:batch_size], features[:batch_size])])
    del_w = (-2/len(labels[:batch_size]))*sum([x*(y-model(x))
                                               for y, x in zip(labels[:batch_size], features[:batch_size])])
    return del_b, del_w


if __name__ == "__main__":
    # Generate a random set of points.
    source = TrainingSet.create_training_set(10)

    # Set up the axes, points, and scaling of the graph.
    plt.plot(source.features, source.labels, 'ro')
    plt.title('epoch 0')
    plt.axis([0, 10, 0, 10])
    plt.xlabel('x')
    plt.ylabel('y')

    # Create an initially empty model and train it using gradient descent. Display the line as it learns.
    lin_model = LinearModel()
    lin_model.train(source, delta=.0001)
    plt.show()
