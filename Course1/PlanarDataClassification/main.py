# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# matplotlib inline
np.random.seed(1)

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples.' % m)

## Simple Logistic Regression
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' %
      float((np.dot(Y, LR_predictions) +
             np.dot(1 - Y, 1 - LR_predictions)) / float(
          Y.size) * 100) + '% ' + "(percentage of correctly labelled data points)")

plt.show()


def layer_sizes(X, Y):
    n_x = X.shape[0]  # size of input layer
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return n_x, n_h, n_y


# n_x, n_h, n_y = layer_sizes(X, Y)


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * .01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros(shape=(n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# parameters = initialize_parameters(n_x, n_h, n_y)


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


# Y_hat, cache = forward_propagation(X, parameters)


def compute_cost(Y, Y_hat, m):
    first_term = np.multiply(Y, np.log(Y_hat))
    last_term = np.multiply(1 - Y, np.log(1 - Y_hat))
    overall = first_term + last_term
    cost = np.sum(overall) / -m
    # print(cost)
    cost = np.squeeze(cost)  # Don't know what is the point
    # print(cost)

    return cost


# cost = compute_cost(Y, Y_hat, m)


def back_propagation(X, Y, cache, parameters, m):
    A1 = cache['A1']
    A2 = cache['A2']
    W2 = parameters['W2']
    Z1 = cache['Z1']

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    # dZ1 = np.multiply(np.dot(W2.T, dZ2), np.dot(gprime, Z1))
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads


# grads = back_propagation(X, Y, cache, parameters, m)


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def nn_model(X, Y, n_h, num_iterations=40000, print_cost=True):
    np.random.seed(3)
    n_x, _, n_y = layer_sizes(X, Y)
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        Y_hat, cache = forward_propagation(X, parameters)
        cost = compute_cost(Y, Y_hat, m)
        grads = back_propagation(X, Y, cache, parameters, m)
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return Y_hat


Y_hat = nn_model(X, Y, 48)
predictions = np.round(Y_hat)

print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

## I don't understand this piece of code why it is not working?


# def predict(parameters, X):
#     A2, cache = forward_propagation(X, parameters)
#     predictions = np.round(A2)
#     return predictions
#
# parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
# # Plot the decision boundary
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()
