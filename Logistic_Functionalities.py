import numpy as np


def sigmoid_activation_function(z):
    sigmoid_value = (1 / (1 + np.exp(-z)))
    return sigmoid_value

#-----------------------------------------------


def net_function(theta, x):
    net_value = np.dot(x, theta)
    return net_value

#-------------------------------------------------


def train_function(x, y_actual):
    l = 0.001
    epochs = 200000
    theta = np.zeros((x.shape[1], 1))
    length = x.shape[0]

    for i in range(epochs):
        y_predicted = predict_probability(x, theta)
        prediction = (np.dot((y_predicted - y_actual).T, x) / length)
        update = (l * prediction.T)
        theta = theta - update

    return theta

#--------------------------------------------------


def predict_probability(theta, x):
    net_value = net_function(x, theta)
    sig_value = sigmoid_activation_function(net_value)
    predicted_probability = sig_value
    return predicted_probability

#-----------------------------------------------------


def predict_classes(theta, x):
    predicted_value = predict_probability(x, theta)
    predicted_classes = np.where(predicted_value >= 0.5, 1, 0)
    return predicted_classes
