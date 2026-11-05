import numpy as np

def neural_networks(inp, weights):
    return inp @ weights

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = np.array([150, 40])
true_prediction = 50

weights = np.array([0.2505, 0.31])

prediction = neural_networks(inp, weights)
error = get_error(true_prediction, prediction)

print("error =", error)
