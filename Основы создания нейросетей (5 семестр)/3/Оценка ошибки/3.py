import numpy as np

def neural_networks(inp, weights):
    return inp @ weights

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = np.array([150, 40])
true_prediction = 50

weights = np.array([0.2, 0.3])
step = 0.0001
it = 0

while True:
    prediction = neural_networks(inp, weights)
    error = get_error(true_prediction, prediction)

    if error < 0.001:
        break

    weights[0] += step
    it += 1

print("iterations ", it)
print("weights ", weights)
print("prediction ", prediction)
print("error ", error)
