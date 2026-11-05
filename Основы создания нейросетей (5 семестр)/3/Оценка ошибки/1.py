import numpy as np

def neuralNetworks(inp, weights):
    return inp @ weights

prediction = neuralNetworks(np.array([150, 40]), [0.2, 0.3])
print("prediction =", prediction)

true_prediction = 50

print("true - prediction =", true_prediction - prediction)

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

error = get_error(true_prediction, prediction)
print("error =", error)
