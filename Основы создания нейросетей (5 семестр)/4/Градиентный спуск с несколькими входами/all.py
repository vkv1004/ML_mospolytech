import numpy as np

def neural_networks(inp, weights):
    return inp @ weights

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


def train(inp, weights, true_prediction, lr, epochs, verbose=True):
    weights = weights.astype(float).copy()
    for i in range(epochs):
        prediction = neural_networks(inp, weights)
        error = get_error(true_prediction, prediction)

        if verbose and (i < 3 or (i + 1) % 100 == 0 or i == epochs - 1):
            print(f"  Эпоха {i+1} | Prediction: {prediction} | Weights: [{weights[0]}, {weights[1]}] | Error: {error}")

        delta = (prediction - true_prediction) * inp * lr
        weights = weights - delta
    return weights


inp = np.array([150, 40])
weights0 = np.array([0.2, 0.3])
true_prediction = 1

#1
train(inp, weights0, true_prediction, 1e-5, 30)


#2
for lr in [1e-5, 1e-4, 1e-3]:
    print(f"\n--- learning_rate = {lr} ---")
    train(inp, weights0, true_prediction, lr, 30)


#3
train(inp, weights0, true_prediction, 1e-5, 1000)
