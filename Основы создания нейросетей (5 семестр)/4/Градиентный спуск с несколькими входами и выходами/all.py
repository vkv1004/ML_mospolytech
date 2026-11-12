import numpy as np

def neural_networks(inp, weights):
    return inp @ weights

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

def train(inp, weights, true_predictions, lr, epochs, verbose=True):
    weights = weights.astype(float).copy()
    for i in range(epochs):
        prediction = neural_networks(inp, weights)
        error = get_error(true_predictions, prediction)
        if verbose and (i < 3 or i == epochs - 1):
            print(f"  Эпоха {i+1} | Prediction: [{prediction[0]}, {prediction[1]}] | Error: [{error[0]}, {error[1]}]")
        
            print(f" Weights:\n{np.round(weights, 8)}")
        delta = np.outer(inp, prediction - true_predictions) * lr
        weights = weights - delta
    return weights


inp = np.array([150, 40])
weights0 = np.array([[0.2, 0.3], [0.5, 0.7]]).T
lr = 1e-5

#1
w = train(inp, weights0, np.array([50, 120]), lr, 50)
print("\nИтог: Prediction =", neural_networks(inp, w))


#2
for true_p in [np.array([50, 120]),
               np.array([70, 90]),
               np.array([30, 110])]:
    print(f"\n--- true_predictions = {true_p} ---")
    w = train(inp, weights0, true_p, lr, 50, verbose=False)
    pred = neural_networks(inp, w)
    err = get_error(true_p, pred)
    print(f"  Prediction = [{pred[0]}, {pred[1]}]")
    print(f"  Error      = [{err[0]}, {err[1]}]")
