import numpy as np

def neural_networks(inp, weights):
    return inp * weights

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

def train(inp, weights, true_predictions, lr, epochs, verbose=True):
    weights = weights.astype(float).copy()
    for i in range(epochs):
        prediction = neural_networks(inp, weights)
        error = get_error(true_predictions, prediction)
        if verbose and (i < 3 or i == epochs - 1):
            print(f"  Эпоха {i+1} | Prediction: [{prediction[0]}, {prediction[1]}] | Weights: [{weights[0]}, {weights[1]}] | Error: [{error[0]}, {error[1]}]")
        delta = (prediction - true_predictions) * inp * lr
        weights = weights - delta
    return weights


weights0 = np.array([0.2, 0.3])
true_predictions = np.array([50, 120])
lr = 1e-5

#1
train(150, weights0, true_predictions, lr, 30)


#2
for inp_val in [50, 150, 200, 447, 500]:
    print(f"\n--- inp = {inp_val} ---")
    w = train(inp_val, weights0, true_predictions, lr, 30, verbose=False)
    pred = neural_networks(inp_val, w)
    err = get_error(true_predictions, pred)
    print(f"  Итоговые веса:    [{w[0]}, {w[1]}]")
    print(f"  Итоговый прогноз: [{pred[0]}, {pred[1]}]")
    print(f"  Итоговая ошибка:  [{err[0]}, {err[1]}]")
    if inp_val < 447:
        print("  сходится")
    elif inp_val == 447:
        print("  на границе устойчивости")
    else:
        print("  расходится")
