import numpy as np

def neural_networks(inp, weights):
    return inp @ weights

def get_error_mse(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

def get_error_rmse(true_prediction, prediction):
    return np.sqrt(np.mean((true_prediction - prediction) ** 2))

def train_sgd(inp, true_predictions, weights, lr, epochs, error_fn, verbose=False):
    weights = weights.astype(float).copy()
    for epoch in range(epochs):
        total_err = 0.0
        for j in range(len(inp)):
            current_inp = inp[j]
            true_prediction = true_predictions[j]
            prediction = neural_networks(current_inp, weights)
            total_err += error_fn(true_prediction, prediction)

            if verbose and epoch < 3:
                print(f"  Эпоха {epoch+1}, набор {j + 1} | Prediction: {prediction} | True: {true_prediction}] | Weights: {np.round(weights, 6)} | Error: {error_fn(true_prediction, prediction)}")
        

            delta = (prediction - true_prediction) * current_inp * lr
            weights = weights - delta

        if verbose and epoch < 3:
            print(f"    Суммарная ошибка эпохи: {total_err}")
    return weights

def eval_all(inp, true_predictions, weights, error_fn):
    preds, total = [], 0.0
    for j in range(len(inp)):
        p = neural_networks(inp[j], weights)
        preds.append(p)
        total += error_fn(true_predictions[j], p)
    return np.array(preds), total

inp = np.array([[150, 40], [170, 80], [160, 90]])
true_predictions = np.array([50, 120, 140])
weights0 = np.array([0.2, 0.3])
lr = 1e-5

#1
w50 = train_sgd(inp, true_predictions, weights0, lr, 50, get_error_mse, verbose=True)
preds50, mse50 = eval_all(inp, true_predictions, w50, get_error_mse)
print("\nИтоговые веса (50 эпох):", np.round(w50, 6))
print("Итоговые прогнозы:      ", np.round(preds50, 4))
print(f"Суммарная MSE:   {mse50}")

#2
w100 = train_sgd(inp, true_predictions, weights0, lr, 100, get_error_mse)
preds100, mse100 = eval_all(inp, true_predictions, w100, get_error_mse)
print("50 эпох : weights =", np.round(w50, 6), f" MSE = {mse50}")
print("100 эпох: weights =", np.round(w100, 6), f" MSE = {mse100}")
print("Прогнозы (50): ", np.round(preds50, 4))
print("Прогнозы (100):", np.round(preds100, 4))
if mse50 > 0:
    print("Улучшение MSE: %.4f%%" % (100 * (mse50 - mse100) / mse50))

#3
w_rmse = train_sgd(inp, true_predictions, weights0, lr, 50, get_error_rmse)
preds_rmse, rmse_total = eval_all(inp, true_predictions, w_rmse, get_error_rmse)
print("Веса при обучении с RMSE-метрикой:", np.round(w_rmse, 6))
print("Веса при обучении с MSE-метрикой: ", np.round(w50, 6))
print("Разница весов (должна быть 0):    ", np.round(w_rmse - w50, 12))
print("Итоговые прогнозы:", np.round(preds_rmse, 4))
print("Сумма RMSE по наборам: {rmse_total}")
