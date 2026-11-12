import numpy as np


def neural_networks(inp, weights):
    return inp @ weights

def get_error_mse(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

def get_error_rmse(true_prediction, prediction):
    return np.sqrt(np.mean((true_prediction - prediction) ** 2))

def train_sgd(inp, true_predictions, weights, lr, epochs, error_fn, verbose=False):
    weights = weights.astype(float).copy()
    with np.errstate(over='ignore', invalid='ignore'):
        for epoch in range(epochs):
            total_err = 0.0
            for j in range(len(inp)):
                current_inp = inp[j]
                true_prediction = true_predictions[j]
                prediction = neural_networks(current_inp, weights)
                total_err += error_fn(true_prediction, prediction)

                if verbose and epoch < 3:
                    print(f"  Эпоха {epoch+1}, набор {j+1} | Prediction: {prediction:.10f} | True: {true_prediction:.10f} | Weights: {np.round(weights, 6)} | Error: {error_fn(true_prediction, prediction):.10f}")

                delta = (prediction - true_prediction) * current_inp * lr
                weights = weights - delta

            # защита от расходимости (NaN/inf)
            if not (np.isfinite(total_err) and np.all(np.isfinite(weights))):
                print(f"  [!] Дивергенция (SGD) на эпохе {epoch+1}")
                return weights

            if verbose and epoch < 3:
                print(f"    Суммарная ошибка эпохи: {total_err:.10f}")
    return weights

def train_full_gd(inp, true_predictions, weights, lr, epochs, error_fn, verbose=False):
    weights = weights.astype(float).copy()
    with np.errstate(over='ignore', invalid='ignore'):
        for epoch in range(epochs):
            total_err = 0.0
            delta = np.zeros_like(weights, dtype=float)
            for j in range(len(inp)):
                current_inp = inp[j]
                true_prediction = true_predictions[j]
                prediction = neural_networks(current_inp, weights)
                total_err += error_fn(true_prediction, prediction)

                if verbose and epoch < 3:
                    print(f"  Эпоха {epoch+1}, набор {j+1} | Prediction: {prediction:.10f} | True: {true_prediction:.10f} | Weights: {np.round(weights, 6)}")

                delta += (prediction - true_prediction) * current_inp * lr

            weights = weights - delta / len(inp)

            # защита от расходимости (NaN/inf)
            if not (np.isfinite(total_err) and np.all(np.isfinite(weights))):
                print(f"  [!] Дивергенция (Full GD) на эпохе {epoch+1}")
                return weights

            if verbose and epoch < 3:
                print(f"    Errors: {total_err:.10f}")
    return weights

def eval_all(inp, true_predictions, weights, error_fn):
    preds, total = [], 0.0
    for j in range(len(inp)):
        p = neural_networks(inp[j], weights)
        preds.append(p)
        total += error_fn(true_predictions[j], p)
    return np.array(preds), total

def rel_error_percent(pred, true):
    return abs(pred - true) / abs(true) * 100


#1
inp = np.array([
    [150, 40],
    [140, 35],
    [155, 45],
    [185, 95],
    [145, 40],
    [195, 100],
    [180, 95],
    [170, 80],
    [160, 90],
])
true_predictions = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100])
weights0 = np.array([0.2, 0.3])
learning_rate = 0.00001

w_sgd = train_sgd(inp, true_predictions, weights0, learning_rate, 500, get_error_mse)
_, err_sgd = eval_all(inp, true_predictions, w_sgd, get_error_mse)
print(f"SGD 500 эпох    -> Errors: {err_sgd:.10f}")

w_gd500 = train_full_gd(inp, true_predictions, weights0, learning_rate, 500, get_error_mse)
_, err_gd500 = eval_all(inp, true_predictions, w_gd500, get_error_mse)
print(f"Full GD 500 эпох    -> Errors: {err_gd500:.10f}")

w_gd5000 = train_full_gd(inp, true_predictions, weights0, learning_rate, 5000, get_error_mse)
_, err_gd5000 = eval_all(inp, true_predictions, w_gd5000, get_error_mse)
print(f"Full GD 5000 эпох   -> Errors: {err_gd5000:.10f}")

print("\nИтоговые веса:", np.round(w_gd5000, 6))
print("Prediction [150, 45] =", neural_networks(np.array([150, 45]), w_gd5000),
      "-> вероятность, что это мужчина")
print("Prediction [170, 85] =", neural_networks(np.array([170, 85]), w_gd5000),
      "-> вероятность, что это мужчина")


#2
for lr in [0.000001, 0.00001, 0.0001]:
    w = train_full_gd(inp, true_predictions, weights0, lr, 5000, get_error_mse)
    if not np.all(np.isfinite(w)):
        print(f"lr = {lr:<10} | РАСХОДИТСЯ (веса = NaN/inf) — обучение остановлено")
        continue
    _, err = eval_all(inp, true_predictions, w, get_error_mse)
    p1 = neural_networks(np.array([150, 45]), w)
    p2 = neural_networks(np.array([170, 85]), w)
    print(f"lr = {lr:<10} | weights = {np.round(w, 6)} | Errors = {err:.6f} | "
          f"[150,45] -> {p1:.4f} | [170,85] -> {p2:.4f}")


#3
for epochs in [100, 5000, 10000]:
    w = train_full_gd(inp, true_predictions, weights0, learning_rate, epochs, get_error_mse)
    _, err = eval_all(inp, true_predictions, w, get_error_mse)
    print(f"epochs = {epochs:<6} | weights = {np.round(w, 6)} | Errors = {err:.6f}")


#4
w_rmse = train_full_gd(inp, true_predictions, weights0, learning_rate, 5000, get_error_rmse)
w_mse  = train_full_gd(inp, true_predictions, weights0, learning_rate, 5000, get_error_mse)

_, rmse_total = eval_all(inp, true_predictions, w_rmse, get_error_rmse)
_, mse_total  = eval_all(inp, true_predictions, w_mse,  get_error_mse)

print("Веса при обучении с RMSE-метрикой:", np.round(w_rmse, 6))
print("Веса при обучении с MSE-метрикой: ", np.round(w_mse, 6))
print("Разница весов (должна быть 0):    ", np.round(w_rmse - w_mse, 12))
print(f"Сумма RMSE по наборам: {rmse_total:.10f}")
print(f"Сумма MSE  по наборам: {mse_total:.10f}")


#5
inp_sum = np.array([
    [10, 5],
    [0, -5],
    [2, 6],
])
true_sum = np.array([15, -5, 8])
weights_sum0 = np.array([0.2, 0.3])

w_sum = train_full_gd(inp_sum, true_sum, weights_sum0, 0.01, 10000, get_error_mse)
_, err_sum = eval_all(inp_sum, true_sum, w_sum, get_error_mse)
print("Обученные веса сумматора:", np.round(w_sum, 6))
print(f"Суммарная MSE на обучающей выборке: {err_sum:.10f}")

for test in [np.array([12, 4]), np.array([3, -8])]:
    p = neural_networks(test, w_sum)
    true = test.sum()
    rel = rel_error_percent(p, true)
    print(f"Вход: {test} | Prediction: {p:.6f} | True: {true} | "
          f"Отн. ошибка: {rel:.4f}%  {'OK' if rel < 1 else 'ПЛОХО'}")
