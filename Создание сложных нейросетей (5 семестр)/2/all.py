import numpy as np

#вспомогательные функции
def relu(x):
    return (x > 0) * x

def reluderiv(x):
    return (x > 0).astype(float)

def train_and_evaluate(layer_hid_1_size=4,
                       learning_rate=0.00001,
                       num_epochs=100,
                       seed=100,
                       verbose=False):
    np.random.seed(seed)
    weights_hid = 2 * np.random.random((layer_in_size, layer_hid_1_size)) - 1
    weights_out = 2 * np.random.random((layer_hid_1_size, layer_out_size)) - 1

    history = []

    for epoch in range(num_epochs):
        layer_out_error = 0.0
        for i in range(len(inp)):
            layer_in  = inp[i:i+1]
            layer_hid = relu(np.dot(layer_in, weights_hid))
            layer_out = layer_hid.dot(weights_out)

            layer_out_error += np.sum((layer_out - true_prediction[i:i+1]) ** 2)

            layer_out_delta = true_prediction[i:i+1] - layer_out
            layer_hid_delta = layer_out_delta.dot(weights_out.T) * reluderiv(layer_hid)

            weights_out += learning_rate * layer_hid.T.dot(layer_out_delta)
            weights_hid += learning_rate * layer_in.T.dot(layer_hid_delta)

        history.append(layer_out_error)
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print("Epoch %3d, Errors: %.4f" % (epoch, layer_out_error))

    preds = np.array([
        (relu(inp[i:i+1] @ weights_hid) @ weights_out)[0, 0]
        for i in range(len(inp))
    ])
    mae = np.mean(np.abs(preds - true_prediction.ravel()))

    return history[-1], mae, history


# данные из предыдущей работы
inp = np.array([
    [15, 10],
    [15, 15],
    [15, 20],
    [25, 10]
], dtype=float)

true_prediction = np.array([[10, 20, 15, 20]], dtype=float).T

layer_in_size  = inp.shape[1]
layer_out_size = true_prediction.shape[1]


# 1. Код из урока
print("1. Код из урока")
np.random.seed(100)

layer_hid_1_size = 4
learning_rate    = 0.00001
num_epoch        = 100

weights_hid = 2 * np.random.random((layer_in_size, layer_hid_1_size)) - 1
weights_out = 2 * np.random.random((layer_hid_1_size, layer_out_size)) - 1

for epoch in range(num_epoch):
    layer_out_error = 0.0
    for i in range(len(inp)):
        layer_in  = inp[i:i+1]
        layer_hid = relu(np.dot(layer_in, weights_hid))
        layer_out = layer_hid.dot(weights_out)
        layer_out_error += np.sum((layer_out - true_prediction[i:i+1]) ** 2)
        layer_out_delta = true_prediction[i:i+1] - layer_out
        layer_hid_delta = layer_out_delta.dot(weights_out.T) * reluderiv(layer_hid)
        weights_out += learning_rate * layer_hid.T.dot(layer_out_delta)
        weights_hid += learning_rate * layer_in.T.dot(layer_hid_delta)

    if epoch % 10 == 0 or epoch == num_epoch - 1:
        print("Epoch %3d, Errors: %.4f" % (epoch, layer_out_error))

# итоговые предсказания
print()
for i in range(len(inp)):
    h = relu(inp[i:i+1] @ weights_hid)
    p = (h @ weights_out)[0, 0]
    print("  inp=%s, pred=%10.4f, true=%5.1f"
          % (inp[i], p, true_prediction[i, 0]))


# 2. Разное количество скрытых нейронов
print("\n 2. Влияние layer_hid_1_size")

print("%12s | %12s | %8s" % ("layer_hid_1", "final error", "MAE"))
for size in [1, 2, 4, 8, 16, 32, 64]:
    err, mae, _ = train_and_evaluate(
        layer_hid_1_size=size,
        learning_rate=0.0001,
        num_epochs=500
    )
    print("%12d | %12.4f | %8.4f" % (size, err, mae))


# 3. Разная скорость обучения
print("\n 3. Влияние learning_rate")

print("%12s | %12s | %8s" % ("learning_rate", "final error", "MAE"))
for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    err, mae, _ = train_and_evaluate(
        layer_hid_1_size=8,
        learning_rate=lr,
        num_epochs=500
    )
    print("%12g | %12.4f | %8.4f" % (lr, err, mae))


#4. Разное количество эпох
print("\n 4. Влияние num_epochs")

print("%12s | %12s | %8s" % ("num_epochs", "final error", "MAE"))
for epochs in [10, 50, 100, 500, 1000, 5000]:
    err, mae, _ = train_and_evaluate(
        layer_hid_1_size=8,
        learning_rate=0.001,
        num_epochs=epochs
    )
    print("%12d | %12.4f | %8.4f" % (epochs, err, mae))
