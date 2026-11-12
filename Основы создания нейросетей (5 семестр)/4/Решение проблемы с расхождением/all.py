import numpy as np


def train_single(lr, epochs, inp=30, w0=0.2, target=70, verbose=True):
    weight = float(w0)
    history = []
    for i in range(epochs):
        prediction = inp * weight
        error = (target - prediction) ** 2
        history.append((i + 1, prediction, weight, error))

        if verbose and (i < 10 or i == epochs - 1):
            print(f"  Эпоха {i+1} | Prediction: {prediction} | Weight: {weight} | Error: {error})")

        delta = (prediction - target) * inp * lr
        weight -= delta
    return history
#1
train_single(0.001, 10)


#2
for lr in [0.0001, 0.001, 0.01, 0.1]:
    print(f"\n--- learning_rate = {lr} ---")
    train_single(lr, 10)


#3
hist = train_single(0.001, 100, verbose=False)
for ep, pred, w, err in hist:
    if ep in (1, 5, 10, 15, 20, 50, 100):
        print(f"  Эпоха {ep} | Prediction: {pred} | Weight: {w} | Error: {err}")

threshold = 1e-10
hit = next((ep for ep, _, _, err in hist if err < threshold), None)
if hit is not None:
    print(f"\n  Ошибка < {threshold} достигнута на эпохе {hit}")
else:
    print("\n  Порог {threshold} за 100 эпох не достигнут")
