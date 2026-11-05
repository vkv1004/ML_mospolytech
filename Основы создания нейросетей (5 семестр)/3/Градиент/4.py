import numpy as np

def neural_networks(inp, weight):
    return inp * weight

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = 0.9
weight = 0.2
true_prediction = 0.2


def train(inp, weight, true_prediction, epochs=10):
    for _ in range(epochs):
        prediction = neural_networks(inp, weight)
        error = get_error(true_prediction, prediction)

        delta = (prediction - true_prediction) * inp
        weight = weight - delta

    prediction = neural_networks(inp, weight)
    error = get_error(true_prediction, prediction)

    return weight, prediction, error


for epochs in [1, 5, 10, 50, 100]:
    w, p, e = train(
        inp=0.9,
        weight=0.2,
        true_prediction=0.2,
        epochs=epochs
    )
    print(
        f"epochs={epochs}, final_weight={w:.10f}, "
        f"prediction={p:.10f}, error={e:.20f}"
    )
