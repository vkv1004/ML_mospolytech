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


experiments = [
    (0.9, 0.0, 0.2),
    (0.9, 0.2, 0.2),
    (0.9, 0.5, 0.2),
    (0.5, 0.2, 0.2),
    (1.5, 0.2, 0.2),
]

for inp, weight, true_prediction in experiments:
    w, p, e = train(inp, weight, true_prediction, epochs=10)
    print(
        f"inp={inp}, start_weight={weight}, "
        f"final_weight={w:.10f}, prediction={p:.10f}, error={e:.20f}"
    )
