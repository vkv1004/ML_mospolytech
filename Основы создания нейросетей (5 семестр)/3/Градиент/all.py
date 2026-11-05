import numpy as np

def neural_networks(inp, weight):
    return inp * weight

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = 0.9
weight = 0.2
true_prediction = 0.2

print("initial error =", get_error(true_prediction, neural_networks(inp, weight)))

for i in range(10):
    prediction = neural_networks(inp, weight)
    error = get_error(true_prediction, prediction)

    print(
        "Prediction: %.10f, Weight: %.10f, Error: %.20f"
        % (prediction, weight, error)
    )

    delta = (prediction - true_prediction) * inp
    weight = weight - delta

prediction = neural_networks(inp, weight)
error = get_error(true_prediction, prediction)

print("final weight =", weight)
print("final prediction =", prediction)
print("final error =", error)
print()
print('-------------------------------')
print()


#2
def train(inp, weight, true_prediction, epochs=10):
    for _ in range(epochs):
        prediction = neural_networks(inp, weight)
        error = get_error(true_prediction, prediction)

        delta = (prediction - true_prediction) * inp
        weight = weight - delta

    prediction = neural_networks(inp, weight)
    error = get_error(true_prediction, prediction)

    return weight, prediction, error

exper = [
    (0.9, 0.0, 0.2),
    (0.9, 0.2, 0.2),
    (0.9, 0.5, 0.2),
    (0.5, 0.2, 0.2),
    (1.5, 0.2, 0.2),
]

for inp, weight, true_prediction in exper:
    w, p, e = train(inp, weight, true_prediction, epochs=10)
    print(
        f"inp={inp}, start_weight={weight}, "
        f"final_weight={w:.10f}, prediction={p:.10f}, error={e:.20f}"
    )

print()
print('-------------------------------')
print()

#3
for true_prediction in [0.2, 0.5, 0.8]:
    w, p, e = train(
        inp=0.9,
        weight=0.2,
        true_prediction=true_prediction,
        epochs=10
    )
    print(
        f"true_prediction={true_prediction}, "
        f"final_weight={w:.10f}, prediction={p:.10f}, error={e:.20f}"
    )

print()
print('-------------------------------')
print()

#4
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

