import numpy as np

def relu(x):
    return (x > 0) * x

np.random.seed(42)

inp = np.array([
    [15, 10],
    [15, 15],
    [15, 20],
    [25, 10]
])
true_prediction = np.array([[10, 20, 15, 20]]).T

layer_in_size = len(inp[0])
layer_out_size = len(true_prediction[0])


#1
layer_hid_size = 3
weights_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
weights_out = 2 * np.random.random((layer_hid_size, layer_out_size)) - 1

prediction_hid = relu(np.dot(inp[0], weights_hid))
prediction = prediction_hid.dot(weights_out)

print("inp[0]              =", inp[0])
print("weights_hid         =\n", np.round(weights_hid, 4))
print("weights_out         =", np.round(weights_out.ravel(), 4))
print("hidden после ReLU   =", np.round(prediction_hid, 4))
print("prediction для inp[0] =", np.round(prediction, 4))
print("Параметров в сети   =", weights_hid.size + weights_out.size)


#2
layer_hid_size = 8
weights_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
weights_out = 2 * np.random.random((layer_hid_size, layer_out_size)) - 1

print("weights_hid =")
print(np.round(weights_hid, 4))
print("weights_out =", np.round(weights_out.ravel(), 4))

print(f"\n{'вход':>12} | {'hidden (ReLU)':<50} | {'prediction':>12}")
for row in inp:
    h = relu(row @ weights_hid)
    p = h @ weights_out
    print(f"{str(row):>12} | {np.round(h, 4)} | {p[0]:>12.4f}")

all_hid = np.array([relu(row @ weights_hid) for row in inp])
dead = (all_hid == 0).mean(axis=0)
print("\nДоля нулевых (мёртвых) нейронов по столбцам:", np.round(dead, 2))
print("Параметров в сети =", weights_hid.size + weights_out.size)


#3
hid1_size, hid2_size = 4, 4
weights_h1 = 2 * np.random.random((layer_in_size, hid1_size)) - 1
weights_h2 = 2 * np.random.random((hid1_size, hid2_size)) - 1
weights_out = 2 * np.random.random((hid2_size, layer_out_size)) - 1

print("weights_h1 =\n", np.round(weights_h1, 4))
print("weights_h2 =\n", np.round(weights_h2, 4))
print("weights_out =", np.round(weights_out.ravel(), 4))

print(f"\n{'вход':>12} | {'hid1 (ReLU)':<34} | {'hid2 (ReLU)':<34} | {'prediction':>12}")
for row in inp:
    h1 = relu(row @ weights_h1)
    h2 = relu(h1 @ weights_h2)
    p = h2 @ weights_out
    print(f"{str(row):>12} | {np.round(h1, 3)} | {np.round(h2, 3)} | {p[0]:>12.4f}")

all_h1 = np.array([relu(row @ weights_h1) for row in inp])
all_h2 = np.array([relu(relu(row @ weights_h1) @ weights_h2) for row in inp])
print("\nДоля мёртвых нейронов, слой 1:", np.round((all_h1 == 0).mean(axis=0), 2))
print("Доля мёртвых нейронов, слой 2:", np.round((all_h2 == 0).mean(axis=0), 2))
print("Параметров в сети =", weights_h1.size + weights_h2.size + weights_out.size)

print("\nСравнение числа параметров:")
print("  1 скрытый слой (3 нейрона)  :", 2 * 3 + 3 * 1)
print("  1 скрытый слой (8 нейронов) :", 2 * 8 + 8 * 1)
print("  2 скрытых слоя (4 + 4)      :", 2 * 4 + 4 * 4 + 4 * 1)
