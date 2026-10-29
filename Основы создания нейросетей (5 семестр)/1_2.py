import numpy as np

def neuralNetwork(inp, weights):
    return inp @ weights[0] @ weights[1]

inp = np.array([23, 45])
weight_h_1 = [0.4, 0.1]
weight_h_2 = [0.3, 0.2]
weight_h_3 = [0.6, 0.2]

weight_out_1 = [0.4, 0.1]
weight_out_2 = [0.3, 0.1]
weight_out_3 = [0.7, 0.4]

#weights_h = np.array([weight_h_1, weight_h_2, weight_h_3]).T 
#weights_out = np.array([weight_out_1, weight_out_2, weight_out_3])
weights_h = np.random.rand(2, 3)
weights_out = np.random.rand(3, 2)

weights = [weights_h, weights_out]
print(neuralNetwork(inp, weights))
