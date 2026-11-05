import numpy as np

# 1
arr_1 = np.array([1, 5, 7, 8, 9])

# 2
arr_2 = arr_1 * 2
print(arr_2, '\n')

# 3
arr_1 = np.random.rand(3, 3)
arr_2 = np.random.rand(3, 3)
print("1:\n", arr_1, '\n')
print("2:\n", arr_2, '\n')

# 4
arr_mul = arr_1 * arr_2
print(arr_mul, '\n')

# 5
arr_rand = np.random.randint(0, 100, size=10)
print(arr_rand, '\n')

# 6
even_arr = arr_rand[::2]
print(even_arr, '\n')

# 7
print("Ср знач: ", np.mean(arr_rand))
print("Стандартное отклонение:  ", np.std(arr_rand))
print("max: ", np.max(arr_rand))
print("min: ", np.min(arr_rand))
