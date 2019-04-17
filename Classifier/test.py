import random

import numpy as np

a = np.array([1, 2, 3])
b = np.array([2, 4, 5])
c = 0.1
# print(np.concatenate([a, b, [c]]))


a = np.array([i for i in range(1, 101)], np.float32)
a = np.reshape(a, [5, 20])
print(a)
# print(a)
# print(np.argmax(a[:,-1]))
# print(a[np.argmax(a[:,-1])])
# print(np.mean(a, axis=0))
a = (a-np.mean(a, axis=0))/(np.max(a, axis=0)-np.min(a, axis=0))
print(a)