import matplotlib.pyplot as plt
import numpy as np
import math
import random
from datetime import datetime

bias = 0
u = np.array([[bias, 0, 0], [bias, 0, 1], [bias, 1, 0], [bias, 1, 1]])
v = 0
t = [0, 0, 0, 1]
random.seed(datetime.now())
w = np.array([-1, random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])
learning_rate = 0.2

error = math.inf
indexes = []
number_of_epoch = 1000
while error > 0.0001:
    for epoch in range(number_of_epoch):
        for i in range(4):
            random.seed(datetime.now())
            indexes.append(random.randint(0, 3))
        for j in range(4):
            random_input = indexes[j]
            random_output = random_input

            input_vector = np.ndarray(u[random_input])
            correct_output = t[random_input]
            z = np.matmul(input_vector, w.T)
            if z <= 0:
                v = 0
            else:
                v = 1
            error = t[random_output] - v
            w = w + (learning_rate / 2) * np.dot(error, input_vector.T)

