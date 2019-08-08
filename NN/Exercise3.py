import math
import numpy as np
import random

training_set = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [1, 1, 1],
                         [0, 0, 0]])

def run():
    f = open("weights.txt", "w")

    w = np.array([random.random(), random.random(), random.random()])

    learning_rate = 0.2


    def calc_weights(weights, input, output, correct_output):
        #return weights + learning_rate*output*(1- output)*(correct_output-output)*input  # sigmoid
        return weights + (0.5*learning_rate*(correct_output-output)*input)

    epochs = 1000
    error = math.inf
    while error > 0.0001:
        for k in range(epochs):
            np.random.shuffle(training_set)
            for i in range(4):
                u = np.append(training_set[i][:2], -1)
                z = (w[0]*u[0]) + (w[1]*u[1]) + (w[2]*u[2])
                #v = 1 / (1+(math.e**(-z)))
                if z <= 0:
                    v = 0
                else:
                    v = 1
                t = training_set[i][2]
                #error = 0.5*((t-v)**2)  # squared error
                error = abs(t-v)
                w = calc_weights(w, u, v, t)


    f.write(np.array_str(w))
    f.close()

def testweights(linenum):
    f = open("weights.txt", "r")
    lines = f.readlines()
    line = lines[linenum]
    line = line.replace("[", "")
    line = line.replace("]", "")
    w = np.fromstring(line, dtype=float, sep=" ")
    np.random.shuffle(training_set)
    for i in range(4):
        u = np.append(training_set[i][:2], -1)
        z = (w[0] * u[0]) + (w[1] * u[1]) + (w[2] * u[2])
        #v = 1 / (1 + (math.e ** (-z)))
        if z <= 0:
            v = 0
        else:
            v = 1
        print("Input:", u, " Output:", v)


testweights(0)
#run()