# Written by Marcel Meimbresse (2019)

import math
import numpy as np
import random

start_data = np.array([[1, 0, 1],
                       [0, 1, 1],
                       [1, 1, 0],
                       [0, 0, 0]])
training_set = []

for i in range(1000):
    training_set.append(start_data[np.random.randint(0, len(start_data))])

print(training_set)

def run_single_perceptron():
    w = np.array([random.random(), random.random(), random.random()])

    learning_rate = 0.2

    def calc_weights(weights, input, output, correct_output):
        return weights + (0.5*learning_rate*(correct_output-output)*input)

    epochs = 400
    success = False
    avg_error = math.inf
    for k in range(epochs):
        if avg_error > 0.0001:
            np.random.shuffle(training_set)
            avg_error = 0
            for i in range(len(training_set)):
                u = np.append(training_set[i][:2], -1)
                z = np.matmul(w, u)
                if z <= 0:
                    v = 0
                else:
                    v = 1
                t = training_set[i][2]
                #error = 0.5*((t-v)**2)  # squared error
                error = abs(t-v)
                avg_error += error
                w = calc_weights(w, u, v, t)
            avg_error = avg_error/len(training_set)
            print(avg_error)
        else:
            success = True
            print("Error threshold reached")
            break
    if success:
        f = open("single_weights.txt", "w")
        f.write(np.array_str(w))
        f.close()

# Ran for a long time (and mostly did not finish) with higher learning rate since it got stuck in a local minimum
# Setting learning rate very low solved it. In general, learning rate should not be set to higher value like 0.2-0.9
def run_mlp():
    w_hidden = np.array([np.array([random.random(), random.random(), random.random()]),
                         np.array([random.random(), random.random(), random.random()])])

    w_output = np.array([random.random(), random.random(), random.random()])

    learning_rate = 0.01

    def calc_weights_output(weights, input, output, correct_output):
        return weights + (learning_rate * (correct_output - output) * input)

    def calc_weights(weights, error, input):
        #return weights + learning_rate*output*(1- output)*(correct_output-output)*input  # sigmoid
        #return weights + (0.5 * learning_rate * (correct_output - output) * input)       # TLU
        return weights + (learning_rate * error * input)

    epochs = 7000
    avg_error = math.inf
    success = False
    for k in range(epochs):
        if avg_error > 0.0001:
            np.random.shuffle(training_set)
            avg_error = 0
            for i in range(len(training_set)):
                u = np.append(training_set[i][:2], -1)
                x_1 = w_hidden[0]
                x_2 = w_hidden[1]
                z_1 = np.matmul(x_1, u)
                z_2 = np.matmul(x_2, u)
                x_1_output = (1 / (1+(math.e**(-z_1))))
                x_2_output = (1 / (1+(math.e**(-z_2))))
                z = (w_output[0] * x_1_output) + (w_output[1] * x_2_output) + (w_output[2] * -1)
                if z <= 0:
                    v = 0
                else:
                    v = 1
                t = training_set[i][2]
                # error = 0.5*((t-v)**2)  # squared error
                error = abs(t-v)
                x_1_error = abs(x_1_output * (1 - x_1_output) * (error * w_output[0]))
                x_2_error = abs(z_2 * (1 - x_2_output) * (error * w_output[1]))
                w_output = calc_weights_output(w_output, np.array([x_1_output, x_2_output, -1]), v, t)
                w_hidden[0] = calc_weights(w_hidden[0], x_1_error, u)
                w_hidden[1] = calc_weights(w_hidden[1], x_2_error, u)
                avg_error += error
            avg_error = avg_error/len(training_set)
            print(avg_error)
        else:
            success = True
            print("Reached error threshold")
            break
    if success:
        f = open("mlp_weights.txt", "w")
        f.write(np.array_str(w_hidden[0]))
        f.write("\n")
        f.write(np.array_str(w_hidden[1]))
        f.write("\n")
        f.write(np.array_str(w_output))
        f.close()


def testweights_mlp():
    f = open("mlp_weights.txt", "r")
    lines = f.readlines()
    f.close()
    w_hidden = []
    w_output = []
    for index, line in enumerate(lines):
        line = line.replace("[", "")
        line = line.replace("]", "")
        if index == 0 or index == 1:
            w_hidden.append(np.fromstring(line, dtype=float, sep=" "))
        else:
            w_output = np.fromstring(line, dtype=float, sep=" ")

    np.random.shuffle(training_set)
    for i in range(10):
        u = np.append(training_set[i][:2], -1)
        x_1 = w_hidden[0]
        x_2 = w_hidden[1]
        z_1 = np.matmul(x_1, u)
        z_2 = np.matmul(x_2, u)
        x_1_output = 1 / (1 + (math.e ** (-z_1)))
        x_2_output = 1 / (1 + (math.e ** (-z_2)))
        z = (w_output[0] * x_1_output) + (w_output[1] * x_2_output) + (w_output[2] * -1)
        if z <= 0:
            v = 0
        else:
            v = 1
        print("Input:", u, " Output:", v)


def testweights_single_perceptron(linenum):
    f = open("single_weights.txt", "r")
    lines = f.readlines()
    f.close()
    line = lines[linenum]
    line = line.replace("[", "")
    line = line.replace("]", "")
    w = np.fromstring(line, dtype=float, sep=" ")
    np.random.shuffle(training_set)
    for i in range(10):
        u = np.append(training_set[i][:2], -1)
        z = np.matmul(w, u)
        if z <= 0:
            v = 0
        else:
            v = 1
        print("Input:", u, " Output:", v)


run_mlp()
#testweights_mlp()
#testweights_single_perceptron(0)
#run_single_perceptron()