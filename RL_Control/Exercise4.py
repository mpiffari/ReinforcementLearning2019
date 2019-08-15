import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from math import *
import random
import itertools
import time
import sys


# Parameters
g = 9.8  # m/s^2
m = 1    # kg
L = 4  # Max x val (mountain "width") in m
h = 1  # Mountain height in m
T = 0.05
max_a = 4
episodes = 30
v_max = sqrt(2*g*h)
discount_rate = 1
learning_rate = 0.225  #so far 0.08 gave best results
grid_size = 5
time_limit = 8  # in seconds
interval = 20
random.seed(42069000)  # unlucky seed: 879856

time_limit = (time_limit*1000)/interval

sigma_x = 0.25
sigma_v = 0.25
start_epsilon = 0.9


#Data structures
cx = np.linspace(0, L, grid_size)
cv = np.linspace(-v_max, v_max, grid_size)
c_vx = list(itertools.product(cx, cv))
w = np.array([[random.random() for x in range(grid_size**2)] for y in range(2)])
feature_vect = np.zeros((grid_size**2, 1))


# Discretizes an velocity value to fit into one of the 5 velocity grid cells
# def nearest_v(cv, value):
#     array = np.asarray(cv)
#     idx = (np.abs(array - value)).argmin()
#     return idx
#
#
# def nearest_x(cx, value):
#     array = np.asarray(cv)
#     idx = (np.abs(array - value)).argmin()
#     return idx
#
#
# def double_to_single_index(x, y):
#     return x * 5 + y


def phi_func(s):
    global feature_vect
    feature_vect = np.zeros((grid_size**2, 1))
    for i in range(25):
        feature_vect[i] = e**(-( ((s[0]-c_vx[i][0])**2) / (sigma_x**2) + ((s[1]-c_vx[i][1])**2) / (sigma_v**2) ) / 2)
    return feature_vect
    #return e**(-( ((s[0]-cx[s[0]])**2) / (sigma_x**2) + ((s[1]-cv[nearest_v(cv, s[1])])**2) / (sigma_v**2) ) / 2)


# State action value function
def Q(s, a):
    feature_vect = phi_func(s)
    if a == -max_a:
        return np.dot(w[0].T, feature_vect)
    if a == max_a:
        return np.dot(w[1].T, feature_vect)



def mountains(x):
    return 0.5*h*(1+cos(2*pi*(x/L)))


def A(x):
    return m*(1+ ((h**2)/L**2) * (pi**2) * (sin(2*pi*(x/L))**2))


def B(x):
    return m * ((h**2)/(L**2) * (pi**3) * sin(4*pi*(x/L)))


def C(x):
    return -((m*g*h)/L) * pi * sin(2*pi*(x/L))


def D(x):
    return sqrt(1 + ((pi**2) * ((h**2)/(L**2)) * (sin(pi*(x/L))**2)))


def next_state(old_state, a_t):
    new_state = [0, 0]
    new_state[0] = old_state[0] + (T*old_state[1])
    new_state[1] = old_state[1] + ((T*(1/A(old_state[0]))) * ((-B(old_state[0]) * old_state[1]) - C(old_state[0]) + (a_t/D(old_state[0]))))
    return new_state


def next_action(state):
    epsilon = start_epsilon/((episode_count+1)/2)
    Q_minus = Q(state, -max_a)
    Q_plus = Q(state, max_a)
    if Q_minus > Q_plus:
        a = -4
    else:
        a = 4
    if random.random() < epsilon:
        a = random.choice([-max_a, max_a])  # might evt. not be random enough
    return a


episode_count = 0
x_vals = np.arange(0.0, L + 1, T)
mountain_vals = [mountains(x) for x in x_vals]
s = np.array([np.argmin(mountain_vals) * T, 0])  # State as a vector initialized in valley with velocity = 0

# Generator function that keeps returning frame numbers until mountain crest is reached or time_limit is reached
def gen():
    global s
    global frame
    global ax
    while 0 < s[0] <= L+0.004 and frame < time_limit and episode_count < episodes:
        frame += 1
        yield frame

avg_error = 0
total_error = 0
error_list = []
a_t = 4
# Updating function, to be repeatedly called by the animation
def update(framenumber):
    # obtain point coordinates
    global s
    global frame
    global episode_count
    global avg_error
    global total_error
    global a_t
    global w
    #print("x_t:", s[0], "v_t:", s[1])
    next_a = next_action(s)
    next_s = next_state(s, next_a)
    y = mountains(s[0])
    car.set_data(s[0], y)
    if np.around(next_s[0], 4) >= L:
        ax.set_title("Crest reached!")
        next_s = np.array([np.argmin(mountain_vals) * T, 0])  # State as a vector initialized in valley with velocity = 0
        print("Episode", episode_count, "finished")
        print("Total reward:", total_error)
        error_list.append(total_error)
        total_error = 0
        frame = 0
        episode_count += 1
    elif frame == time_limit:
        ax.set_title("Time limit reached!")
        next_s = np.array([np.argmin(mountain_vals) * T, 0])  # State as a vector initialized in valley with velocity = 0
        print("Episode", episode_count, "finished")
        print("Total reward:", total_error)
        error_list.append(total_error)
        total_error = 0
        frame = 0
        episode_count += 1
    elif np.around(next_s[0], 4) <= 0:
        ax.set_title("Car went out of bounds!")
        next_s = np.array([np.argmin(mountain_vals) * T, 0])  # State as a vector initialized in valley with velocity = 0
        print("Episode", episode_count, "finished")
        print("Total reward:", total_error)
        error_list.append(total_error)
        total_error = 0
        frame = 0
        episode_count += 1
    if np.around(next_s[0], 4) <= 0 or np.around(next_s[0], 4) >= L:
        error = ((L - s[0]) ** 2) - Q(s, a_t)
    else:
        error = ((L - s[0]) ** 2) + Q(next_s, next_a) - Q(s, a_t)
    total_error += error
    #index_x = nearest_x(cx, s[0])
    #index_y = nearest_v(cv, s[1])
    #index = double_to_single_index(index_x, index_y)
    if a_t == -max_a:
        w[0] = [sum(x) for x in zip(w[0], learning_rate * error * feature_vect)]
    elif a_t == max_a:
        w[1] = [sum(x) for x in zip(w[1], learning_rate * error * feature_vect)]
    a_t = next_a
    s = next_s
    return car,


def train_with_plot():
    global ani
    global car
    global s
    global frame
    global fig
    global ax
    # create a figure with an axes
    fig, ax = plt.subplots()
    l = plt.plot(x_vals, mountain_vals, 'g')
    ax.fill_between(x_vals, 0, mountain_vals, facecolor='green')
    goal = plt.plot([L], [h], 'ro')
    # set the axes limits
    ax.axis([0, L, 0, h])
    s = np.array([np.argmin(mountain_vals) * T, 0])  # State as a vector initialized in valley with velocity = 0
    # create a point in the axes
    car, = plt.plot([s[0]], [mountains(s[0])], 'bo')
    frame = 0
    ani = FuncAnimation(fig, update, interval=interval, frames=gen, blit=True, repeat=False, save_count=sys.maxsize)
    plt.show()
    #mywriter = FFMpegWriter()
    #ani.save('myanimation.mp4', writer=mywriter)


# Does not work yet, since error is not correct (too low, sometimes even negative)
def train_without_plot(time_limit):
    global s
    global frame
    global episode_count
    global avg_error
    global total_error
    global error_list
    global a_t
    global w
    s = np.array([np.argmin(mountain_vals) * T, 0])  # State as a vector initialized in valley with velocity = 0
    for episode in range(episodes):
        total_error = 0
        terminate = False
        s = np.array([np.argmin(mountain_vals) * T, 0])  # State as a vector initialized in valley with velocity = 0
        for t in range(time_limit*10000):
            next_a = next_action(s)
            next_s = next_state(s, next_a)
            if np.around(next_s[0], 4) >= L:
                print("Crest reached!")
                error_list.append(total_error)
                terminate = True
            elif t == time_limit:
                print("Time limit reached!")
                error_list.append(total_error)
                terminate = True
            elif np.around(next_s[0], 4) <= 0:
                print("Car went out of bounds!")
                error_list.append(total_error)
                terminate = True
            if np.around(next_s[0], 4) <= 0 or np.around(next_s[0], 4) >= L:
                error = ((L - s[0]) ** 2) - Q(s, a_t)
            else:
                error = ((L - s[0]) ** 2) + Q(next_s, next_a) - Q(s, a_t)
            total_error += error
            # index_x = nearest_x(cx, s[0])
            # index_y = nearest_v(cv, s[1])
            # index = double_to_single_index(index_x, index_y)
            if a_t == -max_a:
                w[0] = [sum(x) for x in zip(w[0], learning_rate * error * feature_vect)]
            elif a_t == max_a:
                w[1] = [sum(x) for x in zip(w[1], learning_rate * error * feature_vect)]
            a_t = next_a
            s = next_s
            if terminate:
                break
            time.sleep(1/10000)
        print("Episode", episode, "finished")
        print("Total reward:", total_error)
    episode_count = episodes


train_with_plot()
#train_without_plot(5)
episode_x_vals = np.linspace(0, episode_count, episode_count)
plt.plot(episode_x_vals, error_list, 'r', label='Total error per episode')
plt.title("Total error per episode")
plt.xlabel('Episode')
plt.ylabel('Total error per episode')
plt.show()