import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import *


# Parameters
max_accel = 4
g = 9.8  # m/s^2
m = 1    # kg
L = 4  # Max x val (mountain "width") in m
h = 1  # Mountain height in m
T = 0.05
time_limit = 5  # in seconds
interval = 20

time_limit = (time_limit*1000)/interval

A = np.array([-max_accel, max_accel])


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
    if state[1] >= 0:
        return 1
    else:
        return -1


x_vals = np.arange(0.0, L, T)
mountain_vals = [mountains(x) for x in x_vals]


# create a figure with an axes
fig, ax = plt.subplots()
l = plt.plot(x_vals, mountain_vals)
goal = plt.plot([L], [h], 'ro')
# set the axes limits
ax.axis([0, L, 0, h])
s = np.array([np.argmin(mountain_vals) * T, 0])  # State as a vector initialized in valley with velocity = 0
# create a point in the axes
car, = plt.plot([s[0]], [mountains(s[0])], 'go')
frame = 0


# Generator function that keeps returning frame numbers until mountain crest is reached or time_limit is reached
def gen():
    global s
    global frame
    global ax
    while s[0] <= L+0.004 and frame < time_limit:
        frame += 1
        yield frame


# Updating function, to be repeatedly called by the animation
def update(framenumber):
    # obtain point coordinates
    global s
    y = mountains(s[0])
    car.set_data(s[0], y)
    # set point's coordinates
    a_t = next_action(s)
    s = next_state(s, a_t)
    if np.around(s[0], 4) >= L:
        ax.set_title("Crest reached!")
    elif frame == time_limit:
        ax.set_title("Time limit reached!")
    return car,


ani = FuncAnimation(fig, update, interval=interval, frames=gen, blit=True, repeat=False)
plt.show()

#for t in range(5000):
#    print("x_t:", s[0], "v_t:", s[1])
#    if s[0] == 4:
#        print("Reached crest!")
#        break
#    a_t = next_action(s)
#    s = next_state(s, a_t)

