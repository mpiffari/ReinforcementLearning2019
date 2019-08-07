import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
## Basic implementation of iterative policy evaluation (see p. 75 in Sutton
 # and Barto (2018) "Reinformcement Learning, an Introduction") for a deterministic
 # 2D lattice environment.
 #
 # Note: works only for determinsitic policies as the "GetAction()" function
 # returns only a single action. It is, however, straightforward to
 # extend.
 #

#Dimensions of the environment

COLUMS = 4
ROWS = 3

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Environment -- spaces: agent can move, "+": reward, "-": punishment.

environment = [[' ', ' ', ' ', ' '],
               ['+', '#', ' ', '-'],
               [' ', ' ', ' ', ' ']]

Q_matrix = [[[0, 0, 0, 0] for x in range(COLUMS)] for y in range(ROWS)]


# Current estimate of state values under the current policy:
V = [[0.0 for x in range(COLUMS)] for y in range(ROWS)]


class State():
    def __init__(self, row, colum, is_outside_environment):
        self.row = row
        self.colum = colum
        self.is_outside_environment = is_outside_environment

    def is_terminal_state(self):
        return self.colum == self.row == -1 and self.is_outside_environment



# Theta: the threshold for determining the accuracy of the estimation
theta = 0.01
TERMINAL_STATE = State(-1, -1, True)
discount_rate = 0.9

# Get the next state given a current state s and an action a:
def get_next_state(s_param, action):
    s = copy.deepcopy(s_param)
    if environment[s.row][s.colum] == '+':
        return TERMINAL_STATE

    if action == UP:
        s.row -= 1
    elif action == DOWN:
        s.row += 1
    elif action == LEFT:
        s.colum -= 1
    elif action == RIGHT:
        s.colum += 1

    if s.colum < 0 or s.row < 0 or s.colum >= COLUMS or s.row >= ROWS or environment[s.row][s.colum] == '#':
        return s_param

    s.is_outside_environment = False
    return s


# Get the reward given a state and an action:
def get_reward(s, action):

    next_state = get_next_state(s, action)
    if next_state.is_outside_environment:
        return 0
    else:
        if environment[next_state.row][next_state.colum] == '+':
            return 1.0
        if environment[next_state.row][next_state.colum] == '-':
            return -1.0
        else:
            return 0


# OLD DESCRIPTION: Get the next action according to the current policy:
# NEW: Computes the estimates of surrounding states to find the best action to return with probability 1-epsilon, or
# choose a random action with probability epsilon
def get_next_action(s):
    seed = 42
    epsilon = 0.10
    #random.seed(seed)

    possible_actions = [UP, DOWN, LEFT, RIGHT]
    max_val = -math.inf
    best_action = RIGHT
    probability = random.random()

    if probability < epsilon:
        action = random.randint(0, 3)  # Choose an action at random
        return action
    else:                              # Otherwise, greedily choose best action based on estimated reward of surrounding states
        #for action in possible_actions:
        #    next_state = get_next_state(s, action)
        #    reward = get_reward(s, action)
        #    if not next_state.is_outside_environment:
        #        if V[next_state.y][next_state.x] + reward > max_val:
        #            max_val = V[next_state.y][next_state.x] + reward
        #            best_action = action
        #return best_action
        return np.argmax(Q_matrix[s.row][s.colum])


# Print the environment with border around:
def print_environment():
    for y in range(-1, ROWS+1):
        for x in range(-1, COLUMS+1):
            if y < 0 or y >= ROWS or x < 0 or x >= COLUMS:
                print("#", end='')
            else:
                print(environment[y][x], end='')
        print("")


# Print the current estimate of state values:
def print_Q_values():
    for y in range(0, ROWS):
        for x in range(0, COLUMS):
            print("[", end = ' ')
            for action_value in Q_matrix[y][x]:
                print("%5.2f" %action_value, end=' ')
            print("]", end = ' ')
        print("")
    print("")


print("Environment:")
print_environment()


episode_amount = 200
reward_for_each_episode = [0 for x in range(episode_amount)]
index = 0
state = State(2, 0, False)
# Start of estimation loop
for j in range(episode_amount):
    for i in range(j):
        state = State(2, 0, False)
        while True:
            delta = 0

            # Perform a full sweep over the whole state space:

            #for y in range(0, ROWS):
            #    for x in range(0, COLUMS):
        #
            #        state.x = x
            #        state.y = y
            #        if environment[y][x] == ' ':
            #            v = V[y][x]
            #            a = get_next_action(state)
            #            reward = get_reward(state, a)
            #            next_s = get_next_state(state, a)
            #            if not next_s.is_outside_environment:
            #                V[y][x] = reward + discount_rate * V[next_s.y][next_s.x]
            #            delta = max(delta, abs(v - V[y][x]))
            #print("Sweep #", sweep, ", delta:", delta)
            #sweep += 1
            #print_state_values()

            step_size = 0.1
            discount_rate = 0.9

            if state.is_terminal_state():  # If we reach terminal state, stop
                print("Terminal state reached")
                break

            a = get_next_action(state)
            reward = get_reward(state, a)

            reward_for_each_episode[index] = reward_for_each_episode[index] + reward

            next_s = get_next_state(state, a)
            if (next_s == state):
                print("(", state.row, ",", state.colum, ")"", Reward:", reward, "(Bump)")
            else:
                print("(", state.row, ",", state.colum, ")"", Reward:", reward)

            if not next_s.is_outside_environment:
                Q_matrix[state.row][state.colum][a] = Q_matrix[state.row][state.colum][a] + \
                                                step_size*(reward + (discount_rate*max(Q_matrix[next_s.row][next_s.colum]))
                                                           - Q_matrix[state.row][state.colum][a])    # Update Q(S,A)
            state = next_s


        print_Q_values() # Maybe print at each step????!?!??

    index += 1

##################################
plt.plot([x for x in range(episode_amount)], reward_for_each_episode, 'bo', label='UCB')
plt.title('Performance of Q-learning')
plt.xlabel('Episode')
plt.ylabel('Sum of rewards during episode')
plt.legend()
plt.show()
####################################