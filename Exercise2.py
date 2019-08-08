import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Environment -- spaces: agent can move, "+": reward, "-": punishment.

simple_environment = [[' ', ' ', ' ', '+'],
                      [' ', '#', ' ', '-'],
                      [' ', ' ', ' ', ' ']]

complex_environment = [[' ', ' ', ' ', ' ', ' '],
                    [' ', '#', ' ', ' ', '-'],
                    [' ', '#', ' ', '+', ' '],
                    [' ', '#', ' ', ' ', ' '],
                    [' ', '#', ' ', ' ', ' ']]

environment = complex_environment

#Dimensions of the environment
COLUMNS = len(environment[0])
ROWS = len(environment)
start_row = 2
start_column = 0

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
# Algorithm parameters
Q_matrix = [[[0, 0, 0, 0] for x in range(COLUMNS)] for y in range(ROWS)]
action_selected_matrix = [[-1 for x in range(COLUMNS)] for y in range(ROWS)]
step_size = 0.1  # Alpha
discount_rate = 0.9  # Gamma
epsilon = 0.1  # ϵ for greedy
episode_amount = 100
reward_for_each_episode = [0 for x in range(episode_amount)]
index = 0


class State:
    def __init__(self, row, column, is_outside_environment):
        self.row = row
        self.column = column
        self.is_outside_environment = is_outside_environment

    def is_terminal_state(self):
        return self.column == self.row == -1 and self.is_outside_environment


# Constant
TERMINAL_STATE = State(-1, -1, True)


# Get the next state given a current state s and an action a
def get_next_state(s_param, action):
    s = copy.deepcopy(s_param) # to avoid pointer copy
    if environment[s.row][s.column] == '+':
        return TERMINAL_STATE

    if action == UP:
        s.row -= 1
    elif action == DOWN:
        s.row += 1
    elif action == LEFT:
        s.column -= 1
    elif action == RIGHT:
        s.column += 1

    if s.column < 0 or s.row < 0 or s.column >= COLUMNS or s.row >= ROWS or environment[s.row][s.column] == '#':
        return s_param

    s.is_outside_environment = False
    return s


# Get the reward given a state and an action
def get_reward(s, action):
    next_state = get_next_state(s, action)
    if next_state.is_outside_environment:
        return 0
    else:
        if environment[next_state.row][next_state.column] == '+':
            return 1.0
        if environment[next_state.row][next_state.column] == '$':
            return 0.2
        if environment[next_state.row][next_state.column] == '-':
            return -1.0
        else:
            return 0


# NEW: Computes the estimates of surrounding states to find the best action to return with probability 1-epsilon, or
# choose a random action with probability epsilon
def get_next_action(s):
    probability = random.random()
    if probability < epsilon:
        action = random.randint(0, 3)  # Choose an action at random
        return action
    else:                      # Otherwise, greedily choose best action based on estimated reward of surrounding states
        return np.argmax(Q_matrix[s.row][s.column])


def get_action_description(act):
    if act == 0:
        return "UP   "
    elif act == 1:
        return "DOWN "
    elif act == 2:
        return "LEFT "
    elif act == 3:
        return "RIGHT"


# Print the environment with border around:
def print_environment():
    for y in range(-1, ROWS+1):
        for x in range(-1, COLUMNS + 1):
            if y < 0 or y >= ROWS or x < 0 or x >= COLUMNS:
                print("#", end=' ')
            else:
                print(environment[y][x], end=' ')
        print("")


# Print the current estimate of state values:
def print_Q_values():
    for row in range(0, ROWS):
        for column in range(0, COLUMNS):
            print("[", end = ' ')
            for action_value in Q_matrix[row][column]:
                print("%5.2f" %action_value, end=' ')
            print("]", end = ' ')
        print("")
    print("")

# Print the action chose for each state
def print_action_matrix():
    for row in range(0, ROWS):
        for column in range(0, COLUMNS):
            act = action_selected_matrix[row][column]
            if act == 1000:
                print(" T", end=' ')
                # print("Terminal states   ", end=' ')
            if act == -1:
                print(" X", end=' ')
                # print("Punish or wall or not visited   ", end=' ')
            if act == 0:
                print(" ↑", end=' ')
                # print("UP   ", end=' ')
            elif act == 1:
                print(" ↓", end=' ')
                # print("DOWN ", end=' ')
            elif act == 2:
                print("←-", end=' ')
                # print("LEFT ", end=' ')
            elif act == 3:
                print("-→", end=' ')
                # print("RIGHT", end=' ')
        print("")
    print("")

print("Environment: ")
print_environment()
# Estimation loops
if False:
    ############ Estimation loop with fixed number of episode ######################
    for i in range(episode_amount):
        print("Restart")
        start_state = State(start_row, start_column, False)
        while True:
            if start_state.is_terminal_state():  # If we reach terminal state, stop
                print("Terminal state reached")
                print_action_matrix()
                print_Q_values()
                break

            a = get_next_action(start_state)
            reward = get_reward(start_state, a)
            next_s = get_next_state(start_state, a)
            reward_for_each_episode[index] = reward_for_each_episode[index] + reward
            if not next_s.is_outside_environment:
                Q_matrix[start_state.row][start_state.column][a] = Q_matrix[start_state.row][start_state.column][a] + \
                                                                   step_size * (reward + (
                            discount_rate * max(Q_matrix[next_s.row][next_s.column]))
                                                                                - Q_matrix[start_state.row][
                                                                                    start_state.column][
                                                                                    a])  # Update Q(S,A)
            if next_s != start_state:
                print("Move from (", start_state.row, ",", start_state.column, ") to (", next_s.row, ",", next_s.column,
                      ") with this reward:", reward)
                if not(next_s.is_terminal_state()):
                    if environment[next_s.row][next_s.column] == "#":
                        action_selected_matrix[next_s.row][next_s.column] = -1
                    elif environment[next_s.row][next_s.column] == "-":
                        action_selected_matrix[next_s.row][next_s.column] = -1
                        action_selected_matrix[start_state.row][start_state.column] = a
                    elif environment[next_s.row][next_s.column] == "+":
                        action_selected_matrix[next_s.row][next_s.column] = 1000
                        action_selected_matrix[start_state.row][start_state.column] = a
                    else:
                        action_selected_matrix[start_state.row][start_state.column] = a
                        # action_selected_matrix[next_s.row][next_s.column] = a
            else:
                print("Agent BUMPS while going from (", start_state.row, ",", start_state.column, ") going ",
                      get_action_description(a))

            start_state = next_s
else:
    ############ Estimation loop with incremental number of episode ######################
    for j in range(episode_amount):
        for i in range(j):
            print("Restart")
            start_state = State(start_row, start_column, False)
            while True:
                if start_state.is_terminal_state():  # If we reach terminal state, stop
                    print("Terminal state reached")
                    print_action_matrix()
                    print_Q_values()
                    break

                a = get_next_action(start_state)
                reward = get_reward(start_state, a)
                next_s = get_next_state(start_state, a)
                reward_for_each_episode[index] = reward_for_each_episode[index] + reward

                if not next_s.is_outside_environment:
                    Q_matrix[start_state.row][start_state.column][a] = Q_matrix[start_state.row][start_state.column][a] + \
                                                                      step_size * (reward + (discount_rate*max(Q_matrix[next_s.row][next_s.column]))
                                                                                   - Q_matrix[start_state.row][start_state.column][a])    # Update Q(S,A)

                if next_s != start_state:
                    print("Move from (", start_state.row, ",", start_state.column, ") to (", next_s.row, ",",
                          next_s.column,
                          ") with this reward:", reward)
                    if not (next_s.is_terminal_state()):
                        if environment[next_s.row][next_s.column] == "#":
                            action_selected_matrix[next_s.row][next_s.column] = -1
                        elif environment[next_s.row][next_s.column] == "-":
                            action_selected_matrix[next_s.row][next_s.column] = -1
                            action_selected_matrix[start_state.row][start_state.column] = a
                        elif environment[next_s.row][next_s.column] == "+":
                            action_selected_matrix[next_s.row][next_s.column] = 1000
                            action_selected_matrix[start_state.row][start_state.column] = a
                        else:
                            action_selected_matrix[start_state.row][start_state.column] = a
                            # action_selected_matrix[next_s.row][next_s.column] = a
                else:
                    print("Agent BUMPS while going from (", start_state.row, ",", start_state.column, ") going ",
                          get_action_description(a))

                start_state = next_s

        reward_for_each_episode[index] = reward_for_each_episode[index] / (index+1)
        index += 1

    ##################################
    print("Environment:")
    print_environment()
    plt.plot([x for x in range(episode_amount)], reward_for_each_episode, 'b', label='Q-learning')
    plt.title('Performance of Q-learning with ε = %5.2f, γ = %5.2f, α = %5.2f' % (epsilon, discount_rate, step_size))
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()
    plt.show()
    ####################################