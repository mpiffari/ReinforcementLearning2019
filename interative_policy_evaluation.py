import copy
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
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# Environment -- spaces: agent can move, "+": reward, "-": punishment.

environment = [[' ', ' ', ' ', '+'],
               [' ', '#', ' ', '-'],
               [' ', ' ', ' ', ' ']]


# Current estimate of state values under the current policy:
V = [[0.0 for x in range(COLUMS)] for y in range(ROWS)]


class State():
    def __init__(self, x, y, is_outside_environment):
        self.x = x
        self.y = y
        self.is_outside_environment = is_outside_environment


TERMINAL_STATE = State(-1, -1, True)

discount_rate = 0.9

# Theta: the threshold for determining the accuracy of the estimation
theta = 0.01


# Get the next state given a current state s and an action a:
def get_next_state(s_param, action):
    s = copy.deepcopy(s_param)
    if environment[s.y][s.x] != ' ':
        return TERMINAL_STATE

    if action == UP:
        s.y -= 1
    elif action == DOWN:
        s.y += 1
    elif action == LEFT:
        s.x -= 1
    elif action == RIGHT:
        s.x += 1

    if s.x < 0 or s.y < 0 or s.x >= COLUMS or s.y >= ROWS:
        return TERMINAL_STATE

    s.is_outside_environment = False
    return s


# Get the reward given a state and an action:
def get_reward(s, action):

    next_state = get_next_state(s, action)
    if next_state.is_outside_environment:
        return 0
    else:
        if environment[next_state.y][next_state.x] == '+':
            return 1.0
        if environment[next_state.y][next_state.x] == '-':
            return -1.0
        else:
            return 0


# Get the next action according to the current policy:
def get_next_action(s):
    return RIGHT


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
def print_state_values():
    for y in range(0, ROWS):
        for x in range(0, COLUMS):
            print("%5.2f" %V[y][x], end='|')
        print("")
    print("")

print("Environment:")
print_environment()

# Reset all state value estimates to 0:
for y in range(0, ROWS):
    for x in range(0, COLUMS):
        V[y][x] = 0

sweep = 0
state = State(0, 0, False)
# Start of estimation loop
while True:
    delta = 0

    # Perform a full sweep over the whole state space:

    for y in range(0, ROWS):
        for x in range(0, COLUMS):

            state.x = x
            state.y = y
            if environment[y][x] == ' ':
                v = V[y][x]
                a = get_next_action(state)
                reward = get_reward(state, a)
                next_s = get_next_state(state, a)
                if not next_s.is_outside_environment:
                    V[y][x] = reward + discount_rate * V[next_s.y][next_s.x]
                delta = max(delta, abs(v - V[y][x]))
    print("Sweep #", sweep, ", delta:", delta)
    sweep += 1
    print_state_values()
    if (delta <= theta): # Check if our currect estimate is accurate enough.
        break