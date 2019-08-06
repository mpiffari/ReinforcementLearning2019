import matplotlib.pylab as plt
import numpy as np
import random
from math import *




master_seed = 42
half_interval_length = 4
bandit_amount = 10
arm_pulls = 1000
min_interval = -4
max_interval = 4

class Bandit():
    def __init__(self, mean):
        self.mean = mean
    def pull(self):
        return random.randint(self.mean-half_interval_length, self.mean+half_interval_length)
random.seed(master_seed)


#Epsilon-greedy algorithm
def epsilon_greedy():
    bandit_list = []
    for i in range(10):
        bandit_list.append(Bandit(random.randint(min_interval, max_interval)))

    total_reward = 0
    epsilon = 0.8
    Q = [0 for x in range(bandit_amount)]
    N = [0 for x in range(bandit_amount)]
    avg_reward_list = []
    for i in range(arm_pulls):
        action = random.random() #random.uniform(0,1)
        if action < epsilon:                        #With prob. epsilon: explore
            index = random.randint(0,bandit_amount-1)
        else:
            index = np.argmax(Q)
        chosen_bandit = bandit_list[index]
        reward = chosen_bandit.pull()
        total_reward += reward
        avg_reward_list.append(total_reward/(i+1))
        N[index] += 1
        Q[index] = Q[index] + (1 / N[index]) * (reward - Q[index])

    plt.plot([x for x in range(arm_pulls)], ucb(), 'ro', label='Average reward')
    plt.title('The best plot ever with epsilon =%f' % epsilon)
    plt.xlabel('Arm pulls')
    plt.ylabel('Average reward')
    plt.legend()
    plt.show()


#Upper Confidence Bound (UCB)
def ucb():
    bandit_list = []
    for i in range(10):
        bandit_list.append(Bandit(random.randint(min_interval, max_interval)))

    total_reward = 0
    avg_reward_list = []
    c = 2
    Q = [0 for x in range(bandit_amount)]
    N = [0 for x in range(bandit_amount)]
    ubc_list = [0 for x in range(bandit_amount)]
    for t in range(0, arm_pulls):
        for j in range(len(ubc_list)):
            if N[j] == 0:
                ubc_list[j] = 10000
            else:
                ubc_list[j] = Q[j] + (c * sqrt((log(j + 1) / N[j])))
        index = np.argmax(ubc_list)
        chosen_bandit = bandit_list[index]
        reward = chosen_bandit.pull()
        total_reward += reward
        avg_reward_list.append(total_reward / (t+1))
        N[index] += 1
    plt.plot([x for x in range(arm_pulls)], avg_reward_list, 'ro', label='Average reward')
    plt.title('Ubc with c=%f'  %c)
    plt.xlabel('Arm pulls')
    plt.ylabel('Average reward')
    plt.legend()
    plt.show()



ucb()
#’bo’ is for blue dot, ‘b’ is for solid blue line
#plt.plot([x for x in range(10)], Q, 'bo', label='Expected reward')
#plt.plot([x for x in range(10)], N, 'yo', label='Number of arm pulls')


