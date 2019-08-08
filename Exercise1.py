import matplotlib.pyplot as plt
import numpy as np
import random
import statistics
from math import *

master_seed = 42
half_interval_length = 4
bandit_amount = 10
arm_pulls = 1000
min_interval = 0
max_interval = 5

class Bandit():
    rewards_obtained = []
    def __init__(self, mean):
        self.mean = mean
    def pull(self):
        return random.randint(self.mean-half_interval_length, self.mean+half_interval_length)
#random.seed(master_seed)



bandit_list = []
for i in range(bandit_amount):
    bandit_list.append(Bandit(random.randint(min_interval, max_interval)))

#Epsilon-greedy algorithm
def epsilon_greedy(epsilon):

    total_reward = 0
    epsilon = epsilon
    Q = [0 for x in range(bandit_amount)]
    N = [0 for x in range(bandit_amount)]
    avg_reward_list = []
    for i in range(arm_pulls):
        action = random.random()
        if action < epsilon:                        #With prob. epsilon: explore
            index = random.randint(0,bandit_amount-1)
        else:                                       #With prob. epsilon: exploit
            index = np.argmax(Q)
        chosen_bandit = bandit_list[index]
        reward = chosen_bandit.pull()
        total_reward += reward
        chosen_bandit.rewards_obtained.append(reward)
        avg_reward_list.append(total_reward/(i+1))
        N[index] += 1
        Q[index] = Q[index] + (1 / N[index]) * (reward - Q[index])

    #######################################
    return avg_reward_list
    # plt.plot([x for x in range(arm_pulls)], avg_reward_list, 'go', label='Epsilon Greedy')
    # plt.title('Average performance with epsilon = %f' % epsilon)
    # plt.xlabel('Arm pulls')
    # plt.ylabel('Average reward')
    #######################################

    #######################################
    # highest_played_bandit = np.argmax(N)
    # plt.hist(bandit_list[highest_played_bandit].rewards_obtained, bins=(len(bandit_list[highest_played_bandit].rewards_obtained)))
    # plt.title('Reward payed by bandit number %f' % (highest_played_bandit+1))
    # plt.xlabel('Bandits')
    # plt.ylabel('Reward distribution')
    #######################################

    #######################################
    # plt.plot([x for x in range(bandit_amount)], N, 'go', label='Number  of pulls for bandit')
    # plt.title('Number of pull for each bandit with epsilon =%f' % epsilon)
    # plt.xlabel('Bandits')
    # plt.ylabel('# pulls')
    #######################################


#Greedy with optimistic values
def optimistic_greedy(epsilon, initial_val):

    total_reward = 0
    epsilon = epsilon
    optimistic_init_value = initial_val
    Q = [optimistic_init_value for x in range(bandit_amount)]
    N = [0 for x in range(bandit_amount)]
    avg_reward_list = []
    for i in range(arm_pulls):
        action = random.random() #random.uniform(0,1)
        if action < epsilon:                        #With prob. epsilon: explore
            index = random.randint(0,bandit_amount-1)
        else:                                       #With prob. epsilon: exploit
            index = np.argmax(Q)
        chosen_bandit = bandit_list[index]
        reward = chosen_bandit.pull()
        total_reward += reward
        avg_reward_list.append(total_reward/(i+1))
        N[index] += 1
        Q[index] = Q[index] + (1 / N[index]) * (reward - Q[index])

    return avg_reward_list
    #####################################
    # plt.plot([x for x in range(arm_pulls)], avg_reward_list, 'ro', label='Optimistic Greedy')
    # plt.title('Optimistic initial value = %f' %optimistic_init_value)
    # plt.xlabel('Arm pulls')
    # plt.ylabel('Average reward')
    #####################################

#Upper Confidence Bound (UCB)
def ucb(c):
    total_reward = 0
    avg_reward_list = []
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
        Q[index] = Q[index] + (1 / N[index]) * (reward - Q[index])

    ##################################
    return avg_reward_list
    ####################################
    #plt.plot([x for x in range(arm_pulls)], avg_reward_list, 'bo', label='UCB')
    #plt.title('Ubc with c= %f' % c)
    #plt.xlabel('Arm pulls')
    #plt.ylabel('Average reward')
    #plt.legend()


################ Single call of each algorithm ####################
#ucb()
#optimistic_greedy()
#epsilon_greedy(1)
###################################################################

################ Automatic test of epsilon greedy #################
def test_epsilon_greedy():
    number_of_test = 500
    vector = []
    w = arm_pulls
    matrix = [[0 for x in range(w)] for y in range(number_of_test)]

    for i in range(0, number_of_test):
        for j in range(0, w):
            matrix[i][j] = 0

    for k in range(0, 5):
        if k == 0:
            eps = 0
        elif k == 1:
            eps = 1
        elif k == 2:
            eps = 0.01
        elif k == 3:
            eps = 0.1
        elif k == 4:
            eps = 0.5

        for i in range(0, number_of_test):
            matrix[i] = epsilon_greedy(eps)
        for i in range(0, arm_pulls):
            media = statistics.mean([row[i] for row in matrix])
            vector.append(media)

        x = np.linspace(0, arm_pulls, arm_pulls)
        plt.plot(x, vector, label='epsilon = %f' % eps)
        plt.title('Epsilon greedy')
        plt.xlabel('Arm pulls')
        plt.ylabel('Average reward')
        plt.legend()
        vector = []         #Reset of comulated vector
    ##############################################################

    plt.show()

    k = np.linspace(0, bandit_amount, bandit_amount)
    list_of_intervals = []
    for i in range(bandit_amount):
        interval_list = np.linspace(bandit_list[i].mean-half_interval_length, bandit_list[i].mean+half_interval_length,
                                    half_interval_length*2)
        list_of_intervals.append(interval_list)
    for xe, ye in zip(k, list_of_intervals):
        plt.scatter([xe] * len(ye), ye)
    plt.title("Bandit interval distribution")
    plt.xlabel("Bandits")
    plt.xticks(range(bandit_amount+1))
    plt.ylabel("Rewards")
    plt.show()
###################################################################

################ Automatic test of ucb #################
def test_ucb():
    number_of_test = 500
    vector = []
    w = arm_pulls
    matrix = [[0 for x in range(w)] for y in range(number_of_test)]

    for i in range(0, number_of_test):
        for j in range(0, w):
            matrix[i][j] = 0

    for k in range(0, 5):
        if k == 0:
            c = 0
        elif k == 1:
            c = 1
        elif k == 2:
            c = 2
        elif k == 3:
            c = 5
        elif k == 4:
            c = 10

        for i in range(0, number_of_test):
            matrix[i] = ucb(c)
        for i in range(0, arm_pulls):
            media = statistics.mean([row[i] for row in matrix])
            vector.append(media)

        x = np.linspace(0, arm_pulls, arm_pulls)
        plt.plot(x, vector, label='c = %f' % c)
        plt.title('Upper confidence bound')
        plt.xlabel('Arm pulls')
        plt.ylabel('Average reward')
        plt.legend()
        vector = []  # Reset of comulated vector
    ##############################################################

    plt.show()

    k = np.linspace(0, bandit_amount, bandit_amount)
    list_of_intervals = []
    for i in range(bandit_amount):
        interval_list = np.linspace(bandit_list[i].mean - half_interval_length, bandit_list[i].mean + half_interval_length,
                                    half_interval_length * 2)
        list_of_intervals.append(interval_list)
    for xe, ye in zip(k, list_of_intervals):
        plt.scatter([xe] * len(ye), ye)
    plt.title("Bandit interval distribution")
    plt.xlabel("Bandits")
    plt.xticks(range(bandit_amount + 1))
    plt.ylabel("Rewards")
    plt.show()
###########################################################

################ Automatic test of optimistic greedy #################
def test_optimistic_greedy():
    number_of_test = 500
    vector = []
    w = arm_pulls
    matrix = [[0 for x in range(w)] for y in range(number_of_test)]

    for i in range(0, number_of_test):
        for j in range(0, w):
            matrix[i][j] = 0

    for k in range(0, 5):
        if k == 0:
            initial_val = 0
        elif k == 1:
            initial_val = 0.5
        elif k == 2:
            initial_val = 1
        elif k == 3:
            initial_val = 5
        elif k == 4:
            initial_val = 10

        for i in range(0, number_of_test):
            matrix[i] = optimistic_greedy(0, initial_val)
        for i in range(0, arm_pulls):
            media = statistics.mean([row[i] for row in matrix])
            vector.append(media)

        x = np.linspace(0, arm_pulls, arm_pulls)
        plt.plot(x, vector, label='Initial value = %f' % initial_val)
        plt.title('Optimistic greedy')
        plt.xlabel('Arm pulls')
        plt.ylabel('Average reward')
        plt.legend()
        vector = []         #Reset of comulated vector
    ##############################################################

    plt.show()

    k = np.linspace(0, bandit_amount, bandit_amount)
    list_of_intervals = []
    for i in range(bandit_amount):
        interval_list = np.linspace(bandit_list[i].mean-half_interval_length, bandit_list[i].mean+half_interval_length,
                                    half_interval_length*2)
        list_of_intervals.append(interval_list)
    for xe, ye in zip(k, list_of_intervals):
        plt.scatter([xe] * len(ye), ye)
    plt.title("Bandit interval distribution")
    plt.xlabel("Bandits")
    plt.xticks(range(bandit_amount+1))
    plt.ylabel("Rewards")
    plt.show()
###################################################################

################ Automatic comparison of all three algorithms with fixed parameters #################
def compare_algorithms():
    number_of_test = 500
    vector = []
    w = arm_pulls
    epsilon = 0.1
    optimistic_initial_val = 5
    confidence = 2

    matrix = [[0 for x in range(w)] for y in range(number_of_test)]

    for i in range(0, number_of_test):
        for j in range(0, w):
            matrix[i][j] = 0

    for k in range(0, 3):
        if k == 0:
            algorithm = "Epsilon-greedy"
            for i in range(0, number_of_test):
                matrix[i] = epsilon_greedy(epsilon)
        elif k == 1:
            algorithm = "Optimistic-greedy"
            for i in range(0, number_of_test):
                matrix[i] = optimistic_greedy(0, optimistic_initial_val)
        elif k == 2:
            algorithm = "UCB"
            for i in range(0, number_of_test):
                matrix[i] = ucb(confidence)

        for i in range(0, arm_pulls):
            media = statistics.mean([row[i] for row in matrix])
            vector.append(media)

        x = np.linspace(0, arm_pulls, arm_pulls)
        if k == 0:
            plt.plot(x, vector, label=('%s  ε = %5.2f' % (algorithm, epsilon)))
        elif k == 1:
            plt.plot(x, vector, label=('%s  Q₁ = %5.2f' % (algorithm, optimistic_initial_val)))
        elif k == 2:
            plt.plot(x, vector, label=('%s  c = %5.2f' % (algorithm, confidence)))

        plt.title('Performance comparison of 3 bandit algorithms')
        plt.xlabel('Arm pulls')
        plt.ylabel('Average reward')
        plt.legend()
        vector = []         #Reset of comulated vector
    ##############################################################

    plt.show()

    k = np.linspace(0, bandit_amount, bandit_amount)
    list_of_intervals = []
    for i in range(bandit_amount):
        interval_list = np.linspace(bandit_list[i].mean-half_interval_length, bandit_list[i].mean+half_interval_length,
                                    half_interval_length*2)
        list_of_intervals.append(interval_list)
    for xe, ye in zip(k, list_of_intervals):
        plt.scatter([xe] * len(ye), ye)
    plt.title("Bandit interval distribution")
    plt.xlabel("Bandits")
    plt.xticks(range(bandit_amount+1))
    plt.ylabel("Rewards")
    plt.show()
###################################################################
compare_algorithms()
#plt.show()