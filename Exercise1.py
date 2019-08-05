import matplotlib.pylab as plt
import numpy as np
import random




master_seed = 42
half_interval_length = 10
bandit_amount = 10
arm_pulls = 100

class Bandit():
    def __init__(self, mean):
        self.mean = mean
    def pull(self):
        return random.randint(self.mean-half_interval_length, self.mean+half_interval_length)
random.seed(master_seed)
bandit_list = []
for i in range(10):
    bandit_list.append(Bandit(random.randint(-10, 11)))

#Epsilon-greedy algorithm
epsilon = 1
Q = [0 for x in range(10)]
N = [0 for x in range(10)]
for i in range(arm_pulls):
    action = random.uniform(0, 1)
    if action < epsilon:                        #With prob. epsilon: explore
        index = random.randint(0,bandit_amount-1)
    else:
        index = np.argmax(Q)
    chosen_bandit = bandit_list[index]
    reward = chosen_bandit.pull()
    N[index] += 1
    Q[index] = Q[index] + (1 / N[index]) * (reward - Q[index])

#’bo’ is for blue dot, ‘b’ is for solid blue line
plt.plot([x for x in range(10)], Q, 'bo', label='Expected reward')
plt.plot([x for x in range(10)], N, 'yo', label='Number of arm pulls')
plt.title('The best plot ever with epsilon =%f' %epsilon)
plt.xlabel('Bandit')
plt.ylabel('Q')
plt.legend()
plt.show()

