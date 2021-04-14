import numpy as np
import matplotlib.pyplot as plt
from camparing_epsilon import run_experiment as eps_exp


class Bandit:
    def __init__(self,m,upper_limit):
        self.m = m
        self.mean = upper_limit
        self.N = 1

    def pull(self):
        return np.random.randn() + self.m

    def update(self,x):
        self.N += 1
        self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x


def run_experiment(m1,m2,m3,upper_limit,N):
    bandits = [Bandit(m1,upper_limit), Bandit(m2,upper_limit), Bandit(m3,upper_limit)]

    data = np.empty(N)
    for i in range(N):
        #optimistic initial
        j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        #for the plot
        data[i] = x
    cumulative_average =np.cumsum(data)/(np.arange(N)+1)

    #plot moving average ctr

    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for bandit  in bandits:
        print(bandit.mean)

    return cumulative_average


if __name__ == '__main__':
    c_1  = run_experiment(1.0,2.0,3.0,10,100000)
    c_05  = run_experiment(1.0,2.0,3.0,0.05,100000)
    c_e= eps_exp(1.0,2.0,3.0,0.1,100000)


    #log scale plot
    plt.title(label="log_scale")
    plt.plot(c_1,label= 'optimistic = 10')
    plt.plot(c_e,label= 'eps = 0.1')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.title(label="linear plot")
    plt.plot(c_1,label= 'optimistic = 10')
    plt.plot(c_e, label='eps = 0.1')

    plt.legend()
    plt.show()