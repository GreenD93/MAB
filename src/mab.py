from scipy.stats import bernoulli
import numpy as np

class Env:
    def __init__(self):
        pass
    def react(prob):
        did_click = bernoulli.rvs(prob)
        return did_click
    
class BeroulliTSAgent:
    def __init__(self):
        self.counts = [0 for _ in range(n_arm)] 
        self.wins = [0 for _ in range(n_arm)]
        
    def get_arm(self):
        beta = lambda N, a: np.random.beta(a+1, N - a + 1)
        result = [beta(self.counts[i], self.wins[i]) for i in range(n_arm)]
        arm, prob = result.index(max(result)), max(result)
        return arm, prob
    
    def update(self, arm, reward):
        self.counts[arm] = self.counts[arm] + 1
        self.wins[arm] = self.wins[arm] + reward

n_arm = 3

agent = BeroulliTSAgent()
# select arm
arm, prob = agent.get_arm()
# reaction
reward = Env.react(prob)
# update beta distribution
agent.update(arm, reward)