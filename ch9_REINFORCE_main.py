import gym

import torch
from torch.distributions import Categorical

from ch9_REINFORCE import REINFORCE

def main():
    env = gym.make('CartPole-v1')
    pi = REINFORCE()
    score = 0.0
    print_interval = 20
    
    for n_epi in range(2001):
        s, _ = env.reset()
        done = False
        
        while not done: 
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample() 
            s_prime, r, done, truncated, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            
        pi.train_net() 
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()
