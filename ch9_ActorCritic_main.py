import gym

import torch
from torch.distributions import Categorical

from ch9_ActorCritic import TDActorCritic

#Hyperparameters
n_rollout = 10

def main():  
    env = gym.make('CartPole-v1')
    model = TDActorCritic()    
    print_interval = 20
    score = 0.0

    for n_epi in range(1001):
        done = False
        s, _ = env.reset()
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float()) # softmax_dim=0
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s,a,r,s_prime,done))
                
                s = s_prime
                score += r
                
                if done:
                    break                     
            
            model.train_net()
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()