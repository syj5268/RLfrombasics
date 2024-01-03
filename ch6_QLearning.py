from src.ch6.environment import GridWorld
from src.ch6.agent import Agent
import numpy as np

# QLearning Agent
class QAgent(Agent):
    def update_table(self, transition):
        s,a,r,s_prime=transition
        x,y=s
        next_x,next_y=s_prime
        self.q_table[x,y,a]=self.q_table[x,y,a]+self.alpha*(r+np.amax(self.q_table[next_x,next_y,:])-self.q_table[x,y,a])
    
    def anneal_eps(self):
        self.eps-=0.01
        self.eps=max(self.eps,0.2)

def TD_main(): 
    env= GridWorld()
    agent= QAgent() # Agent만 다름
    for n_epi in range(1000):
        done=False

        s = env.reset() # SARSA와 동일한 코드
        while not done:
            a = agent.select_action(s)
            s_prime,r,done = env.step(a)
            agent.update_table((s,a,r,s_prime))
            s=s_prime
        agent.anneal_eps()
    
    return agent.show_table()

if __name__ == '__main__':
    Q_action = TD_main()
    print(Q_action)