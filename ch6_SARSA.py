from src.ch6.environment import GridWorld
from src.ch6.agent import Agent

# SARSA Agent
class SAgent(Agent):
    def update_table(self, transition):
        s,a,r,s_prime=transition
        x,y=s
        next_x,next_y=s_prime
        a_prime=self.select_action(s_prime)
        self.q_table[x,y,a]=self.q_table[x,y,a]+self.alpha*(r+self.q_table[next_x,next_y,a_prime]-self.q_table[x,y,a])

def TD_main():
    env= GridWorld()
    agent= SAgent()
    for n_epi in range(1000):
        done=False

        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime,r,done = env.step(a)
            agent.update_table((s,a,r,s_prime))
            s=s_prime
        agent.anneal_eps()
    
    return agent.show_table()

if __name__ == '__main__':
    S_action = TD_main()
    print(S_action)