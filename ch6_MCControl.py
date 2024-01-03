from src.ch6.environment import GridWorld
from src.ch6.agent import Agent

def MC_main():
    env= GridWorld()
    agent= Agent()
    for n_epi in range(1000):
        done=False
        history=[]

        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime,r,done = env.step(a)
            history.append((s,a,r,s_prime))
            s=s_prime
        agent.update_table(history)
        agent.anneal_eps()
    
    return agent.show_table()  

if __name__ == '__main__':
    MC_action = MC_main()
    print(MC_action)