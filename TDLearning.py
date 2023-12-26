from src.environment import GridWorld
from src.agent import Agent

def temporal_difference():
    env = GridWorld()
    agent = Agent()
    data = [[0] * 4 for _ in range(4)]
    gamma = 1.0
    reward = -1
    alpha = 0.01

    for _ in range(50000):
        done = False
        while not done:
            x, y = env.get_state()
            action = agent.select_action()
            (x_prime, y_prime), reward, done = env.step(action)
            x_prime, y_prime = env.get_state()
            data[x][y] = data[x][y] + alpha*(reward+gamma*data[x_prime][y_prime]-data[x][y])
        env.reset()
            
    return data

if __name__ == '__main__':
    TDResult = temporal_difference()
    for row in TDResult:
        print(row)