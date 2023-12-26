from src.ch5.environment import GridWorld
from src.ch5.agent import Agent

def monte_carlo():
    env = GridWorld()
    agent = Agent()
    data = [[0] * 4 for _ in range(4)]
    gamma = 1.0
    reward = -1
    alpha = 0.001

    for _ in range(50000):
        done = False
        history = []

        while not done:
            action = agent.select_action()
            (x,y), reward, done = env.step(action)
            history.append((x,y,reward))
        env.reset()

        cum_reward = 0
        for transition in history[::-1]:
            x, y, reward = transition
            data[x][y] = data[x][y] + alpha*(cum_reward-data[x][y])
            cum_reward = reward + gamma*cum_reward  # 책에 오타가 있어 수정하였습니다

    return data  

if __name__ == '__main__':
    MCResult = monte_carlo()
    for row in MCResult:
        print(row)