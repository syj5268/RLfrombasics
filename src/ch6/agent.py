import numpy as np
import random

class Agent():
    def __init__(self):
        self.q_table=np.zeros((5,7,4)) # q-value를 저장하는 변수, 모두 0으로 초기화
        self.eps=0.9 # 랜덤 행동을 할 확률
        self.gamma=1.0
        self.alpha=0.01 # learning rate

    def select_action(self,s):
        # eps-greedy로 액션을 선택
        x,y=s
        coin=random.random()
        if coin<self.eps:
            action=random.randint(0,3)
        else:
            action_val=self.q_table[x,y,:]
            action=np.argmax(action_val)
        return action
    
    def update_table(self,history):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q-table을 업데이트
        cum_reward=0
        for transition in history[::-1]:
            s,a,r,s_prime=transition
            x,y=s
            # 몬테카를로 방식
            self.q_table[x,y,a] = self.q_table[x,y,a]+self.alpha*(cum_reward-self.q_table[x,y,a])
            cum_reward= self.gamma*cum_reward+r
    
    def anneal_eps(self):
        self.eps-=0.03
        self.eps=max(self.eps,0.1)

    def show_table(self):
        q_lst=self.q_table.tolist()
        data = np.zeros((5,7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx,col_idx] = action
        return data