import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class REINFORCE(nn.Module):
    def __init__(self):
        super(REINFORCE, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128) # 4 inputs
        self.fc2 = nn.Linear(128, 2) # 2 outputs (left or right)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []