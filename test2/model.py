from random import randint
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from pathlib import Path
from main import args
import numpy as np
import random
import sys

# define some variables
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

class ReplayBuffer(object):
    '''
    replay buffer
    
    '''
    
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque()
        self.buffer_size = replay_buffer_size
        self.batch_size = batch_size

    def clear(self):
        self.buffer = deque()
    
    def __len__(self):
        return len(self.buffer)
        
                
    def accumulate(self, frames, action, reward, next_frames):
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()
        self.buffer.append(self.transision(frames, action, reward, next_frames))
        
    def sampling(self):
        '''
        배치 크기만큼 샘플링
        '''
        state, action, reward, next_state = zip(*random.sample(self.buffer, self.batch_size))
        state = torch.stack(state).to(device)
        next_state = torch.stack(next_state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        return state, action, reward, next_state
    
    
class Network(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.convlayer = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                                       nn.batchnorm2d(32),
                                       nn.Conv2d(32, 64, 4, stride=2),
                                       nn.batchnorm2d(64),
                                       nn.Conv2d(64, 64, 3, stride=1),
                                       nn.batchnorm2d(64))
        self.fc = nn.Sequential(nn.Linear(3136, 512),
                                nn.leaky_relu(),
                                nn.Linear(512, num_actions),
                                nn.leaky_relu())

        
    def forward(self, x):
        x = self.convlayer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
    
class Agent(nn.Module):
    def __init__(self, num_actions, device = 'cuda:1' if torch.cuda.is_available() else 'cpu:0')
        super(Agent, self).__init__()
        self.num_actions = num_actions
        self.buffer = ReplayBuffer(replay_buffer_size, batch_size)
        self.q_network = Network(num_actions).to(device)
        self.target_network = Network(num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.num_actions)
        else:
            state = torch.FloatTensor(state).to(device)
            return self.q_network(state).argmax().item()
        
    def train(self):
        if len(self.buffer) < batch_size:
            return 0.0
        
        state, action, reward, next_state = self.buffer.sampling()
        
        q_value = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = self.target_network(next_state).max(1)[0]
        expected_q_value = reward + gamma*next_q_value
        
        loss = F.smooth_l1_loss(q_value, expected_q_value.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)