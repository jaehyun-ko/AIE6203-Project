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
    
    def __init__(self, replay_buffer_size, batch_size):
        self.buffer = deque()
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.transition = namedtuple('sample', ('state', 'action', 'reward', 'next_state'))    

        
    def clear(self):
        self.buffer = deque()
    
    def __len__(self):
        return len(self.buffer)
    
                
    def accumulate(self, frames, action, reward, next_frames):
        if len(self.buffer) > self.replay_buffer_size:
            self.buffer.popleft()
        self.buffer.append(self.transition(frames, action, reward, next_frames))
        
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
    def __init__(self, in_dim, num_actions):
        super(Network, self).__init__()
        # self.convlayer = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
        #                                nn.BatchNorm2d(32),
        #                                nn.Conv2d(32, 64, 4, stride=2),
        #                                nn.BatchNorm2d(64),
        #                                nn.Conv2d(64, 64, 3, stride=1),
        #                                nn.BatchNorm2d(64))
        self.fc = nn.Sequential(nn.Linear(in_dim, 256),
                                nn.LeakyReLU(),
                                nn.Linear(256, 256),
                                nn.LeakyReLU(),
                                nn.Linear(256, num_actions)
                                )

        
    def forward(self, x):
        # x = self.convlayer(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def init_weights(layer):
    if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
        torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        layer.bias.data.fill_(0.01)
    
    
class Agent(nn.Module):
    def __init__(self, num_indim, num_actions, replay_buffer):
        super(Agent, self).__init__()
        self.criterion = nn.MSELoss()
        self.epsilon = args.epsilon
        self.start_epsilon = self.epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_step = 5000
        self.num_actions = num_actions
        self.buffer = replay_buffer
        self.q_network = Network(num_indim, num_actions).to(device)
        self.target_network = Network(num_indim, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.num_actions)
        else:
            state = torch.FloatTensor(state).to(device)
            return self.q_network(state).argmax().item()
    
    def save_model(self, model_path = args.model_path):
        torch.save({"state_dict": self.q_network.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, model_path)
    
    def load_model(self, model_path = args.model_path):
        checkpoint = torch.load(model_path)
        self.q_network.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
    
    def epsilon_greedy(self, frame):        
        if torch.rand(1) <= self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            if device.type == 'cuda':
                frame = torch.cuda.FloatTensor(frame)
            else:
                frame = torch.FloatTensor(frame)
            action = self.q_network(frame.unsqueeze(0))
            action = action.max(1)[1].item()
        
        return action
    
    def take_action(self, frame):
        return self.epsilon_greedy(frame)
    
    
    def check_buffer(self):
        if len(self.buffer) > self.buffer.replay_buffer_size:
            return True
        return False        
        
    def train_step(self):
        '''
        벨만방정식.
        '''
        frames, actions, rewards, next_frames = self.buffer.sampling()
        
        # get q values
        q = self.q_network(frames)
        q = torch.gather(q, index=actions.unsqueeze(1), dim = 1)
        
        # q_next from target network
        q_next = self.target_network(next_frames).detach()
        q_next = q_next.max(1)[0]
        
        # bellman equation
        target = rewards + args.gamma * q_next
        target = target.view(32, 1)
        
        # loss
        loss = self.criterion(q, target)
        loss.backward()
        
        return loss
        
    def update(self, episode, theta = 50):
        if episode % theta == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.min_epsilon + (self.start_epsilon - self.min_epsilon) \
            * torch.exp(torch.tensor(-1. * episode / self.episode_step))
            
            
    def eval():
        '''
        모델 학습 중단
        action 선택 시 greedy하게만 선택
        '''
        self.q_network.eval()
        self.epsilon = 0.0
        
    def plot(self, history):
        """
        Show graph of performance
        Really useful in learning
        """
        if len(history) > 10:
            del history[0]

        x, y = zip(*history)
        plt.plot(x, y, color='r')
        plt.title('Score per episode')
        plt.xlabel('Episodes')
        plt.ylabel('Scores')
        plt.draw()
        plt.pause(1e-8)