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

#named tuple 형태로 transition을 정의
Transition = namedtuple('transition', ('state', 'action', 'reward', 'next_state', 'done', 'raw_state'))
class ReplayBuffer(object):
    '''
    replay buffer

    '''
    
    def __init__(self, replay_buffer_size, batch_size):
        self.buffer = deque()
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.device = device

        
    def clear(self):
        self.buffer = deque()
    
    def __len__(self):
        return len(self.buffer)
    
                
    def accumulate(self, *args):
        if len(self.buffer) > self.replay_buffer_size:
            self.buffer.popleft()
        self.buffer.append(Transition(*args))
        
    def sample_batch(self):
        '''
        배치 크기만큼 샘플링
        '''
        batch = Transition(*zip(*random.sample(self.buffer, self.batch_size)))
        return batch
    
    def sampling(self):
        batch = self.sample_batch()
        state_shape = batch.state[0].shape
        # 언패킹
        state = torch.tensor(batch.state).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(self.device)
        action = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward = torch.tensor(batch.reward).float().unsqueeze(1).to(self.device)
        next_state = torch.tensor(batch.next_state).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(self.device)
        done = torch.tensor(batch.done).float().unsqueeze(1).to(self.device)

        return state, action, reward, next_state, done
    
    
class Network(nn.Module):
    def __init__(self, in_dim, num_actions):
        super(Network, self).__init__()
        self.input_dim = in_dim
        self.num_actions = num_actions
        self.fc_size = 512
        self.convlayer =  nn.Sequential(
                        nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4, padding=2),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                    )
        self.fc = nn.Sequential(
                nn.Linear(64*10*10, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
                )
    
        conv_out_dim = self.calc_conv_out_dim(self.input_dim)
        
    def calc_conv_out_dim(self, input_dim):
        x = torch.zeros(1, *self.input_dim)
        x = self.convlayer(x)
        return int(np.prod(x.shape))
    
        
    def forward(self, x):
        x = self.convlayer(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        module.bias.data.fill_(0.0)
    elif isinstance(module, nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    
    
class Agent(nn.Module):
    def __init__(self, num_indim, num_actions, replay_buffer):
        super(Agent, self).__init__()
        self.batch_size = args.batch_size
        self.criterion = nn.SmoothL1Loss()
        self.epsilon = args.epsilon
        self.start_epsilon = self.epsilon
        self.min_epsilon = args.min_epsilon
        self.gamma = args.gamma
        self.episode_step = 5000
        self.num_actions = num_actions
        self.buffer = replay_buffer
        self.q_network = Network(num_indim, num_actions).to(device)
        self.target_network = Network(num_indim, num_actions).to(device).eval()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.loss = 0
        self.step = 0

                
    def take_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state = torch.tensor(state).float().to(self.device)
            state = state.unsqueeze(0)
            return self.q_network(state).argmax().item()
    
    def save_model(self, model_path = args.model_path):
        torch.save({"state_dict": self.q_network.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, model_path)
    
    def load_model(self, model_path = args.model_path):
        checkpoint = torch.load(model_path)
        self.q_network.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
    
    
    
    def check_buffer(self):
        if len(self.buffer) > self.buffer.replay_buffer_size:
            return True
        return False        
        
    def train_step(self, num_iters = 1):
        if len(self.buffer)<self.batch_size:
            return 

        for i in range(num_iters):

            # 배치 하나 뽑아서
            state, action, reward, next_state, done = self.buffer.sampling()
            # print(action, reward)

            # Calculate the value of the action taken
            q = self.q_network(state).gather(1, action)

            # target q 계산
            q_next = self.target_network(next_state).detach().max(1)[0].unsqueeze(1)
            q_target = (1-done) * (reward + self.gamma * q_next) + (done * reward)

            # Compute the loss
            loss = self.criterion(q, q_target).to(self.device)

            # Perform backward propagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss = loss.item()

        
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