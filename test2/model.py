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
    
    

        