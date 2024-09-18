import torch
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 

from collections import deque

# Hyperparameters
DISCOUNT_FACTOR = 0.6
LEARNING_RATE = 0.001
EPSILON = 0.1
NUMBER_OF_FEATURES = 13
NUMBER_OF_ACTIONS = 6

# Experience
CAPACITY = 20000
BATCH_SIZE = 32

class DQN_Lapsus(nn.Module):
    def __init__(self, input_size = NUMBER_OF_FEATURES, number_actions= NUMBER_OF_ACTIONS, alpha=LEARNING_RATE):
        super(DQN_Lapsus, self).__init__()
        
        # first and last layer number of neuros
        self.input_size = input_size
        self.number_actions = number_actions

        # fully connected layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, number_actions)

        # Hubert Loss
        self.loss_fn = nn.SmoothL1Loss()

        # Adam optimizer for weights update (lr=learning rate)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha) 

        self.epsilon = EPSILON
        self.gamma = DISCOUNT_FACTOR

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity = CAPACITY, batch_size = BATCH_SIZE):
        
        self.buffer = deque(maxlen = capacity)
        self.batch_size = batch_size

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
