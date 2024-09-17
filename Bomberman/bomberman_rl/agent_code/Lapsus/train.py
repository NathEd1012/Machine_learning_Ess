import os
import random
import copy
import torch
import numpy as np

import events as e

from typing import List
from collections import deque
from .callbacks import state_to_features, get_bomb_features, get_danger_map
from .model import Lapsus


ACTIONS = {'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'WAIT':4, 'BOMB':5}

def add_exp(self, old_game_state, self_action, new_game_state, events, done):
    """
    adds experience to the experience buffer (list of tuples) 
    done is a boolean, only true if episode has ended (max. amount of steps or agent died)
    each tuple = (old feature vector, performed action, total reward for detected events, new feature vector)
    """
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    action = ACTIONS[self_action]
    reward = reward_from_events(events)

    self.exp_buffer.append((old_features, action, reward, new_features, done))

def train_net(self):
    """
    if experience buffer long enough train policy network
    """
    if len(self.exp_buffer) < self.batch_size:
        return
    
    batch = random.sample(self.exp_buffer, self.batch_size)

    states, actions, rewards, next_states, dones = zip(*batch)

    # convert quantities to torch tensors 
    states = torch.tensor(np.array(states), dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float)
    dones = torch.tensor(dones, dtype=torch.float)

    # compute current Q-values using the policy network
    q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # compute the maximal next Q-values using the target network
    next_q_values = self.target_net(next_states).max(1)[0]

    # bellman equation, equal to reward if done was True
    self.gamma = self.policy_net.gamma
    target_q_values = rewards + (self.gamma * next_q_values * (1 - dones)) 

    # compute loss using the loss function (MSE)
    # Q-values are tensors here loss is computed as an average over all matrix entries
    loss = self.policy_net.loss_function(q_values, target_q_values.detach())

    # backpropagation on policy network
    self.policy_net.optimizer.zero_grad()
    loss.backward()
    self.policy_net.optimizer.step()

def update_epsilon(self, current_eps, final_eps, reward_history):
    avg = sum(reward_history) / len(reward_history)
    target_reward = 300
    decay_rate = 1 - (avg / target_reward)
    self.epsilon = max(final_eps, current_eps * decay_rate)


def setup_training(self):
    """
    called once after setup in callbacks.py
    """
    # model file path
    model_path = 'current_model.pth'
    
    # check if model for training exists, if not create a new one
    if not os.path.isfile(model_path):
        self.logger.info('Creating new model for training.')
        torch.save(self.policy_net.state_dict(), model_path)
    else:
        self.logger.info('Loading existing model for training.')
        self.policy_net.load_state_dict(torch.load(model_path))
    
    # set policy network to training mode and initialize training
    self.policy_net.train()
    self.policy_net.start_training()

    # initialize experience buffer (list of tuples)
    self.exp_buffer = deque(maxlen=self.policy_net.buffer_size)

    self.batch_size = self.policy_net.batch_size

    # make a copy of policy network (target network)
    self.target_net = copy.deepcopy(self.policy_net)

    # initalize episode counter (+1 after each game)
    self.episode_counter = 0

    self.reward_history = []

    self.epsilon = self.policy_net.epsilon

    if self.epsilon == 0:
        self.epsilon = self.policy_net.initial_epsilon

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    called once per step each game
    """
    if old_game_state is None or new_game_state is None:
        return
    
    # collect experience
    add_exp(self, old_game_state, self_action, new_game_state, events, done=False)

    # update policy network (only if enough experience)
    train_net(self)
    
    self.reward_history.append(reward_from_events(events))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    called once at the end of each game
    """
    self.episode_counter = self.episode_counter + 1
    self.reward_history.append(reward_from_events(events))

    # if game counter equals training episode number save policy network and synchronize both networks
    if self.episode_counter % self.policy_net.synchronization_rate == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        #self.logger.info('Networks synchronized.')
    
    if self.episode_counter % self.policy_net.saving_rate == 0:
        torch.save(self.policy_net.state_dict(), 'current_model.pth')
        self.logger.info('Model saved.')
        self.logger.info(f'Total rewards this game:{sum(self.reward_history)}')

    add_exp(self, last_game_state, last_action, 0, events, done=True)
    train_net(self)

    
    if sum(self.reward_history) > 0:
        update_epsilon(self, self.epsilon, self.policy_net.final_epsilon, self.reward_history)
        self.logger.info(f'Current epsilon:{self.epsilon}')

    self.reward_history = []

def reward_from_events(events: List[str]):
    """
    calculate total reward from events
    """
    game_rewards = {
        e.COIN_COLLECTED: 50,
        e.KILLED_OPPONENT: 125,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -20,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: -10,
        e.GOT_KILLED: -70,
    }
    reward = sum(game_rewards.get(event, 0) for event in events)
    return reward
