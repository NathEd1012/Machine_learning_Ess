import csv
import os
from collections import namedtuple, deque, defaultdict
import pickle
from typing import List
import time
import events as e
from .features import state_to_features, get_bomb_features, get_danger_map
import numpy as np

import sys
import argparse

import random

import torch

from .callbacks import ACTIONS
from .training_logger import TrainingLogger
from .model import ReplayBuffer

# Data log
LOG_FREQUENCY = 10

# Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# get n-rounds from command line without modifying main.py
def get_n_rounds_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rounds", type=int, default=1, help="Number of rounds to play")
    args, _ = parser.parse_known_args(sys.argv)
    return args.n_rounds

def setup_training(self):
    """
    Initialize self for training purposes.

    This is called after `setup` in callbacks.py.
    """
    self.survived_steps = 0
    # Initialize replay buffer
    self.replay_buffer = ReplayBuffer()
    self.batch_size = self.replay_buffer.batch_size

    # Initialize best reward
    self.best_reward = float('-inf')

    ## Logs
    self.stat_logger = TrainingLogger(
        'training_stats.csv',
        [event for event in dir(e) if not event.startswith('__')],
        ACTIONS,
        log_frequency = LOG_FREQUENCY
        )

    # Log counter
    self.log_counter = 0

    # Parse n-rounds from command-line arguments
    self.total_rounds = get_n_rounds_from_args()


def train_dqn(self):
    """
    Train dqn using experience buffer and Bellman equation
    """
    # Only train if enough experiences in the buffer
    if self.replay_buffer.size() < self.batch_size:
        return

    # Sample a batch of transitions from replay buffer
    transitions = self.replay_buffer.sample(self.batch_size)
    batch = Transition(*zip(*transitions))

    # Convert states, actions, rewards, and next_states into tensors
    state_batch = torch.tensor(np.array(batch.state), dtype = torch.float32)
    action_batch = torch.tensor([ACTIONS.index(a) for a in batch.action], dtype = torch.long)
    reward_batch = torch.tensor(batch.reward, dtype = torch.float32)
    next_state_batch = torch.tensor(np.array([
        np.zeros_like(state_batch[0]) if next_state is None else next_state
        for next_state in batch.next_state
    ]), dtype=torch.float32)

    # Compute q values
    q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

    # Compute next state's max q-value
    with torch.no_grad():
        next_q_values = self.model(next_state_batch).max(1)[0]

    # Compute target q-value
    q_targets = reward_batch + self.model.gamma * next_q_values

    # Compute loss
    loss = self.model.loss_fn(q_values, q_targets)

    # Perform gradient descent
    self.model.optimizer.zero_grad()
    loss.backward()

    # Gradient clipping (vll nicht notwendig)
    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

    self.model.optimizer.step()

    # Log the losss
    #self.logger.debug(f"Loss: {loss.item()}")
    self.stat_logger.update_loss(loss.item())


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    self.survived_steps = self.survived_steps + 1

    for _ in range(self.survived_steps):
        events.append("STEP_SURVIVED")

    # 
    old_state_features = state_to_features(old_game_state)
    new_state_features = state_to_features(new_game_state)

    # Penalize if standing on bomb place
    own_position = new_game_state['self'][3]
    bomb_danger_here = get_bomb_features(own_position, new_game_state)[4]


    if bomb_danger_here == - 1 / 3: # bomb explosion in 3 ticks
        events.append("WARM")
    elif bomb_danger_here == - 1 / 2: # bomb explosion in 2 ticks
        events.append("HOT")
    elif bomb_danger_here == - 1: # bomb explosion in 1 ticks
        events.append("BOILING")

    # Reward moving out of danger
    previous_pos = old_game_state['self'][3]
    previous_danger_map = get_danger_map(old_game_state)
    previous_danger = previous_danger_map[previous_pos[0], previous_pos[1]]

    current_pos = new_game_state['self'][3]
    current_danger_map = get_danger_map(new_game_state)
    current_danger = current_danger_map[current_pos[0], current_pos[1]]

    if current_danger == 0 and  previous_danger < 0: # Think: values are negative, so pd < 0 bad.
        events.append("FRESHENED_UP")

    # Calculate rewards and accumulate them
    reward = reward_from_events(self, events)
    self.stat_logger.total_reward += reward

    # Store transition in replay buffer
    transition = Transition(old_state_features, self_action, new_state_features, reward)
    self.replay_buffer.push(transition)

    # Train if enough experiences
    train_dqn(self)

    # Log
    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    self.stat_logger.update_event_count(events)
    self.stat_logger.update_action_count(self_action)
    self.stat_logger.update_score(new_game_state["self"][1])
    self.stat_logger.update_reward(reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is also a good place to store an agent that you updated.
    """
    # Increment round counter
    self.stat_logger.round_counter += 1

    # Save model after every round
    torch.save(self.model.state_dict(), "my-saved-model.pt")
    #self.logger.info("Model saved to my-saved-model.pt")

    # Save best performing model:
    current_reward = self.stat_logger.total_reward
    if current_reward > self.best_reward:
        #self.logger.info(f"New best reward: {current_reward}. Saving best-model.pt")
        self.best_reward = current_reward
        torch.save(self.model.state_dict(), "best-model.pt")

    # Set total_rounds if not already set (e.g., passed in from main.py)
    if self.total_rounds is None:
        self.total_rounds = self.round_counter  # Default to current round count if not set elsewhere

    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Push final transition
    reward = reward_from_events(self, events) # Repeat last_game_state
    last_state_features = state_to_features(last_game_state)
    final_transition = (last_state_features, last_action, None, reward)
    self.replay_buffer.push(final_transition)

    train_dqn(self)

    self.survived_steps = 0

    # Log data
    self.stat_logger.total_reward += reward

    self.stat_logger.update_event_count(events)
    self.stat_logger.update_action_count(last_action)
    self.stat_logger.update_score(last_game_state["self"][1])
    self.stat_logger.update_reward(reward)


    if self.stat_logger.round_counter % self.stat_logger.log_frequency == 0:
        self.stat_logger.log_statistics()


def reward_from_events(self, events: List[str]):
    """
    Modify the rewards your agent gets to encourage certain behavior.
    """
    game_rewards = {
        e.INVALID_ACTION: -5,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.KILLED_SELF: -50,
        e.BOMB_DROPPED: -1,
        e.SURVIVED_ROUND: 100,
        e.COIN_COLLECTED: 25,
        e.CRATE_DESTROYED: 50,
        # Custom events
        e.WARM: - 10,
        e.HOT: -20,
        e.BOILING: -40,
        #e.STEP_SURVIVED: 0.1
        #e.FRESHENED_UP: 2
    }
    reward_sum = 0

    # Event based rewards
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

    return reward_sum