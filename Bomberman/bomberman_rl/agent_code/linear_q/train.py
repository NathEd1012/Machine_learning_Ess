from collections import namedtuple, deque
import numpy as np

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS

ALPHA = 0.2 # Learning rate
GAMMA = 0.25 # Discount rate
TRANSITION_HISTORY_SIZE = 10000

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def bias(array):
    # Insert bias at first position
    array = np.insert(array, 0, 1)
    return array

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events):
    # Break if last step
    if new_game_state is None:
        return

    # Features
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    old_features_bias = bias(old_features)
    new_features_bias = bias(new_features)

    if self.theta.size == 0:
        self.theta = np.random.random((len(ACTIONS), len(new_features)+1))  # transform weight array into the correct shape

    # Current Q-values
    self.q_values = np.dot(self.theta, old_features_bias) 

    # Update weights
    action_number = ACTIONS.index(self_action)
    current_weights = self.theta[action_number]
    current_q = self.q_values[action_number]
    max_next_q = np.max(np.dot(self.theta, new_features_bias))
    reward = reward_from_events(self, events)
    td_error = reward + GAMMA * max_next_q - current_q

    self.theta[action_number] = current_weights + ALPHA * td_error * old_features_bias

    # Append the transition
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events):
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Features
    features = state_to_features(last_game_state)
    features_bias = bias(features)

    # Current Q-values
    self.q_values = np.dot(self.theta, features_bias) 

    # Update weights
    action_number = ACTIONS.index(last_action)
    current_weights = self.theta[action_number]
    current_q = self.q_values[action_number]
    max_next_q = np.max(np.dot(self.theta, features_bias))
    reward = reward_from_events(self, events)
    td_error = reward + GAMMA * max_next_q - current_q

    self.theta[action_number] = current_weights + ALPHA * td_error * features_bias
    
    # Store the model
    with open("q_values.pkl", "wb") as file:
        pickle.dump(self.q_values, file)
    
    with open("weights.pkl", "wb") as file:
        pickle.dump(self.theta, file)


def reward_from_events(self, events) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 50,
        e.INVALID_ACTION: -5,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
