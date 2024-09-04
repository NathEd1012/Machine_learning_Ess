import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
EPSILON = 0.1


def setup(self):
    # initialize training
    if self.train and not os.path.isfile("q_values.pkl"):
        # Initialize weights
        self.theta = np.array([])
        # Q-values
        self.q_values = {}
    else:
        with open("q_values.pkl", "rb") as file:
            self.q_values = pickle.load(file)
        with open("weights.pkl", "rb") as file:
            self.theta = pickle.load(file)


def act(self, game_state: dict) -> str:
    if self.train and np.random.random() < EPSILON:
        return np.random.choice(ACTIONS)
    else:
        return ACTIONS[np.argmax(self.q_values)]

def state_to_features(game_state: dict) -> tuple:
    if game_state is None:
        return None
    else:
        game_map = game_state['field']

        def nearby_fields(position):
            up = game_map[position[0]+1,position[1]]
            down = game_map[position[0]-1,position[1]]
            left = game_map[position[0],position[1]-1]
            right = game_map[position[0],position[1]+1]

            return(up,down,left,right)
        
        score = game_state['self'][1]
        coins = game_state['coins']
        own_position = game_state['self'][3]
        agent_nearby_fields = nearby_fields(own_position)
        nearest_coin = nearest_coin_calc(own_position, coins)


        return(own_position[0],own_position[1],agent_nearby_fields[0],agent_nearby_fields[1],agent_nearby_fields[2],agent_nearby_fields[3],score, nearest_coin[0], nearest_coin[1], nearest_coin[2])
    
def nearest_coin_calc(position, coins):
    x_0, y_0 = position
    # Nearest coin coordinates and distance (Manhattan distance)
    nearest = min(coins, key=lambda coin: abs(coin[0] - x_0) + abs(coin[1] - y_0))
    distance = abs(nearest[0] - x_0) + abs(nearest[1] - y_0)
    return [nearest[0], nearest[1], distance]
