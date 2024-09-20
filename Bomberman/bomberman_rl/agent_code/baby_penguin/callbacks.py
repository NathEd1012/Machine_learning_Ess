import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT'] #, 'WAIT', 'BOMB']
EPSILON = 0.10

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game features.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    # Initialize with tracker info
    self.round_counter = 0
    self.n_rounds = getattr(self, 'n_rounds', None)

    q_table_path = os.path.join(os.path.dirname(__file__), "q_table.pkl")  # Adjust this based on your file structure

    if self.train and not os.path.isfile(q_table_path):
        self.logger.info("Setting up Q-table from scratch")
        self.q_table = {}
    else:
        self.logger.info("Loading Q-table from saved features.")
        try:
            with open(q_table_path, "rb") as file:
                self.q_table = pickle.load(file)
        except EOFError:
            self.logger.error("Error: Q-table file is empty or corrupted")
            self.q_table = {}
    print("Q_Table_Length", len(self.q_table))  # Prints the number of entries in the Q-table
'''     
    # Q-table
    if self.train and not os.path.isfile("q_table.pkl"):
        self.logger.info("Setting up Q-table from scratch")
        self.q_table = {}
    else:
        self.logger.info("Loading Q-table from saved features.")
        with open("q_table.pkl", "rb") as file:
            self.q_table = pickle.load(file)
'''
    

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)

    # Explore
    if self.train and np.random.rand() < EPSILON:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    # Exploit
    self.logger.debug("Choosing action based on Q-table")
    if features not in self.q_table:
        self.q_table[features] = np.zeros(len(ACTIONS))

    return ACTIONS[np.argmax(self.q_table[features])]

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
        if len(coins) == 0:
            nearest_coin = 0,0
        else:
            nearest_coin = nearest_coin_calc(own_position, coins)
        return(agent_nearby_fields[0],agent_nearby_fields[1],agent_nearby_fields[2],agent_nearby_fields[3],score, nearest_coin[0], nearest_coin[1])
    
def nearest_coin_calc(position, coins):
    x_0, y_0 = position
    # Nearest coin coordinates and distance (Manhattan distance)
    nearest = min(coins, key=lambda coin: abs(coin[0] - x_0) + abs(coin[1] - y_0))
    dx = nearest[0] - x_0
    dy = nearest[1] - y_0
    closest_coin = dx, dy
    return closest_coin

def nearest_coin_calc2(position, coins):
    distance = 35
    sight = 0
    while distance == 35:
        sight += 2
        for coin in coins:
            dx = abs(coin[0] - position[0])
            dy = abs(coin[1] - position[1])
            if dx <= sight and dy <= sight:
                if distance >= dx + dy:
                    distance = dx + dy
                    closest_coin = (coin[0] - position[0]), (coin[1] - position[1])
    #print(closest_coin)
    return closest_coin
    