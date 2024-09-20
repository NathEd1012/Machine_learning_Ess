import os
import pickle
import random
from collections import deque
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']#, 'WAIT', 'BOMB']#'BOMB']
EPSILON = 0.10
explosions1 = np.array(None)
explosions2 = np.array(None)

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
    
    # Initialize Q-table (or load if available)
    if os.path.exists('q_table.pkl'):
        with open('q_table.pkl', 'rb') as file:
            self.q_table = pickle.load(file)
        self.logger.info(f"Q-table loaded from file, current table size: {len(self.q_table)}")
    else:
        self.q_table = {}
        self.logger.info(f"No Q-table found, initializing a new one")

    self.action_count = {action: 0 for action in ACTIONS}

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
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
    #print(features)
    # Explore
    if self.train and np.random.rand() < EPSILON:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    # Exploit
    self.logger.debug("Choosing action based on Q-table")
    if features not in self.q_table:
        self.q_table[features] = np.zeros(len(ACTIONS))
    
    act = ACTIONS[np.argmax(self.q_table[features])]
    #print(act)
    return act 

def state_to_features(game_state: dict) -> tuple:
    if game_state is None:
        return None
    
    own_position = game_state['self'][3] 

    '''
    danger_map = bomb_danger(own_position, game_state)
    #print(danger_map)
    # Neighboring tiles
    neighboring_tiles_features = get_neighboring_tiles(own_position, game_state, danger_map)

    ########### Avoiding Bombs #############
    #print("ko",game_state['explosion_map'][own_position],game_state['explosion_map'])
    if danger_map[own_position] != 0 or game_state['explosion_map'][own_position] > 0:
        hot_space_danger = ["DANGER"]
    else: 
        hot_space_danger = ["Safe"]
    # Neighboring bombs
    #bomb_features = get_bomb_features(own_position, game_state, danger_map)    
    

    ############ Farming ################
    
    #next step to target
    next_coin = get_path_bfs(game_state, target_types = ['coin'])
    if next_coin == [-1]:
        next_move_target_features = get_path_bfs(game_state, target_types = ['crate'])
        #crate_or_coin = 1
    else:
        next_move_target_features = next_coin
        #crate_or_coin = 2
    #crate_or_coin = np.atleast_1d(crate_or_coin)

    # How many crates destroyed by bomb
    how_many_crates_boom = calculate_crates_destroyed(game_state)
    #print(next_move_target_features)
    farming_features = np.concatenate([
        next_move_target_features, # 1: in which direction does the bfs say we should go for coin/crate     
        how_many_crates_boom, # 1: how many crates get destroyed by placing a bomb here?
        #crate_or_coin
    ])

    ############## Fighting ##############
    
    fighting_features = (0, 0, 0)

    features = np.concatenate([
        neighboring_tiles_features, # 4: up, right, down, left
        hot_space_danger,
        #bomb_features, # 5: up, right, down, left, here
        farming_features,
        fighting_features
    ])
    '''

    next_move_target_features = get_path_bfs(game_state, target_types = ['coin']) # 0up, 1right, 2down, 3left
    if next_move_target_features == [-1]:
        #print("crate")
        next_move_target_features = get_path_bfs(game_state, target_types = ['crate'])
    how_many_crates_boom = calculate_crates_destroyed(game_state)


    if len(game_state['bombs']) > 0:
        danger_map = bomb_danger(own_position, game_state)
    else:
        danger_map = np.zeros_like(game_state['field'], dtype = float)

    if 2 >= danger_map[own_position] > 0:
        hot_space_danger = ["DANGER"]
    elif danger_map[own_position] >= 3:
        hot_space_danger = ["HOT"]
    else: 
        hot_space_danger = ["Safe"]

    if next_move_target_features[0] == 0:
        step = [0,1]
    elif next_move_target_features[0] == 1:
         step = [1,0]
    elif next_move_target_features[0] == 2:
         step = [0,-1]
    elif next_move_target_features[0] == 3:
         step = [-1,0]
    else:
        step = [0,0]

    sugg_pos = (own_position[0] + step[0], own_position[1] + step[1])
    #print(danger_map[sugg_pos])
    #print(danger_map)
    #print(game_state['field'][sugg_pos])
    
    if 2 >= danger_map[sugg_pos] > 0:
        hot_run_danger = ["DANGER"]
    elif danger_map[sugg_pos] >= 3 and not game_state['field'][sugg_pos] == 1:
        hot_run_danger = ["HOT"]
    elif game_state['field'][sugg_pos] == 1:
        hot_run_danger = ["crate"]
    else: 
        hot_run_danger = ["Safe"]


    if hot_run_danger == ["DANGER"] or hot_run_danger == ["HOT"]:
        next_move_safe_tile = get_path_bfs_safe_tile(game_state, danger_map)
    else:
        next_move_safe_tile = [-2] #get_path_bfs_safe_tile(game_state, danger_map)

    
    
    features = np.concatenate([next_move_target_features,
                            #next_move_safe_tile,
                            #how_many_crates_boom,
                            #hot_space_danger,
                            #hot_run_danger
                            ]) 
    

    
    #print("... ^; >; u; <; ..D; next; crates; fight x3")
    #print(feature)

    return tuple(features)


def bomb_danger(own_position, game_state):
    field = game_state['field']
    rows, cols = field.shape
    x, y = own_position
    global explosions1
    global explosions2

    danger_map = np.zeros_like(field, dtype = float)


    all_bombs = game_state['bombs']
    
    if explosions1:
        all_bombs.append(explosions1)
    if explosions2:
        all_bombs.append(explosions2)
    
    explosions2 = explosions1
    explosions1 = np.array(None)


    for bomb_pos, bomb_timer in all_bombs:
        if bomb_timer == 1:
            if explosions1:
                explosions1.append((bomb_pos, 0))
        

    for bomb_pos, bomb_timer in all_bombs:
        bx, by = bomb_pos
        danger_score = bomb_timer + 1

        # Mark danger map
        danger_map[bx, by] = 5 #bomb_timer
        # Mark explosion range (stop if wall)
        # Up
        for i in range(1, 4):
            if by - i >= 0 and field[bx, by - i] == - 1: # Wall, interrupt danger 
                break
            elif by - i >= 0: # No wall, mark danger
                danger_map[bx, by - i] = danger_score

        for i in range(1, 4):
            if bx - i >= 0 and field[bx, bx - i] == - 1: # Wall, interrupt danger 
                break
            elif bx - i >= 0: # No wall, mark danger
                danger_map[bx - i, by] = danger_score

        for i in range(1, 4):
            if by + i >= 0 and field[bx, by + i] == - 1: # Wall, interrupt danger 
                break
            elif by + i >= 0: # No wall, mark danger
                danger_map[bx, by + i] = danger_score

        for i in range(1, 4):
            if bx + i >= 0 and field[bx, bx + i] == - 1: # Wall, interrupt danger 
                break
            elif bx + i >= 0: # No wall, mark danger
                danger_map[bx + i, by] = danger_score
        #print("Joooo",danger_map)
    return danger_map

def get_path_bfs(game_state, target_types =['coin', 'crate']):
    """
    Using breadth-first-search, we want to determine the shortest path to our target.
    Since there are walls and crates, this could make it complicated as a feature. 
    For that reason, only return the next step: up, right, down, left
    """
    # dx, dy
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    direction_names = [0, 1, 2, 3] # up, right, down, left
    # Own position and field
    field = game_state['field']
    start_x, start_y = game_state['self'][3]

    rows, cols = field.shape
    visited = set() # Keep track of tiles already visited

    # BFS queue: stores (x, y, first_move) where first_move is initial direction
    queue = deque([(start_x, start_y, None)])  
    visited.add((start_x, start_y))  

    # Get target positions (coins, crates, enemies)
    targets = []
    if 'coin' in target_types:
        targets = game_state['coins']
    elif 'crate' in target_types:
        targets.extend((x, y) for x in range(rows) for y in range(cols) if field[x, y] == 1)
        #print("crates", targets)
        targets = targets[:]
    # BFS to find shortest path
    distance = -1
    while queue:
        x, y, first_move = queue.popleft()
        distance += 1
        # Check if reached target
        if (x, y) in targets:
            
            if first_move is not None:
                if distance == 1 and 'crate' in target_types:
                    #print(distance)
                    return [str(direction_names[first_move]) + target_types[0]]
                else: 
                    return [direction_names[first_move]]

        # Explore neighboring tiles
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy
            # Check if new position within bounds and not visited
            if 0 <= new_x < rows and 0 <= new_y < cols and (new_x, new_y) not in visited:
                if field[new_x, new_y] >= 0: # Free tile
                    visited.add((new_x, new_y))
                    # Enque new position, passing first move
                    if first_move is None:
                        queue.append((new_x, new_y, direction_names[i]))
                    else:
                        queue.append((new_x, new_y, first_move))

    # Return if no path to target
    return [-1] # No valid move

def calculate_crates_destroyed(game_state):
    """
    How many crates can we destroy by placing a bomb in the current position? 
    Only bombs dropped by tha agent
    """
    field = game_state['field']
    agent_x, agent_y = game_state['self'][3]

    rows, cols = field.shape

    # Bomb exlposion radius:
    explosion_radius = 3

    # Directions: up, right, down, left
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    # Initialize crates destroyed:
    crates_destroyed = 0

    # Check all four directions from the agent's position:
    for dx, dy in directions: 
        for step in range(1, explosion_radius + 1):
            new_x = agent_x + dx * step
            new_y = agent_y + dy * step

            # Check if within bounds:
            if new_x < 0 or new_y < 0 or new_x >= rows or new_y >= cols:
                break
            # Check what tile
            tile = field[new_x, new_y]
            # Break if wall:
            if tile == -1:
                break
            elif tile == 1:
                crates_destroyed +=1
                break

    return [crates_destroyed]


def get_path_bfs_safe_tile(game_state, danger_map):
    """
    Using breadth-first-search, determine the shortest path to a safe tile.
    Safe tiles are those that are free (no walls or crates) and not within bomb blast radii.
    Only return the next step: up, right, down, left.
    """
    # Directions: (dx, dy) for UP, RIGHT, DOWN, LEFT
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    direction_names = [0, 1, 2, 3]  # up, right, down, left

    # Own position and field
    field = game_state['field']
    start_x, start_y = game_state['self'][3]  # agent's current positions

    rows, cols = field.shape
    visited = set()  # Keep track of tiles already visited

    # BFS queue: stores (x, y, first_move) where first_move is the initial direction
    queue = deque([(start_x, start_y, None)])  
    visited.add((start_x, start_y))  

    # BFS to find shortest path to a safe tile (free and outside danger)
    while queue:
        x, y, first_move = queue.popleft()

        # Check if the current tile is both free and safe
        if field[x, y] == 0 and danger_map[x, y] == 0:
            if first_move is not None:
                return [first_move]

        # Explore neighboring tiles
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy

            # Check if new position is within bounds and not visited
            if 0 <= new_x < rows and 0 <= new_y < cols and (new_x, new_y) not in visited:
                if field[new_x, new_y] == 0:  # Free tile (no wall or crate)
                    visited.add((new_x, new_y))
                    # Enqueue the new position, passing the first move
                    if first_move is None:
                        queue.append((new_x, new_y, direction_names[i]))
                    else:
                        queue.append((new_x, new_y, first_move))

    # Return if no safe path is found
    return [-1]  # No valid move found

#########
######






def nearest_coin_calc(position, coins):
    x_0, y_0 = position
    # Nearest coin coordinates and distance (Manhattan distance)
    nearest = min(coins, key=lambda coin: abs(coin[0] - x_0) + abs(coin[1] - y_0))
    dx = nearest[0] - x_0
    dy = nearest[1] - y_0
    closest_coin = dx, dy
    return closest_coin

def get_neighboring_tiles(own_position, game_state, danger_map):

    field = game_state['field']
    x, y = own_position
    rows, cols = field.shape
    #print(danger_map.shape)
    if danger_map[x][y - 1] != 0:
        tile_up = -1 #danger_map[x][y - 1] 
    elif field[x][y - 1] == 0:
        tile_up = 1 
    else:
        tile_up = 0

    if danger_map[x][y + 1] != 0:
        tile_down = -1 #danger_map[x][y + 1]    
    elif field[x][y + 1] == 0:
        tile_down = 1 
    else:
        tile_down = 0   

    if danger_map[x + 1][y] != 0:
        tile_right = -1 #danger_map[x + 1][y] 
    elif field[x + 1][y] == 0:
        tile_right = 1 
    else: 
        tile_right = 0

    if danger_map[x - 1][y] != 0: 
        tile_left = -1 #danger_map[x - 1][y]    
    elif field[x - 1][y] == 0:
        tile_left = 1 
    else: 
        tile_left = 0

    neighboring_tiles = [tile_up, tile_right, tile_down, tile_left]

    return neighboring_tiles