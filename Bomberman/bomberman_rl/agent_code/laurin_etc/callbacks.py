import os
import pickle
import heapq
from random import shuffle


import numpy as np
import copy


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(10)
        #self.model = np.concatenate((np.full(11,0.1),np.full(5,0.01)))
        #self.model = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.05,0.01,0.005,0.0025, 0.05525])
        self.model = np.full(19,0.1)
        #quickstart after training with no crates
        #self.model = np.array([5.09,4.4598,2.7454,2.9469,2.3599,1.7345,0.8671,1.2299,1.871,1.0779,0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1])
        #self.model = np.array([5.09,4.4598,2.7454,2.9469,2.3599,1.7345,0.8671,1.2299,1.871,1.0779,0.5,0.05,0.045,0.02,0.04,0.01,0.02, 0.95])
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


#q-function
def q_hat(self,S,A,w):
    """
    Estimated q-function that that maps q_hat: (States,Actions) -> R. It gives back the value of A in state S
    It is estimated by the weights w, that gives a special weight to each feature.
    The weights w are learned during Training process and determine the final policy.

    :param S: A dictionary with current game state
    :param A: A string with an action out of list ACTIONS
    :param w: An array with current weights
    :return: float
    """
    
    #self.logger.debug(f"CHECKING ACTION {A}")
    S_temp=copy.deepcopy(S) ###avoid bug caused by shallow-copy (.copy())
    p = S_temp['self']
    f = S_temp['field']

    ###before applying new step, test whether step is possible (field-entry equals zero)
    if A=='UP':
        if f[p[3][0],p[3][1]-1]==0:
            S_temp['self']=(p[0],p[1],p[2],(p[3][0],p[3][1]-1))
    if A=='DOWN':
        if f[p[3][0],p[3][1]+1]==0:
            S_temp['self']=(p[0],p[1],p[2],(p[3][0],p[3][1]+1))
    if A=='RIGHT':
        if f[p[3][0]+1,p[3][1]]==0:
            S_temp['self']=(p[0],p[1],p[2],(p[3][0]+1,p[3][1]))
    if A=='LEFT':
        if f[p[3][0]-1,p[3][1]]==0:
            S_temp['self']=(p[0],p[1],p[2],(p[3][0]-1,p[3][1]))
    if A=='BOMB':
        if p[2]==True:#test if bombing is possible
            S_temp['bombs'].append((p[3],4)) #4 shows that it is virtual
            S_temp['self']=(p[0],p[1],False,p[3]) ##avoid bug, that after dropping a bomb the agent is able to drop another
    
    X=state_to_features(self, S_temp)
    
    #self.logger.debug(f"X:,{A},{X}")
    # self.logger.debug(f"w:,{w}")
    #print("X:",A,X)
    #print("w:",w)
    #self.logger.debug(f"X: {X}")
    #print(len(w),len(X))
    assert len(w)==len(X)
    return w@X


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    S=game_state
    
    w=self.model
    
    ###find greedy action (maximizes q_hat for given S and w)
    tester = np.array([q_hat(self,S,a,w) for a in ACTIONS])
    
    if np.all(tester==tester[0]):###if all entries are equal the first entry is chosen by argmax
        greedy = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
    #if all step-entries have equal value and value of bomb is less, do random step
    if np.all(tester[:4]==tester[0]) and tester[0]>tester[4] and tester[0]>tester[5]: 
        greedy =np. random.choice(['RIGHT','LEFT','UP','DOWN'])
    else:
        greedy_ind = np.argmax(tester)
        greedy=ACTIONS[greedy_ind]
    


    # print("A",np.array([q_hat(self,S,a,w) for a in ACTIONS]))
    # print("w", w)
    # print(greedy)
    self.logger.debug(f"q for all actions: {tester}")
    self.logger.info(f"DOING ACTION: {greedy}")
    #Exploration vs exploitation
    # definie hyperparameter for epsilon-greedy-policy (lecture 2, p.3)
    epsilon = 0.01
    
    if self.train:
        #in training-mode use epsilon-greedy-policy
         return np.random.choice([greedy,np.random.choice(ACTIONS,1)],1,p=[1-epsilon,epsilon])

    self.logger.debug("Querying model for action")
    #in game-mode take greedy action
    return greedy

#TODO: not sure if there is a bug that explosions are reaching through walls in this function, might be good to check someday
def in_range(bomb, field = [[None]], player=None):
    """
    evaluates where a bomb is possibly threatening. 
    If a players and a bombs position are given it returns whether the player is in range of an exploding bomb.
    If player is not given, it returns an array with all threatend spaces.

    :param field: 2D-array True for free tiles False elsewhere
    :param bomb: list, array or tuple with 2D-coordinates of the bomb
    :param player (optional): list, array or tuple with 2D-coordinates of the agent

    :return: bool or list, depending on arguments handed in
    """

    ###initialize list of coordinates that are in bomb range
    bomb_range=[[bomb[0],bomb[1]],]
    if field[0][0] != None:
        
        if field[bomb[0] + 1,bomb[1]] == True:
            for i in range(1,4):
                bomb_range.append([bomb[0] + i,bomb[1]])

        if field[bomb[0],bomb[1] + 1] == True:
            for i in range(1,4):
                bomb_range.append([bomb[0],bomb[1] + i])
        
        if field[bomb[0] - 1,bomb[1]] == True:
            for i in range(1,4):
                bomb_range.append([bomb[0] - i,bomb[1]])

        if field[bomb[0],bomb[1] - 1] == True:
            for i in range(1,4):
                bomb_range.append([bomb[0],bomb[1] - i])

    else:
        for i in range(1,4):
            bomb_range.append([bomb[0]+i,bomb[1]])
            bomb_range.append([bomb[0]-i,bomb[1]])
            bomb_range.append([bomb[0],bomb[1]+i])
            bomb_range.append([bomb[0],bomb[1]-i])
    

    bombs = np.array(bomb_range)

    if player != None:
        p = np.asarray(player)
        in_range = [np.all(b==p) for b in bombs]

        return np.any(np.array(in_range)==True)


    return bombs


def is_in(point, point_list, index=False):
    """
    short auxillary function to check whether a 2-dim coordinate is in a list of 2-dim coordinates.
    In default it simply gives back a boolean value.
    With index=True, it gives back a list of indices where the point was found or False if the point is not contained.

    :param point: list, array or tuple with 2D-coordinates of one point
    :param point_list: list with 2D-coordinates
    :param index (optional): boolean if list of indices where a matching entry was found should be returned

    :return: bool or list
    """
    IN = False
    index = []
    
    #check if a matching point can be found
    for i in range(len(point_list)):
        element=point_list[i]
        if element[0]==point[0] and element[1]==point[1]:
            IN=True
            index.append(i)
    
    if index==False:
        return IN
    else:
        if IN==True:
            return index
        else:
            return False

def dijkstra(graph, weights, start, destination):
    parents = [None] * len(graph)

    q = []
    heapq.heappush(q, (0.0, start, start))

    while len(q) > 0:
        length, node, pred = heapq.heappop(q)

        if parents[node] == None:
            parents[node] = pred

            if node == destination:
                break
            for neighbor in graph[node]:
                if parents[neighbor] is None:
                    newlength = length + weights[(node, neighbor)]
                    heapq.heappush(q, (newlength, neighbor, node))
    
    if parents[destination] is None:
        return None, None

    path = [destination]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    return path, length

#TODO speed that function up, maybe there is a special algorithm that fits better for that challenge @Laurin
# def find_path(self, start, target, free_tiles):
    """
    Inserting starting and end point and the underlaying field as an array.
    Gives back the amount of steps that have to be taken, False if there is no path or True if starting point equals end point.

    :param starting_point: list, array or tuple with 2D-coordinates of the starting point
    :param end_point: list, array or tuple with 2D-coordinates of the end point
    :param field: array of the underlaying field
    :param maxiter (optional): int with maximum of iterations

    :return: the amount of steps that have to be taken as int
    """

def find_path(self, free_space, start, targets):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    if len(targets) == 1 and targets[0][0] == 0:
        return 200
    if is_in(start, targets):
        return 0
    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    # Determine the first step towards the best found target tile
    current = best
    if is_in(best, targets) == False:
        #self.logger.debug(f"No target found")
        return 201

    counter = 0
    while True:
        counter += 1
        if parent_dict[current] == start: 
            #self.logger.debug(f"Found target {best} with a distance of {counter}")
            return counter
        current = parent_dict[current]


    # min_x = max(0, starting_point[0] - 5)
    # max_x = min(16, starting_point[0] + 5)
    # min_y = max(0, starting_point[1] - 5)
    # max_y = min(16, starting_point[1] + 5)

    # if starting_point[0] != 0 or end_point[0] != 0:

    #     if starting_point[0]==end_point[0] and starting_point[1]==end_point[1]: #catch that easy case before looping
    #         self.logger.debug(f"How did I get here")
    #         return 0

    #     start = (starting_point[0] - 1) * 15 + starting_point[1] - 1
    #     end = (end_point[0] - 1) * 15 + end_point[1] - 1
    
    #     valid_fields = np.argwhere(field == 0)
    #     valid_nodes = []
    #     for tile in valid_fields:
    #         if min_x < tile[0] and max_x > tile[0] and min_y < tile[1] and max_y > tile[1]:
    #             valid_nodes.append((tile[0] - 1) * 15 - min_x + tile[1] - 1 - min_y)
        

    #     nodes = []
    #     for i in range(max_x + 1 - min_x):
    #         for j in range(max_y + 1 - min_y):
    #             nodes.append(i * (max_x - min_x) + j)

    #     graph = [None] * len(nodes)

    #     weights = {}
        
    #     self.logger.debug(f"{valid_nodes}")
    #     self.logger.debug(f"{nodes}")

    #     for node in valid_nodes:
    #         new_node = node + 1
    #         if new_node in valid_nodes:
    #             if graph[node] == None:
    #                 graph[node] = []
    #             weights[(node, new_node)] = 1
    #             graph[node].append(new_node)
            
    #         new_node = node - 1
    #         if new_node in valid_nodes:
    #             if graph[node] == None:
    #                 graph[node] = []
    #             weights[(node, new_node)] = 1
    #             graph[node].append(new_node)
            

    #         new_node = node + max_y - min_y
    #         if new_node in valid_nodes:
    #             if graph[node] == None:
    #                 graph[node] = []
    #             weights[(node, new_node)] = 1
    #             graph[node].append(new_node)
                
    #         new_node = node - max_y + min_y
    #         if new_node in valid_nodes:
    #             if graph[node] == None:
    #                 graph[node] = []
    #             weights[(node, new_node)] = 1
    #             graph[node].append(new_node)

    #     self.logger.debug(f"{graph}")
    #     self.logger.debug(f"{weights}")
    #     path, length = dijkstra(graph, weights, start, end)

    #     if length != None:
    #         self.logger.debug(f"{length}")
    #         return length
    #     else:
    #         return 200
    # else:
    #     return 201

    ##################



    # #initialize the list of coordinates that build the path, and auxilliary list to check how often path has crossed a point
    # path=[starting_point,]
    # rounds = [0,]
    
    # if starting_point[0]==end_point[0] and starting_point[1]==end_point[1]: #catch that easy case before looping
    #     return True

    # ##start loop to find a path
    # while (not is_in([end_point[0],end_point[1]],path)) and (len(path)<maxiter and len(path)>0):
        
    #     point = path[-1]
    #     #initialize a step in every direction
    #     steps = [[point[0]+1,point[1]],[point[0]-1,point[1]],[point[0],point[1]+1],[point[0],point[1]-1]]
    #     rank = rounds[-1]

    #     if rank<4: #if we didn't already tried every direction (catch loops caused by nonreachable end points)
    #         steplen = 1
    #         while len(steps)!=0: #loop until there are no more steps to try
    #             arr_steps = np.array(steps)
    #             end = np.full_like(arr_steps,end_point)
    #             diff = np.linalg.norm(arr_steps-end, axis=1)

    #             #in a previous try we tried the best step that led to a uncomplete path, now take the next best step
    #             for i in range(rank): # the rank-th best step
                    
    #                 min_ind = np.argmin(diff)
    #                 diff[min_ind] = 200 #don't delete the entry not to get problems with indices
                    
    #             min_ind = np.argmin(diff)
    #             best_step = steps[min_ind] #choose the best step out of all steps that are left
                
    #             if field[best_step[0],best_step[1]]==0: #check whether step is possible
    #                 path.append(best_step)
    #                 rounds.append(0)
    #                 steplen = len(steps)
    #                 steps=[] #can forget the remaining options
    #             else:
    #                 steps.remove(best_step) #remove that option and try with another

    #         #if we are trapped on a circle, remove the circle and give sign to take another step
    #         if is_in(path[-1],path[:-1]): #if this is true we are crossing our path somewhere ->circle

    #             if steplen == 1 or rounds[-1]==4: #if this is true we are trapped
    #                 #go back two steps
    #                 del path[-2:] 
    #                 del rounds[-2:]
    #                 rounds[-1]+=1
    #             else:
    #                 #go back one step
    #                 del path[-1]
    #                 del rounds[-1]
    #                 rounds[-1]+=1
    #     else:
    #         path=[]
        
    
    # if path==False:
    #     return False
    # else:
    #     return len(path)


def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    features = []
    channels = []

    #initialize some lists
    coins=game_state['coins']
    player=game_state['self'][3]
    bombs = [b[0] for b in game_state['bombs']]

    l=4-len(bombs)
    for i in range(l): #get full list, with (0,0) as placeholder
        bombs.append((0,0))
    
    l2=9-len(coins)
    if l2!=9:
        for i in range(l2):
            coins.append((0,0))
    
    # self.logger.debug(f"{game_state['field'] == 0}")
    ###Feature 1: Define continous potential but avoid 1/r with r->0
    coin_distance=[]
    #self.logger.debug(f"searching for coins!")
    
    if (coins == None) or (coins == []): #if there are no coins to collect, we don't want to be confused
        coin_distance=[200 for i in range(9)]
    else:
        for i in range(len(coins)):
            if coins[i][0]!=0:
                #dis.append(np.linalg.norm(np.asarray(player)-np.asarray(coins[i])))
                path_it = find_path(self, game_state["field"] == 0, tuple(player),[tuple(coins[i])])
                
                coin_distance.append(path_it)
            else:
                coin_distance.append(200)
                
        nextcoin=min(coin_distance)

        #nextcoin=min(coin_distance)
    for nextcoin in coin_distance:
        if nextcoin==0:
            inverted_coin_distance=2
        else:
            inverted_coin_distance = 1/nextcoin #<=1 and >0
        features.append(inverted_coin_distance)
    

    ###Remark: I will remove that soon
    #features.append(0)


    ###Feature 2: look for next safe_space

    #self.logger.debug(f"Looking for safe spaces!")
    field = np.copy(game_state['field'])

    closest_spot = None

    #max 4*13=52 iterations, probably won't take to long
    
    if game_state['bombs']!=[]: #only if bombs are active on the field

        #determine all spaces that are currently threatend by a bomb and insert entry 2 in our copy
        for bomb in game_state['bombs']:
            if bomb != (player,4): #don't mind virtual bombs
                if bomb[1]<=4:
                    bomb_range = in_range(bomb[0], game_state['field'] == 0)
                    for s in bomb_range:
                        if np.all(s<=16) and np.all(s>=0):
                            if field[s[0],s[1]]!=-1:
                                field[s[0],s[1]]=2

        #determine all spaces that are threatend by a remaining explosion and insert entry 3 in our copy
        explosion_map = game_state['explosion_map']
        explosions = np.argwhere(explosion_map!=0)
        for e in explosions:
            
            if field[e[0],e[1]]!=-1:
                field[e[0],e[1]]=3

        
        #now all spaces with entry 0 are safe spaces for the current state
        free_s = np.argwhere(field==0)
        pos = np.full_like(free_s,np.asarray(player))
        free_s_distances = np.linalg.norm(pos-free_s,axis=1)
        test_s = []
        ###To speed training up I worked with euclidian distances, as soon as Laurin has optimized find_path we can rechange that
        
        
        #just consider near spaces (otherwise it would take to long)
        for i in range(len(free_s_distances)):

            if free_s_distances[i]<=6:
                test_s.append(tuple(free_s[i]))

        # safe_space_distance=[]
        # maximal_iteration = 1000
        valid_fields = copy.deepcopy(game_state["field"])
        for other in game_state["others"]:
            valid_fields[other[3][0], other[3][1]] = 1
        
        closest_spot = find_path(self, valid_fields == 0, tuple(player), test_s)          

            
            # if path_iter < maximal_iteration:

            #     safe_space_distance.append(path_iter) 
            #     maximal_iteration=path_iter #we are only interessted in better choice, therefore we can speed calculation up

        
        # closest_spot = min(safe_space_distance)

        # if len(safe_space_distance)!=0:
        #     closest_spot = min(safe_space_distance)
        # else:

        #     closest_spot = np.min(free_s_distances)
    
        #closest_spot = np.min(free_s_distances) #part of upspeeding
    
    if closest_spot==0:
        inverted_closest_spot=2
    elif closest_spot==None:
        inverted_closest_spot=2
    else:
        inverted_closest_spot=1/closest_spot
    
    features.append(inverted_closest_spot)

    
    ###Feature 3: Count destroyable crates by a dropped bomb
    
    if (player,4) in game_state['bombs']: #only active if a bomb is dropped by our agent

        reachable_crates=0
        reachable_spaces = in_range(player)
        
        for r in reachable_spaces:
            if np.all(r<=16) and np.all(r>=0):
                if game_state['field'][r[0],r[1]]==1:
                    reachable_crates+=1
        
        if reachable_crates==0:
            crates=0
        else:
            crates=reachable_crates/13
    else:
        crates=0

    # features.append(crates)



    ###Feature 3: Check how many crates are next to the agent
    #self.logger.debug(f"checking for crates!")
    if (player,4) in game_state['bombs'] or (player,3) in game_state['bombs']: #only active if a bomb is dropped by our agent (plus bugfix for training)

        spaces_around_agent = [[player[0]+1,player[1]],[player[0]-1,player[1]],[player[0],player[1]+1],[player[0],player[1]-1]]
        field_around_agent = np.array([game_state['field'][space[0]][space[1]] for space in spaces_around_agent])
        crates_around_agent = np.count_nonzero(field_around_agent==1)
        if crates_around_agent > 0:
            features.append(crates_around_agent/4)
        else:
            features.append(0)
    else:
        features.append(0)


    ###Feature 4: Find next crate

    #self.logger.debug(f"Finding next crate!")
    field=np.copy(game_state['field'])
    crate_indices = np.argwhere(field==1) #find all crates on the field
    player = game_state['self'][3]

    minimal_index=False
    
    p = np.full_like(crate_indices,player)
    diff=np.linalg.norm(p-crate_indices,axis=1)

    for i in range(3):
        
        #and game_state['self'][2]==True
        if len(crate_indices)!=0: #if there are no crates we can't find any distances
 
            minimal_index = np.argmin(diff)
            
            minimal_crate = crate_indices[minimal_index]
            #self.logger.debug(f"{crate_indices}")
            #self.logger.debug(f"{minimal_crate}")

            #we need to virtually remove the crate to make the find_path function work
            field[minimal_crate[0],minimal_crate[1]] = 0
            minimal_crate_distance = find_path(self, field == 0, tuple(player), [tuple(minimal_crate)]) #safe computation time by only calulating the euclidian closest 
            field[minimal_crate[0],minimal_crate[1]] = 1
            
            #minimal_crate_distance=np.min(diff)
            assert minimal_crate_distance != 0 #by definition one can't stand on a crate spot. Thus diff can't be 0.

            inverted_minimal_crate_distance = 1/minimal_crate_distance

            if  minimal_crate_distance == 200: #the euclidian next crate is not reachable, therefore we ignore him this round
                inverted_minimal_crate_distance = 0

            elif field[player[0],player[1]]==0: #don't walk in danger
                #to get min_dis we virtually removed the crate, but now the crate is back. Therefore it is reached one step earlier.

                inverted_minimal_crate_distance = 1/minimal_crate_distance

            features.append(inverted_minimal_crate_distance)
            #features.append(1/np.min(diff))
                
            diff = np.delete(diff, minimal_index, 0)
            crate_indices = np.delete(crate_indices, minimal_index, 0)
            
        else:
            features.append(0)  
            #features.append(0)                                
    
    ###Feature 5: Count threatened opponents by a dropped bomb
    opponents = game_state['others']
    
    if (player,4) in game_state['bombs']: #only active if a bomb is dropped by our agent

        reachable_opponents=0

        for enemy in opponents:
            if is_in(enemy[3], reachable_spaces):
                reachable_opponents+=1
        
        if reachable_opponents==0:
            inverted_reachable_opponents=0
        else:
            inverted_reachable_opponents=reachable_opponents/3
    else:
        inverted_reachable_opponents=0

    features.append(inverted_reachable_opponents)

    ###Feature 6: Distance to opponents

    enemys = np.zeros((4,2))
    for i in range(len(opponents)):
        enemys[i] = opponents[i][3]

    for opponent in enemys:
        if opponent[0]!=0:
            distance_to_opponent = np.linalg.norm(player-opponent)
        else:
            distance_to_opponent = 200

        if distance_to_opponent == 0:
            inverted_distance_to_opponent = 2
        else:
            inverted_distance_to_opponent = 1/distance_to_opponent
        features.append(inverted_distance_to_opponent)

     #I kept this distinction between features and channels if a feature augmentation would be neccessary at some point (Sutton, 9.5)
    for feature in features:
        channels.append(feature)
    
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    
    return stacked_channels.reshape(-1)
    