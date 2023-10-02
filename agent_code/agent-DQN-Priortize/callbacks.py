from collections import deque
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT','BOMB']

class DQN(nn.Module):
        def __init__(self, input_size, output_size):
         super(DQN, self).__init__()
         self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
            )

        def forward(self, x):
         return self.fc(x)



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
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    input_size = 20
    output_size = 6
   # Initialize networks and optimizer
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = DQN(input_size, output_size)
    
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)




def act(self,game_state):

    #Exploration vs exploitation
    random_prob = .1

    if self.train:
        if(random.random() < random_prob or game_state['round'] < 100):
            self.logger.debug("Choosing action purely at random.")
            #80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        else:
            features= state_to_features(self,game_state)
            features_tensor = torch.tensor(features, dtype=torch.float32)

            with torch.no_grad():
                q_values = self.model(features_tensor)
    # Choose the action with the highest Q-value 
            action_idx = q_values.argmax()
            self.logger.debug("Querying model for action.")

        return ACTIONS[action_idx]
    else:
        features= state_to_features(self,game_state)
        features_tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            q_values = self.model(features_tensor)
    # Choose the action with the highest Q-value 
        action_idx = q_values.argmax()
        self.logger.debug("Querying model for action.")
        print(q_values)

    return ACTIONS[action_idx]

def state_to_features(self,game_state: dict) -> np.array:

    """
    Converts the game state to a feature vector.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None
    # Extract relevant information from the game_state dictionary
    round_number = game_state['round']
    step_number = game_state['step']
    agent_score = game_state['self'][1]
    _, score, bombs_left, (x, y) = game_state['self']


    field = game_state['field']
    
    # features about walls 
    is_wall_left = 1 if (x > 0 and field[x - 1, y] == -1) else 0
    is_wall_right = 1 if (x < field.shape[0] - 1 and field[x + 1,y] == -1) else 0
    is_wall_up = 1 if (y > 0 and field[x, y - 1] == -1) else 0
    is_wall_down = 1 if (y < field.shape[1] - 1 and field[x, y + 1] == -1) else 0




    # Coin Features
    num_coins= len(game_state['coins'])
    coins = game_state['coins']

    in_loop = 1 if self.coordinate_history.count((x, y)) > 2 else 0
    self.coordinate_history.append((x, y))
    #BOMB PATH
    
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    bomb_map = np.ones(field.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles = []
    for d in directions:
        if ((field[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    can_take_left = 1 if (x - 1, y) in valid_tiles else 0
    can_take_right = 1 if (x + 1, y) in valid_tiles else 0
    can_take_up  = 1 if (x, y - 1) in valid_tiles else 0
    can_take_down = 1 if (x, y + 1) in valid_tiles else 0
    can_take_wait = 1 if (x, y) in valid_tiles else 0
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    can_place_bomb = 1 if (bombs_left > 0) and (x, y) not in self.bomb_history else 0

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, field.shape[0] - 1)
    rows = range(1, field.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (field[x, y] == 0)
                 and ([field[x + 1, y], field[x - 1, y], field[x, y + 1], field[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (field[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = field == 0
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    target_up= 1 if d == (x, y - 1) else 0 
    target_down = 1 if d == (x, y + 1) else 0
    target_left= 1 if d == (x - 1, y) else 0
    target_right = 1 if d == (x + 1, y) else 0 


    # Add proposal to drop a bomb if at dead end
    should_bomb= 1 if (x, y) in dead_ends else 0
        
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        should_bomb = 1 if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1 else 0
            
    # Add proposal to drop a bomb if arrived at target and touching crate
    should_bomb = 1 if d == (x, y) and ([field[x + 1, y], field[x - 1, y], field[x, y + 1], field[x, y - 1]].count(1) > 0) else 0 
        

    



    # Create a list of feature values
    feature_values = [
        round_number,
        step_number,
        agent_score,
        is_wall_left,
        is_wall_right,
        is_wall_up,
        is_wall_down,
        num_coins,
        can_take_up,
        can_take_down,
        can_take_left,
        can_take_right,
        can_place_bomb,
        can_take_wait,
        target_up,
        target_down,
        target_left,
        target_right,
        should_bomb,
        in_loop
    ]
    
    input_features = np.array(feature_values, dtype=np.float32)
    

    


    return input_features

def look_for_targets(free_space, start, targets, logger=None):
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
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]







    


