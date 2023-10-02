import os
import pickle
import random
import numpy as np
import networkx as nx


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']





def setup(self):
        """
        Setup your code. This is called once when loading each agent.
        Make sure that you prepare everything such that act(...) can be called.

        When in training mode, the separate `setup_training` in train.py is called
        after this method. This separation allows you to share your trained agent
        with other students, without revealing your training code.

        :param self: This object is passed to all callbacks and you can set arbitrary values.
        """

    
        if self.train or not os.path.isfile("my-saved-model.pt"):
            self.logger.info("Setting up model from scratch.")
            weights = np.random.rand(len(ACTIONS))
            self.model = weights / weights.sum()
        else:
            self.logger.info("Loading model from saved state.")
            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)





def should_place_bomb(game_state: dict) -> bool:
    # Implement logic to check if it's safe to place a bomb
    # Consider immediate and future danger
    bombs = game_state['bombs']
    player_position = game_state['self'][3]
    blast_strength = game_state['self'][2]
    explosion_map = game_state['explosion_map']

    # Check if there is an immediate danger at the current position
    if explosion_map[player_position] > 0:
        return False

    # Check if there will be danger in the next step or two due to bomb placement
    for bomb in bombs:
        bomb_position, bomb_timer = bomb
        if bomb_timer <= 2:  # The bomb will explode in the next two steps (including the current one)
            if np.linalg.norm(np.array(player_position) - np.array(bomb_position)) <= blast_strength:
                return False

    return True

def get_safe_actions(game_state: dict) -> list:
    # Implement logic to get a list of safe movement actions
    # Avoid actions that lead to immediate danger
    safe_actions = []
    player_position = game_state['self'][3]
    potential_moves = {
        "UP": (0, -1),
        "RIGHT": (1, 0),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
    }
    for action, move in potential_moves.items():
        new_position = (player_position[0] + move[0], player_position[1] + move[1])
        if is_position_safe(new_position, game_state):
            safe_actions.append(action)
    return safe_actions

def is_position_safe(position, game_state: dict) -> bool:
    # Implement logic to check if a position is safe
    # Consider immediate danger (explosions)
    explosion_map = game_state['explosion_map']

    # Check if the position is within an explosion in the next step
    if explosion_map[position] > 0:
        return False

    return True



def act(self,game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Exploration vs exploitation
    random_prob = 0.1

    if random.random() < random_prob or game_state['round'] < 100:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    # Implement logic to check if it's safe to place a bomb
    if should_place_bomb(game_state):
        return "BOMB"

    # Implement logic for safe movement
    safe_actions = get_safe_actions(game_state)
    
    if safe_actions:
        return random.choice(safe_actions)
    else:
        # If there are no safe actions, wait.
        return "WAIT"
    




def state_to_features(game_state: dict) -> np.array:
    features = []

    # Extract relevant information
    board = game_state['field']
    player_position = game_state['self'][3]
    bombs = game_state['bombs']

    # Iterate through the board and create a feature map
    for row in board:
        for cell in row:
            # Feature: 1 if there's a wall, 0 otherwise
            wall_feature = 1 if cell == -1 else 0
            features.append(wall_feature)

    # Create features for bomb safety
    for bomb in bombs:
        bomb_position = bomb[0]

        # Calculate Manhattan distance
        manhattan_distance_to_bomb = abs(player_position[0] - bomb_position[0]) + abs(player_position[1] - bomb_position[1])

        # Calculate Euclidean distance
        euclidean_distance_to_bomb = np.linalg.norm(np.array(player_position) - np.array(bomb_position))

        blast_radius = bomb[1]

        # Feature: 1 if the bomb is within blast radius, 0 otherwise
        within_blast_radius = 1 if euclidean_distance_to_bomb <= blast_radius else 0
        
        features.append(manhattan_distance_to_bomb)
        #features.append(euclidean_distance_to_bomb)
        features.append(within_blast_radius)

    # Normalize the features
    normalized_features = np.array(features) / np.max(features)
    return normalized_features.tolist()




