import os
import pickle
import random
import time
from typing import Deque

import torch

import settings as s
import numpy as np
import events as e

from environment import BombeRLeWorld
from agents import Agent


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


# Define your setup method


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def change_game_state(game_state):
    """
    Convert the game state to a format suitable for input to the DQN model.
    Modify this method to extract and preprocess the relevant features from the game state.

    :param game_state: The current game state.
    :return: The preprocessed state for input to the model.
    """
    player_position = game_state['self'][3]
    coin_positions = [coin[0] for coin in game_state['coins']]

    # Other relevant features you might want to extract from the game state

    # Combine and normalize the features
    features = [player_position[0], player_position[1]] + \
        coin_positions  # Modify this as needed
    normalized_features = np.array(features) / np.max(features)
    return normalized_features.tolist()


def act(self, game_state: dict) -> str:
    # Exploration vs. Exploitation
    random_prob = 0.1  # Adjust this exploration probability as needed

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # Exploration: Choose a random action
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    self.logger.debug("Querying model for action.")
    # Exploitation: Choose an action based on your model (weights)
    action = np.random.choice(ACTIONS, p=self.model)

    # Check if there are coins at the current position
    current_position = game_state['self'][3]
    coins_at_position = [
        coin for coin in game_state['coins'] if coin == current_position]

    # Check if there are bomb explosions nearby
    explosion_radius = 2  # Adjust this radius as needed
    explosions = game_state['explosion_map']
    nearby_explosions = []

    for x in range(current_position[0] - explosion_radius, current_position[0] + explosion_radius + 1):
        for y in range(current_position[1] - explosion_radius, current_position[1] + explosion_radius + 1):
            if 0 <= x < len(explosions) and 0 <= y < len(explosions[0]) and explosions[x][y] > 0:
                nearby_explosions.append((x, y))

    # If there are coins at the current position, avoid moving in that direction
    if coins_at_position:
        # Remove actions that move toward the current position
        valid_actions = [a for a in ACTIONS if a != 'WAIT' and a != 'BOMB']
        safe_actions = [a for a in valid_actions if a !=
                        opposite_action(action)]
        return np.random.choice(safe_actions)

    # If there are nearby bomb explosions, move away from them
    if nearby_explosions:
        # Calculate the direction to move away from the closest explosion
        closest_explosion = nearby_explosions[0]
        x_diff = current_position[0] - closest_explosion[0]
        y_diff = current_position[1] - closest_explosion[1]

        if abs(x_diff) > abs(y_diff):
            # Move horizontally to get away from the explosion
            if x_diff > 0:
                return 'RIGHT'
            else:
                return 'LEFT'
        else:
            # Move vertically to get away from the explosion
            if y_diff > 0:
                return 'DOWN'
            else:
                return 'UP'

    # Implement your logic to hunt and blow up opponent agents strategically
    if action == 'BOMB' and game_state['self'][2] > 0:
        # Check if it's safe to drop a bomb (you have bombs in stock)

        # Check if there are active bombs about to explode
        active_bombs = game_state['bombs']
        if any(bomb[1] == 0 for bomb in active_bombs):
            # There is an active bomb about to explode, so it's not safe to drop another one
            return action

        # Check the positions of opponent agents
        opponent_positions = [opponent[3] for opponent in game_state['others']]

        # Check your agent's position
        agent_position = game_state['self'][3]

        # Iterate through opponent positions
        for opponent_pos in opponent_positions:
            # Calculate the Manhattan distance between your agent and the opponent
            distance = abs(opponent_pos[0] - agent_position[0]) + \
                abs(opponent_pos[1] - agent_position[1])

            # Check if the opponent is nearby and in the same row or column
            if distance <= 4 and (opponent_pos[0] == agent_position[0] or opponent_pos[1] == agent_position[1]):
                # Drop a bomb because an opponent is nearby
                return action

    # For example, calculate distances to nearby crates and hidden coins
    crate_radius = 3  # Adjust this radius as needed
    hidden_coins = [
        coin for coin in game_state['coins'] if coin not in game_state['field']]

    # Calculate distances to nearby crates and hidden coins
    distances_to_crates = [abs(crate[0] - current_position[0]) +
                           abs(crate[1] - current_position[1]) for crate in game_state['field']]
    distances_to_hidden_coins = [abs(coin[0] - current_position[0]) +
                                 abs(coin[1] - current_position[1]) for coin in hidden_coins]

    # Check if there are nearby crates and hidden coins
    nearby_crates = [
        game_state['field'][i] for i, distance in enumerate(distances_to_crates) if distance <= crate_radius]
    nearby_hidden_coins = [
        hidden_coins[i] for i, distance in enumerate(distances_to_hidden_coins) if distance <= crate_radius]

    if nearby_crates:
        # There are nearby crates
        # You can implement a strategy to decide when to drop a bomb
        # For example, if there are nearby crates, and your agent has bombs in stock, drop a bomb
        if action == 'BOMB' and game_state['self'][2] > 0:
            return action  # Drop a bomb

        # If there are nearby crates and no bomb in stock, consider moving towards them
        # You can prioritize directions that move closer to the nearest crate
        # Implement your strategy to navigate towards crates here
        crate_distances = [abs(crate[0] - current_position[0]) +
                           abs(crate[1] - current_position[1]) for crate in nearby_crates]
        closest_crate_index = crate_distances.index(min(crate_distances))
        closest_crate = nearby_crates[closest_crate_index]

        # Calculate the direction to move towards the closest crate
        x_diff = closest_crate[0] - current_position[0]
        y_diff = closest_crate[1] - current_position[1]

        if x_diff > 0:
            return 'RIGHT'
        elif x_diff < 0:
            return 'LEFT'
        elif y_diff > 0:
            return 'DOWN'
        elif y_diff < 0:
            return 'UP'

    # If there are nearby hidden coins, prioritize collecting them
    if nearby_hidden_coins:
        # You can implement a strategy to navigate towards hidden coins
        # Prioritize directions that move closer to the nearest hidden coin
        # Implement your strategy to collect hidden coins here
        coin_distances = [abs(coin[0] - current_position[0]) +
                          abs(coin[1] - current_position[1]) for coin in nearby_hidden_coins]
        closest_coin_index = coin_distances.index(min(coin_distances))
        closest_coin = nearby_hidden_coins[closest_coin_index]

        # Calculate the direction to move towards the closest hidden coin
        x_diff = closest_coin[0] - current_position[0]
        y_diff = closest_coin[1] - current_position[1]

        if x_diff > 0:
            return 'RIGHT'
        elif x_diff < 0:
            return 'LEFT'
        elif y_diff > 0:
            return 'DOWN'
        elif y_diff < 0:
            return 'UP'

    # If it's not safe to drop a bomb or no opponent is nearby, proceed with your chosen action
    return action


def opposite_action(action):
    # Helper function to find the opposite action
    if action == 'UP':
        return 'DOWN'
    elif action == 'DOWN':
        return 'UP'
    elif action == 'LEFT':
        return 'RIGHT'
    elif action == 'RIGHT':
        return 'LEFT'
    else:
        return action


def state_to_features(game_state: dict) -> np.array:
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
    """
    Convert the game state to a feature tensor.
    """
    """
    Convert the game state to a feature tensor.
    """
    features = []

    # Extract the relevant information from the game_state dictionary
    board = game_state['field']
    coins = game_state['coins']
    player_position = game_state['self'][3]

    # Iterate through the board and create a feature map
    for row in board:
        for cell in row:
            # Feature: 1 if there's a wall, 0 otherwise
            wall_feature = 1 if cell == -1 else 0
            features.append(wall_feature)

    # Create features for each coin's distance from the player's position
    for coin in coins:
        coin_position = coin[0]
        distance_feature = np.linalg.norm(
            np.array(player_position) - np.array(coin_position))
        features.append(distance_feature)

    # Normalize the features
    normalized_features = np.array(features) / np.max(features)
    return normalized_features.tolist()


def reward_from_events(events):
    """
    Calculate the reward based on the events that occurred in the game.
    """
    game_rewards = {
        e.COIN_COLLECTED: 50,        # Reward for collecting a coin
        e.KILLED_OPPONENT: 10,       # Reward for killing an opponent
        e.INVALID_ACTION: -1,        # Penalty for taking an invalid action
        e.WAITED: -0.5,              # Penalty for waiting
        # Define your custom rewards for other events here
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum
