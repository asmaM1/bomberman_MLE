from collections import namedtuple, deque

import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
from typing import List
import settings as s
import events as e
# from .callbacks import state_to_features
from environment import BombeRLeWorld as env


# During training, collect experiences and add them to the replay buffer


def store_experience(state, action, reward, next_state):
    replay_buffer.append((state, action, reward, next_state))

# Periodically, sample a mini-batch of experiences from the replay buffer


def sample_mini_batch(batch_size):
    return random.sample(replay_buffer, batch_size)


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Define your DQN model here


class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define other constants
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
EPS_START = 0.9  # Epsilon-greedy exploration
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 200  # Update target network every N episodes

# Define Transition namedtuple
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

# Define a replay buffer with a maximum capacity
REPLAY_BUFFER_SIZE = 10000
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)


class MyTrainingClass:
    def __init__(self, state_size, action_size, learning_rate):
        self.q_network = DQN(state_size, action_size)
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + \
            (EPS_START - EPS_END) * np.exp(-1.0 * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.tensor(
                    state, dtype=torch.float32, device=self.device)
                q_values = self.policy_net(state)
                action = q_values.argmax().item()
        else:
            action = random.choice(range(self.action_size))
        return action

    def store_transition(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def update_target_network(self):
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_current_state(self):
        # Return the current state of the game
        return self.current_state


# Encapsulate the training loop in a method


def train_agent_with_experience_replay(training_instance, agent, num_episodes, steps_per_episode, batch_size):
    for episode in range(num_episodes):
        for step in range(steps_per_episode):
            # Your agent's actions and environment interactions here
            # Get the current state from the game environment
            state = env.get_state_for_agent(training_instance, agent)

            # Select an action using your agent's policy
            action = agent.select_action(state)

            # Take the selected action and observe the next state and reward
            next_state, reward = env.take_action(action)

            # Store experiences as (state, action, reward, next_state) in replay_buffer
            agent.store_transition(state, action, reward, next_state)

            # Periodically update the agent using experience replay
            if len(replay_buffer) >= batch_size:
                mini_batch = random.sample(replay_buffer, batch_size)
                # Implement this method to update your agent
                agent.update_with_mini_batch(mini_batch)


# Define the number of episodes and steps per episode
num_episodes = 400
steps_per_episode = 400

state_size = 100
action_size = 6
learning_rate = 0.001

# Create an instance of your training class
agent = MyTrainingClass(state_size, action_size, learning_rate)


def calculate_q_values(agent, state):
    # Convert the state to a tensor
    state = torch.tensor(state, dtype=torch.float32, device=agent.device)

    # Pass the state through the Q-network to get Q-values for all actions
    q_values = agent.policy_net(state)

    # Convert the Q-values tensor to a numpy array for further processing if needed
    q_values = q_values.cpu().detach().numpy()

    return q_values


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.memory = deque(maxlen=10000)
    self.train = True
    self.epsilon = EPS_START
    self.epsilon_decay = (EPS_START - EPS_END) / EPS_DECAY
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(training_instance, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param training_instance: The instance of YourTrainingClass.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    training_instance.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    reward = reward_from_events(training_instance, events)

    training_instance.transitions.append(
        Transition(old_state, self_action, new_state, reward))

    # Provide rewards for crate destruction and coin collection
    if e.BOMB_DROPPED in events:
        # Provide a reward for dropping a bomb (encourage bomb usage)
        training_instance.transitions[-1] = training_instance.transitions[-1]._replace(
            reward=training_instance.transitions[-1].reward + 0.1)

    if e.BOMB_EXPLODED in events:
        # Provide a reward for crate destruction
        # Adjust the reward value as needed
        crate_destroyed_reward = events.count(e.BOMB_EXPLODED) * 0.2
        training_instance.transitions[-1] = training_instance.transitions[-1]._replace(
            reward=training_instance.transitions[-1].reward + crate_destroyed_reward)

    if e.COIN_COLLECTED in events:
        # Provide a reward for collecting a coin
        training_instance.transitions[-1] = training_instance.transitions[-1]._replace(
            reward=training_instance.transitions[-1].reward + 1.0)
    # Provide rewards for killing opponent agents
    if e.KILLED_OPPONENT in events:
        # Provide a reward for killing an opponent agent
        # Adjust the reward value as needed
        opponent_killed_reward = events.count(e.KILLED_OPPONENT) * 5.0
        training_instance.transitions[-1] = training_instance.transitions[-1]._replace(
            reward=training_instance.transitions[-1].reward + opponent_killed_reward)

    # Check for specific events related to opposing agents
    for event in events:
        if event == e.KILLED_OPPONENT:
            # You successfully defeated an opponent, assign a positive reward
            opponent_killed_reward = events.count(e.KILLED_OPPONENT) * 5.0
            training_instance.transitions[-1] = Transition(
                old_game_state, self_action, new_game_state, reward=5
            )

        if event == e.BOMB_DROPPED:
            # Encourage the agent to move away from opponent bombs
            training_instance.transitions[-1] = training_instance.transitions[-1]._replace(
                reward=training_instance.transitions[-1].reward - 0.1)

        if event == e.KILLED_SELF:
            # Penalize the agent for getting killed by an opponent
            training_instance.transitions[-1] = training_instance.transitions[-1]._replace(
                reward=training_instance.transitions[-1].reward - 10.0)

        if event == e.OPPONENT_ELIMINATED:
            # Provide a reward when an opponent gets eliminated
            eliminated_opponent_reward = events.count(
                e.OPPONENT_ELIMINATED) * 2.0
            training_instance.transitions[-1] = training_instance.transitions[-1]._replace(
                reward=training_instance.transitions[-1].reward + eliminated_opponent_reward)

        if event == e.SURVIVED_ROUND:
            # Provide a reward for surviving the round
            training_instance.transitions[-1] = training_instance.transitions[-1]._replace(
                reward=training_instance.transitions[-1].reward + 3.0)

        if event == e.GOT_KILLED:
            # Encourage the agent to move away from opponent bombs
            training_instance.transitions[-1] = training_instance.transitions[-1]._replace(
                reward=training_instance.transitions[-1].reward - 0.1)


def optimize_model(training_instance):
    if len(training_instance.memory) < BATCH_SIZE:
        return

    transitions = random.sample(training_instance.memory, BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=training_instance.device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                         if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = training_instance.policy_net(
        state_batch).gather(1, action_batch.unsqueeze(1))

    next_state_values = torch.zeros(
        BATCH_SIZE, device=training_instance.device)
    next_state_values[non_final_mask] = training_instance.target_net(
        non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (
        next_state_values * GAMMA) + reward_batch

    # Compute Hubert loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    training_instance.optimizer.zero_grad()
    loss.backward()
    training_instance.optimizer.step()
    # training_instance.optimize_model()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(
        last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -0.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

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
