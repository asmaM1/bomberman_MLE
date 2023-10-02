import numpy as np
from collections import namedtuple, deque
import settings as s

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
gamma=0.1


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    #save total rewards for each game
    self.total_rewards = 0

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)




def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))




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
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def monte_carlo_updated(self, episode_rewards_updated, timestep_updated):
    cumulative_reward = 0
    for i, t in enumerate(range(timestep_updated - 1, len(episode_rewards_updated))):
        cumulative_reward = cumulative_reward + np.power(gamma, i) * episode_rewards_updated[t]
    return cumulative_reward


def train_model(self):
    """
    Train the model using the Monte Carlo method.
    """
    # Create a dictionary to store cumulative returns for state-action pairs
    cumulative_returns = {}

    # Initialize cumulative returns to zero
    for episode in self.episodes:
        for state, action, _ in episode:
            if (state, action) not in cumulative_returns:
                cumulative_returns[(state, action)] = 0.0

    # Update cumulative returns with episode returns
    for episode in self.episodes:
        for state, action, return_ in episode:
            cumulative_returns[(state, action)] += return_

    # Compute the average return for each state-action pair
    for (state, action), total_return in cumulative_returns.items():
        count = sum(1 for ep in self.episodes if (state, action) in ep)
        average_return = total_return / count if count > 0 else 0.0

        # Update your model with the estimated value (average return)
        self.model.update(state, action, average_return)

        # Calculate and print the Monte Carlo return using monte_carlo function
        episode_rewards = [transition[2] for transition in episode]
        timestep = len(episode_rewards)
        monte_carlo_return = self.monte_carlo(episode_rewards, timestep)
        self.logger.debug(f'Monte Carlo Return for ({state}, {action}): {monte_carlo_return}')

        # Update your model with the estimated value (average return)
        self.model.update(state, action, average_return)



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 50,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -100,
        e.INVALID_ACTION: -20,
        e.WAITED:-30,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

