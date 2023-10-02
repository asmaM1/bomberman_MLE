from collections import namedtuple, deque
import pickle
from typing import List

from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import BombeRLeWorld
import events as e
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_INT = {'UP':0, 'RIGHT' : 1 , 'DOWN': 2, 'LEFT': 3, 'WAIT':4, 'BOMB': 5}

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

     # Define the Prioritized Replay Buffer
class PrioritizedReplayBuffer:
        def __init__(self, capacity, alpha=0.6):
            self.capacity = capacity
            self.alpha = alpha
            self.buffer = []
            self.priorities = np.zeros(capacity)
            self.position = 0

        def add(self, experience, priority):
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
                self.priorities[self.position] = priority
                self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            priorities = self.priorities[:len(self.buffer)]
            probs = priorities ** self.alpha
            probs /= probs.sum()

            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]
            weights = (len(self.buffer) * probs[indices]) ** -self.beta
            weights /= weights.max()

            return indices, samples, torch.tensor(weights, dtype=torch.float32)

        def update_priorities(self, indices, priorities):
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority


def setup_training(self):

    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
    


    #Hyperparameters
    # 
    learning_rate = 0.001 
    memory_capacity = 10000
    alpha = 0.5
    
    input_size = 20
    output_size = 6
   # Initialize networks and optimizer
    self.model = DQN(input_size, output_size)
    self.target_net = DQN(input_size, output_size)
    self.target_net.load_state_dict(self.model.state_dict())
    self.target_net.eval()
    self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    self.replay_buffer = PrioritizedReplayBuffer(memory_capacity, alpha)
    self.total_reward = 0

    
    
    
    





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

    

    gamma = 0.99
    batch_size = 32
    


    
    next_state = state_to_features(self,new_game_state)
    action = self_action
    action_int = ACTION_TO_INT[action]
    state = state_to_features(self,old_game_state)
    reward = reward_from_events(self, events)
    step_reward = reward_from_events(self,append_custom_events(self,old_game_state,events))
    total_reward = step_reward + reward
    self.replay_buffer.add((state, action_int, total_reward, next_state), priority=1.0)

    if len(self.replay_buffer.buffer) >= batch_size:
        indices, batch, weights = self.replay_buffer.sample(batch_size)
        indices = torch.tensor(indices, dtype=torch.long)

        batch = list(zip(*batch))
        states = torch.tensor(batch[0], dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.long)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.tensor(batch[3], dtype=torch.float32)
        

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_values = rewards + gamma * next_q_values

        loss = (weights * (q_values - target_values).pow(2)).mean()
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_errors = (q_values - target_values).detach().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)


        # Update the target network with the current model's weights
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()



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
    self.transitions.append(Transition(state_to_features(self,last_game_state), last_action, None, reward_from_events(self, events)))

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
        e.COIN_COLLECTED: 50,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -5000,
        e.INVALID_ACTION: -400,
        e.WAITED:-100,
    
        'LIFE_SAVING_MOVE': 20,
        'DEADLY_MOVE': -150,
        'GOOD_BOMB': 10,
        'BAD_BOMB':-50,
        'MOVES_TOWARD_TARGET': 10,
        'BAD_MOVE': -40,
        'DOING_SAME':-1000
        
        
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

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
    

def append_custom_events(self,old_game_state: dict, events: List[str]) -> List[str]:
    features = state_to_features(self,old_game_state)
    """
    Appends all our custom events to the events list
    so we can calculate the total rewards out of these

    Args:
        events (List[str]): The non custom events, that happened between two game states
        old_game_state: dict: The game state before the events happened
        new_game_state: dict: The game state after the events happened

    Returns:
        List[str]: Full event list with custom events appended
    """
    

    danger_up = features[8]
    danger_down= features[9]
    danger_left = features[10]
    danger_right = features[11]
    can_bomb = features[12]
    danger_wait = features[13]
    target_up = features[14]
    target_down= features[15]
    target_left = features[16]
    target_right= features[17]
    should_bomb = features[18]
    in_loop = features[19]

    if e.INVALID_ACTION in events:
        return events

    #check, if waiting is dangerous we need to move 
    if in_loop == 1:
        events.append("DOING_SAME")
    else:
        if danger_wait == 0: 
        #check if did a life saving move
         if danger_left == 1 and e.MOVED_LEFT in events:
            events.append("LIFE_SAVING_MOVE")
         elif danger_right == 1 and e.MOVED_RIGHT in events:
            events.append("LIFE_SAVING_MOVE")
         elif danger_up == 1 and e.MOVED_UP in events:
            events.append("LIFE_SAVING_MOVE")
         elif danger_down == 1 and e.MOVED_DOWN in events:
            events.append("LIFE_SAVING_MOVE")
         else: 
            events.append("DEADLY_MOVE")
        elif should_bomb == 1 and e.BOMB_DROPPED in events and can_bomb ==1 :
         events.append("GOOD_BOMB")
        elif should_bomb == 0 and e.BOMB_DROPPED:
            events.append("BAD_BOMB")
    
        else:
         if (target_left == 1 and e.MOVED_LEFT in events) or ( target_right == 1 and e.MOVED_RIGHT in events) or ( target_up == 1 and e.MOVED_UP in events) or ( target_down and e.MOVED_DOWN in events):
            events.append("MOVES_TOWARD_TARGET")
         else:
            events.append("BAD_MOVE")
    return events








