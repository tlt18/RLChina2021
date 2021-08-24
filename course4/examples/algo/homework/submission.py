# # This is homework.
# # Load your model and submit this to Jidi

import torch
import os

# load critic
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from critic import Critic
import numpy as np
# 1.prepare agent for game
# TODO
class IQL:
    def __init__(self):
        # pass
        self.input_size, self.output_size, self.hidden_size = [18, 4, 64]
        self.critic_eval = Critic(self.input_size, self.output_size, self.hidden_size)

    def load(self,critic_net):
        self.critic_eval.load_state_dict(torch.load(critic_net))
        
    def choose_action(self,obs):
        obs = torch.tensor(obs,dtype=torch.float).view(1,-1)
        action = torch.argmax(self.critic_eval(obs)).item()
        return action

# 2.convert state to obvervation
def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding

def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map

def get_observations(state, id, obs_dim):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    observations = np.zeros((1, obs_dim)) # todo
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    agents_index = [id]
    for i, element in enumerate(agents_index):
        # # self head position
        observations[i][:2] = snakes_positions_list[element][0][:]

        # head surroundings
        head_x = snakes_positions_list[element][0][1]
        head_y = snakes_positions_list[element][0][0]

        head_surrounding = get_surrounding(state, board_width, board_height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions # todo: to check
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, element, 0)
        observations[i][16:] = snake_heads.flatten()[:]
    return observations.squeeze().tolist()


def action_from_algo_to_env(joint_action):
    '''
    :param joint_action:
    :return: wrapped joint action: one-hot
    '''
    joint_action_ = []
    for a in range(1):
        action_a = joint_action
        each = [0] * 4
        each[action_a] = 1
        joint_action_.append(each)
    return joint_action_


critic_net0 = os.path.dirname(os.path.abspath(__file__)) + '/critic_0_999.pth'
agent0 = IQL()
agent0.load(critic_net0)
critic_net1 = os.path.dirname(os.path.abspath(__file__)) + '/critic_1_999.pth'
agent1 = IQL()
agent1.load(critic_net1)

# todo
def my_controller(observation, action_space, is_act_continuous=False):
    play_ID = observation['controlled_snake_index'] - 2
    obs = get_observations(observation, play_ID, 18)
    if play_ID == 0:
        action_ = agent0.choose_action(obs)
    elif play_ID == 1:
        action_ = agent1.choose_action(obs)
    reaction_ = action_from_algo_to_env(action_)
    return reaction_