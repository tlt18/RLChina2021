import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
# ====================================== helper functions ======================================
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from common import make_grid_map, get_surrounding, get_observations


# ====================================== define algo ===========================================
# todo
class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# todo
class DQN(object):
    def __init__(self):
        self.state_dim, self.action_dim, self.hidden_size = [18 ,4 ,256 ]
        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)

    def load(self, file):
        self.critic_eval.load_state_dict(torch.load(file))

    def choose_action(self,obs):
        obs = torch.tensor(obs, dtype=torch.float).view(1, -1)
        action = torch.argmax(self.critic_eval(obs)).item()
        return action

def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        joint_action.append(one_hot_action)
    return joint_action


# ===================================== define agent =============================================
#todo
agent = DQN()
critic_path = os.path.dirname(os.path.abspath(__file__))+'/critic_20000.pth'
agent.load(critic_path)


# ================================================================================================
"""
input:
    observation: dict
    {
        1: 豆子，
        2: 第一条蛇的位置，
        3：第二条蛇的位置，
        "board_width": 地图的宽，
        "board_height"：地图的高，
        "last_direction"：上一步各个蛇的方向，
        "controlled_snake_index"：当前你控制的蛇的序号（2或3）
        }
return: 
    action: eg. [[[0,0,0,1]]]
"""
# todo
def my_controller(observation, action_space_list, is_act_continuous):
    platID = observation['controlled_snake_index']
    # platID最后在get_observations里面没有用，直接采用observation的信息了
    obs =  get_observations(observation, platID, 18)
    action_ = agent.choose_action(obs)
    return to_joint_action(action_, 1)

