import numpy as np

import os
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from common.buffer import Replay_buffer as buffer
from common.utils import plot_action_values


def get_trajectory_property():
    return ["action"]


class SARSA(object):

    def __init__(self, args):

        self.args = args
        self.state_dim = args.obs_space
        self.action_dim = args.action_space
        self.buffer_size = args.buffer_capacity
        self.gamma = args.gamma
        self.lr = args.lr
        self.eps = args.epsilon
        self.eps_end = args.epsilon_end
        self.eps_delay = 1 / (args.max_episodes * 100)

        # define initial Q table
        self._q = np.zeros((self.state_dim, self.action_dim))
        self._last_action = 1

        trajectory_property = get_trajectory_property()
        self.memory = buffer(self.buffer_size, trajectory_property)
        self.memory.init_item_buffers()

    @property
    def q_values(self):
        return self._q

    def behaviour_policy(self, q):
        self.eps = max(self.eps_end, self.eps - self.eps_delay)
        return self.epsilon_greedy(q, epsilon=self.eps)

    def one_hot_policy(self, q, a):
        return np.eye(len(q))[a]

    def epsilon_greedy(self, q_values, epsilon):
        if epsilon < np.random.random():
            return np.argmax(q_values)
        else:
            return np.random.randint(np.array(q_values).shape[-1])

    def choose_action(self, observation, train=True):
        inference_output = self.inference(observation, train)
        if train:
            self.add_experience(inference_output)
        return inference_output

    def inference(self, observation, train):
        action = self._last_action
        return {"action": action}

    def add_experience(self, output):
        agent_id = 0
        for k, v in output.items():
            self.memory.insert(k, agent_id, v)

    def learn(self):
        data = self.memory.get_step_data()

        next_state = data['states_next']
        state = data['states']
        reward = data['rewards']
        action = data['action']
        done = data['dones']

        next_action = self.behaviour_policy(self.q_values[next_state, :])
        target_index = self.one_hot_policy(self._q[next_state, :], next_action)
        target = reward + self.gamma * (self._q[next_state, :] @ target_index) * (1 - done)
        self._q[state, action] += self.lr * (target - self._q[state, action])
        self._last_action = next_action

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        data_path = os.path.join(base_path, "q_" + str(episode) + ".pth")
        np.savetxt(data_path, self._q, delimiter=",")

    def load(self, file):
        self._q = np.loadtxt(file, delimiter=",")
