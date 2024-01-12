import random
from collections import deque

import torch

from config import Config


class OUNoise:
    def __init__(self, config: Config):
        self.embedded_action_size = config.embedded_action_size
        self.ou_mu = config.ou_mu
        self.ou_theta = config.ou_theta
        self.ou_sigma = config.ou_sigma
        self.ou_epsilon = config.ou_epsilon
        self.ou_state = None
        self.reset()

    def reset(self):
        self.ou_state = torch.ones(self.embedded_action_size) * self.ou_mu

    def evolve_state(self):
        self.ou_state += self.ou_theta * (self.ou_mu - self.ou_state) \
            + self.ou_sigma * torch.randn(self.embedded_action_size)

    def get_ou_noise(self):
        self.evolve_state()
        return self.ou_state.copy()


class ReplayMemory:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, experience: tuple):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return batch
