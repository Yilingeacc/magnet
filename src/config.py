import os

import torch


class Config:
    def __init__(self, dataset):
        # Data
        self.dataset = dataset
        self.data_folder_path = os.path.join('..', 'data', self.dataset)
        self.item_path = os.path.join(self.data_folder_path, 'movies.dat')
        self.user_path = os.path.join(self.data_folder_path, 'users.dat')
        self.group_path = os.path.join(self.data_folder_path, 'groupMember.dat')
        self.saves_folder_path = os.path.join('saves', self.dataset)

        self.item_num = 7710  # CAMRa2011
        self.user_num = 602  # CAMRa2011
        self.group_num = None
        self.total_group_num = None

        # Recommendation system
        self.history_length = 5
        self.top_K_list = [5, 10, 20]
        self.rewards = [0, 1]

        # neural network parameters
        self.embedding_size = 32
        self.state_size = self.history_length + 1
        self.action_size = 1
        self.embedded_state_size = self.state_size * self.embedding_size
        self.embedded_action_size = self.action_size * self.embedding_size
        self.hidden_sizes = (256, 256)
        self.n_head = 2

        # Environment
        self.env_n_components = self.embedding_size
        self.env_tol = 1e-4
        self.env_max_iter = 1000
        self.env_alpha = 0.001

        # DDPG algorithm
        self.tau = 1e-3
        self.gamma = 0.9

        # OU noise
        self.ou_mu = 0.0
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.ou_epsilon = 1.0

        # Optimization parameters
        self.batch_size = 64
        self.buffer_size = 100000
        self.num_episodes = 10000
        self.num_steps = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-6

        # Eval
        self.eval_per_iter = 10

        # GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
