from typing import List
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as functional
from torch import optim, nn
import time

import model as model
import utils as utils
from config import Config


class DDPGAgent:
    """
    DDPG(Deep Deterministic Policy Gradient) Agent Implementation
    See: Continuous control with deep reinforcement learning (https://arxiv.org/abs/1509.02971)
    """
    def __init__(self, config: Config, noise: utils.OUNoise, group2members_dict: dict):
        self.config = config
        self.noise = noise
        self.group2members_dict = group2members_dict
        self.tau = config.tau
        self.gamma = config.gamma
        self.device = config.device
        self.dataset = config.dataset

        self.embedding = model.Embedding(embedding_size=config.embedding_size,
                                         user_num=config.user_num,
                                         item_num=config.item_num,
                                         group_num=config.total_group_num,
                                         nhead=config.n_head).to(config.device)
        self.actor = model.Actor(embedded_state_size=config.embedded_state_size,
                                 action_weight_size=config.embedded_action_size,
                                 hidden_sizes=config.hidden_sizes).to(config.device)
        self.critic = model.Critic(embedded_state_size=config.embedded_state_size,
                                   embedded_action_size=config.embedded_action_size,
                                   hidden_sizes=config.hidden_sizes).to(config.device)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.embedding_optimizer = optim.Adam(self.embedding.parameters(), lr=config.learning_rate,
                                              weight_decay=config.weight_decay)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay)

        self.replay_memory = utils.ReplayMemory(buffer_size=config.buffer_size)
        self.critic_criterion = nn.MSELoss()

        print(self.embedding)
        print(self.actor)
        print(self.critic)

    def soft_update(self, network: nn.Module, network_target: nn.Module):
        for cur, tar in zip(network.parameters(), network_target.parameters()):
            tar.data.copy_(cur.data * self.tau + tar.data * (1 - self.tau))

    def get_action(self, state: list, item_candidates: list = None, top_K: int = 1, with_noise=False):
        with torch.no_grad():
            states = [state]
            embedded_states = self.embed_states(states)
            action_weights = self.actor(embedded_states)
            action_weight = torch.squeeze(action_weights)
            if with_noise:
                action_weight += self.noise.get_ou_noise()

            if item_candidates is None:
                item_embedding_weight = self.embedding.item_embedding.weight.clone()
            else:
                item_candidates = np.array(item_candidates)
                item_candidates_tensor = torch.tensor(item_candidates, dtype=torch.int).to(self.device)
                item_embedding_weight = self.embedding.item_embedding(item_candidates_tensor)

            scores = torch.inner(action_weight, item_embedding_weight).detach().cpu().numpy()
            sorted_score_indices = np.argsort(scores)[:top_K]

            if item_candidates is None:
                action = sorted_score_indices
            else:
                action = item_candidates[sorted_score_indices]
            action = np.squeeze(action)
            if top_K == 1:
                action = action.item()
        return action

    def get_embedded_actions(self, embedded_states: torch.Tensor, target=False):
        if not target:
            action_weights = self.actor(embedded_states)
        else:
            action_weights = self.actor_target(embedded_states)

        item_embedding_weight = self.embedding.item_embedding.weight.clone()
        scores = torch.inner(action_weights, item_embedding_weight)
        embedded_actions = torch.inner(functional.gumbel_softmax(scores, hard=True), item_embedding_weight.t())
        return embedded_actions

    def embed_state(self, state: list):
        group_id = state[0]
        group_id_tensor = torch.tensor([group_id], dtype=torch.int).to(self.device)
        group_members = torch.tensor(self.group2members_dict[group_id], dtype=torch.int).to(self.device)
        history = torch.tensor(state[1:], dtype=torch.int).to(self.device)
        embedded_state = self.embedding(group_id_tensor, group_members, history)
        return embedded_state

    def embed_states(self, states: List[list]):
        embedded_states = torch.stack([self.embed_state(state) for state in states], dim=0)
        return embedded_states

    def embed_actions(self, actions: list):
        actions = torch.tensor(actions, dtype=torch.int).to(self.device)
        embedded_actions = self.embedding.item_embedding(actions)
        return embedded_actions

    def update(self):
        batch = self.replay_memory.sample(self.config.batch_size)
        states, actions, rewards, next_states = list(zip(*batch))

        self.embedding_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        embedded_states = self.embed_states(states)
        embedded_actions = self.embed_actions(actions)
        rewards = torch.unsqueeze(torch.tensor(rewards, dtype=torch.int).to(self.device), dim=-1)
        embedded_next_states = self.embed_states(next_states)
        q_values = self.critic(embedded_states, embedded_actions)

        with torch.no_grad():
            embedded_next_actions = self.get_embedded_actions(embedded_next_states, target=True)
            next_q_values = self.critic_target(embedded_next_states, embedded_next_actions)
            q_values_target = rewards + self.gamma * next_q_values

        critic_loss = self.critic_criterion(q_values, q_values_target)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        embedded_states = self.embed_states(states)
        actor_loss = -self.critic(embedded_states, self.get_embedded_actions(embedded_states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.embedding_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()

    def save(self, episode: int):
        print(f'save to disk: ./model/{self.dataset}/')
        torch.save(self.actor.state_dict(), f'./model/{self.dataset}/act_{episode}.pth')
        torch.save(self.critic.state_dict(), f'./model/{self.dataset}/cri_{episode}.pth')
        torch.save(self.embedding.state_dict(), f'./model/{self.dataset}/embed_{episode}.pth')

    def load(self, episode: int):
        print(f'load from disk: ./model/{self.dataset}/')
        self.actor.load_state_dict(torch.load(f'./model/{self.dataset}/act_{episode}.pth'))
        self.critic.load_state_dict(torch.load(f'./model/{self.dataset}/cri_{episode}.pth'))
        self.embedding.load_state_dict(torch.load(f'./model/{self.dataset}/embed_{episode}.pth'))
