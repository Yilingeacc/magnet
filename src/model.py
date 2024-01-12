from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def build_mlp(dims: [int]) -> nn.Sequential:
    net_list = list()
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]
    return nn.Sequential(*net_list)


def clones(module: nn.Module, n: int):
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


class Actor(nn.Module):
    def __init__(self, embedded_state_size: int, action_weight_size: int, hidden_sizes: Tuple[int]):
        super(Actor, self).__init__()
        self.net = build_mlp([embedded_state_size, *hidden_sizes, action_weight_size])

    def forward(self, embedded_state):
        return self.net(embedded_state)


class Critic(nn.Module):
    def __init__(self, embedded_state_size: int, embedded_action_size: int, hidden_sizes: Tuple[int]):
        super(Critic, self).__init__()
        self.net = build_mlp([embedded_state_size + embedded_action_size, *hidden_sizes, 1])

    def forward(self, embedded_state, embedded_action):
        return self.net(torch.cat([embedded_state, embedded_action], dim=-1))


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Implementation
    See: Attention is all you need(https://arxiv.org/abs/1706.03762)
    """
    def __init__(self, d_model, n_head):
        super().__init__()
        d_model = d_model
        d_k = d_model // n_head
        assert d_k * n_head == d_model
        self.h = n_head
        self.d_k = d_k
        self.d_model = d_model
        self.linears = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value):
        q, k, v = [l(x).view(-1, self.h, self.d_k).transpose(0, 1) for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** 0.5
        attn = F.softmax(scores, dim=-1)
        x = torch.matmul(attn, v).transpose(0, 1).contiguous().view(-1, self.h * self.d_k)
        return self.linears[-1](x)


class Embedding(nn.Module):
    def __init__(self, embedding_size: int, user_num: int, item_num: int, group_num: int, nhead: int = 8):
        super(Embedding, self).__init__()
        self.user_embedding = nn.Embedding(user_num + 1, embedding_size)
        self.item_embedding = nn.Embedding(item_num + 1, embedding_size)
        self.group_embedding = nn.Embedding(group_num + 1, embedding_size)

        self.self_attn = MultiHeadAttention(embedding_size, nhead)
        self.cross_attn = MultiHeadAttention(embedding_size, nhead)
        self.proj = build_mlp([embedding_size, 2 * embedding_size, 1])

        self.embedding_size = embedding_size
        nn.init.zeros_(self.group_embedding.weight)

    def forward(self, group_id, group_members, history):
        group_bias = self.group_embedding(group_id)
        embedded_members = self.user_embedding(group_members)
        embedded_history = self.item_embedding(history)

        self_attn = self.self_attn(embedded_members, embedded_members, embedded_members)
        cross_attn = self.cross_attn(self_attn, embedded_history, embedded_history)
        group_attn = F.softmax(self.proj(cross_attn), dim=0)

        embedded_group = torch.squeeze(torch.inner(group_attn.T, embedded_members.T))
        embedded_group = (embedded_group + group_bias).squeeze()
        embedded_state = torch.cat([embedded_group, torch.flatten(embedded_history, start_dim=-2)], dim=-1)
        return embedded_state
