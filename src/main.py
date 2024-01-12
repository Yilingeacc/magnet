import pandas as pd
import argparse

import torch

from agent import DDPGAgent
from config import Config
from data import DataLoader
from env import Env
from eval import Evaluator
from utils import OUNoise


def train(config: Config, env: Env, agent: DDPGAgent, evaluator: Evaluator,
          df_eval_user: pd.DataFrame(), df_eval_group: pd.DataFrame()):
    for episode in range(config.num_episodes):
        state = env.reset()
        agent.noise.reset()
        episode_reward = 0

        for step in range(config.num_steps):
            action = agent.get_action(state)
            new_state, reward, _, _ = env.step(action)
            agent.replay_memory.push((state, action, reward, new_state))
            state = new_state
            episode_reward += reward

            if len(agent.replay_memory) >= config.batch_size:
                agent.update()

        avg_reward = episode_reward / config.num_steps
        print(f'episode: {episode}, average reward: {avg_reward:.3f}')
        if (episode + 1) % config.eval_per_iter == 0:
            agent.save(episode)
            for top_K in config.top_K_list:
                evaluator.evaluate(agent=agent, df_eval=df_eval_user, mode='user', top_K=top_K)
            for top_K in config.top_K_list:
                evaluator.evaluate(agent=agent, df_eval=df_eval_group, mode='group', top_K=top_K)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MovieLens-Rand')
    args = parser.parse_args()

    config = Config(dataset=args.dataset)
    config.device = torch.device('cpu')
    data_loader = DataLoader(config)
    rating_matrix_train = data_loader.load_rating_matrix(dataset_name='train')
    df_eval_user_test = data_loader.load_eval_data(mode='user', dataset_name='test')
    df_eval_group_test = data_loader.load_eval_data(mode='group', dataset_name='test')
    env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='train')
    noise = OUNoise(config=config)
    agent = DDPGAgent(config=config, noise=noise, group2members_dict=data_loader.group2members_dict)
    evaluator = Evaluator(config=config)
    train(config=config, env=env, agent=agent, evaluator=evaluator,
          df_eval_user=df_eval_user_test, df_eval_group=df_eval_group_test)
