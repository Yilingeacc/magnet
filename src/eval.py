import numpy as np
import pandas as pd
import torch
import argparse

from agent import DDPGAgent
from config import Config
from data import DataLoader
from env import Env
from utils import OUNoise


class Evaluator:
    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, agent: DDPGAgent, df_eval: pd.DataFrame(), mode: str, top_K=5):
        recall_scores = []
        ndcg_scores = []

        for _, row in df_eval.iterrows():
            group = row['group']
            history = row['history']
            item_true = row['action']
            item_candidates = row['negative samples'] + [item_true]
            np.random.shuffle(item_candidates)

            state = [group] + history
            items_pred = agent.get_action(state=state, item_candidates=item_candidates, top_K=top_K)

            recall_score = 0
            ndcg_score = 0

            for k, item in enumerate(items_pred):
                if item == item_true:
                    recall_score = 1
                    ndcg_score = np.log2(2) / np.log2(k + 2)
                    break

            recall_scores.append(recall_score)
            ndcg_scores.append(ndcg_score)

        avg_recall_score = float(np.mean(recall_scores))
        avg_ndcg_score = float(np.mean(ndcg_scores))
        print(f'{mode.capitalize()}: Recall@{top_K} = {avg_recall_score:.4f}, NDCG@{top_K} = {avg_ndcg_score:.4f}')
        return avg_recall_score, avg_ndcg_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CAMRa2011')
    # parser.add_argument('--test_no', type=int)
    args = parser.parse_args()

    # test_no = args.test_no
    test_no = 'test'

    config = Config(dataset=args.dataset)
    data_loader = DataLoader(config)
    rating_matrix_train = data_loader.load_rating_matrix(dataset_name='train')
    df_eval_user_test = data_loader.load_eval_data(dataset_name='test', mode='user')
    df_eval_group_test = data_loader.load_eval_data(dataset_name='test', mode='group')
    env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='train')
    noise = OUNoise(config=config)
    agent = DDPGAgent(config=config, noise=noise, group2members_dict=data_loader.group2members_dict)

    # load
    agent.actor.load_state_dict(torch.load(f'./model/{args.dataset}/act_{test_no}.pth', map_location=torch.device('cpu')))
    agent.embedding.load_state_dict(torch.load(f'./model/{args.dataset}/embed_{test_no}.pth', map_location=torch.device('cpu')))

    evaluator = Evaluator(config=config)

    for top_K in config.top_K_list:
        evaluator.evaluate(agent=agent, df_eval=df_eval_user_test, mode='user', top_K=top_K)
    for top_K in config.top_K_list:
        evaluator.evaluate(agent=agent, df_eval=df_eval_group_test, mode='group', top_K=top_K)
