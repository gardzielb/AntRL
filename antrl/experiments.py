import gymnasium as gym
import gymnasium
import sys

import pandas as pd

sys.modules["gym"] = gymnasium
from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_model(path: str, algorithm, seed: int, n_eval_episodes: int, render: bool) -> pd.DataFrame:
	episode_rewards: dict[int, list[float]] = dict()

	def callback(eval_globals, eval_locals):
		episode_no = eval_globals['episode_counts'][0]
		if episode_no in episode_rewards:
			episode_rewards[episode_no].append(eval_globals['current_rewards'][0])
		else:
			episode_rewards[episode_no] = [eval_globals['current_rewards'][0]]

	env = gym.make('Ant-v4', render_mode = 'human') if render else gym.make('Ant-v4')
	env.reset(seed = seed)
	model = algorithm.load(path, env)
	mean_reward, reward_std = evaluate_policy(
		model, env, return_episode_rewards = True, callback = callback,
		deterministic = False, n_eval_episodes = n_eval_episodes
	)

	return pd.DataFrame.from_dict(episode_rewards, orient = 'index').T
