from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import gymnasium as gym
import gymnasium
import sys

import pandas as pd

sys.modules["gym"] = gymnasium
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac import SAC
from sb3_contrib.ars import ARS


# @dataclass
# class TestResult:
# 	mean_reward: float
# 	reward_std: float
# 	training_seconds: int
#
#
# def test_algorithm(
# 		algorithm: BaseAlgorithm, n_epochs: int, n_eval_episodes: int,
# 		save_algorithm: Optional[Path] = None, seed = None
# ) -> TestResult:
# 	if seed:
# 		algorithm.set_random_seed(seed)
#
# 	start_time = datetime.now()
# 	algorithm.learn(total_timesteps = n_epochs, progress_bar = True)
# 	training_time = datetime.now() - start_time
#
# 	if save_algorithm:
# 		algorithm.save(save_algorithm)
#
# 	env = gym.make('Ant-v4')
# 	mean_reward, reward_std = evaluate_policy(algorithm, env = env, n_eval_episodes = n_eval_episodes, render = False)
#
# 	return TestResult(mean_reward, reward_std, training_seconds = training_time.seconds)
#
#
# @dataclass
# class EvaluationResult:
# 	mean_reward: float
# 	reward_std: float
# 	episode_rewards: pd.DataFrame


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
