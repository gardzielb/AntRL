from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import gymnasium as gym
import gymnasium
import sys

sys.modules["gym"] = gymnasium
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac import SAC
from sb3_contrib.ars import ARS


@dataclass
class TestResult:
	mean_reward: float
	reward_std: float
	training_seconds: int


def test_algorithm(
		algorithm: BaseAlgorithm, n_epochs: int, n_eval_episodes: int, save_algorithm: Optional[str] = None
) -> TestResult:
	start_time = datetime.now()
	algorithm.learn(total_timesteps = n_epochs, progress_bar = True)
	training_time = datetime.now() - start_time

	if save_algorithm:
		algorithm.save(save_algorithm)

	env = gym.make('Ant-v4')
	mean_reward, reward_std = evaluate_policy(algorithm, env = env, n_eval_episodes = n_eval_episodes, render = False)

	return TestResult(mean_reward, reward_std, training_seconds = training_time.seconds)


if __name__ == '__main__':
	print(f'Testing ARS')
	ars_env = gym.make('Ant-v4', healthy_reward = 0.95)
	ars = ARS(
		'LinearPolicy', ars_env,
		alive_bonus_offset = -1,
		delta_std = 0.025,
		learning_rate = 0.015,
		n_delta = 60,
		n_top = 20,
		device = 'cuda'
	)
	ars_result = test_algorithm(ars, n_epochs = 1_000_000, n_eval_episodes = 5)

	print(f'Testing SAC')
	sac_env = gym.make('Ant-v4', healthy_reward = 0.01)
	sac = SAC("MlpPolicy", ars_env, learning_starts = 10_000)
	sac_result = test_algorithm(sac, n_epochs = 100_000, n_eval_episodes = 5)

	print(f'ARS result = {ars_result}, SAC result = {sac_result}')
