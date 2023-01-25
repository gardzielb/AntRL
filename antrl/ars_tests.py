import sys
from pathlib import Path

import gymnasium
import gymnasium as gym
import numpy as np
import pandas as pd

sys.modules["gym"] = gymnasium
from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib.ars import ARS
from stable_baselines3 import SAC

from antrl.experiments import TestResult, test_algorithm


def fill_result(result_data: dict, results: list[TestResult], params: list):
	for param, extractor in params:
		param_vals = [extractor(r) for r in results]
		result_data[param].append(np.mean(param_vals))
		result_data[f'{param}_std'].append(np.std(param_vals))


class AlgorithmTester:
	def __init__(self, rng_seeds: list[int], n_eval_episodes: int, out_dir: str):
		self.rng_seeds = rng_seeds
		self.n_eval_episodes = n_eval_episodes
		self.out_dir = Path(out_dir)
		self.out_dir.mkdir(parents = True, exist_ok = True)

	def test_epochs(self, algorithm: BaseAlgorithm, n_min: int, step: int, n_steps: int, alg_desc: str) -> pd.DataFrame:
		result_data = {
			'n_epochs': [], 'seed': [], 'time': [], 'mean_reward': [], 'reward_std': []
		}

		for n_epochs in range(n_min, n_min + step * n_steps, step):
			results = []

			for seed in self.rng_seeds:
				result = test_algorithm(
					algorithm, n_epochs, self.n_eval_episodes, seed = seed,
					save_algorithm = self.out_dir.joinpath(f'{alg_desc}_n{n_epochs}')
				)
				# result = TestResult(random.random(), random.random(), random.randint(0, 3600))
				results.append(result)

				result_data['n_epochs'].append(n_epochs)
				result_data['seed'].append(seed)
				result_data['time'].append(result.training_seconds)
				result_data['mean_reward'].append(result.mean_reward)
				result_data['reward_std'].append(result.reward_std)

		return pd.DataFrame(data = result_data)

	def test_ars_params(
			self, tops: list[int], deltas: list[int], healthy_rewards: list[float], n_epochs: int
	) -> pd.DataFrame:
		result_data = {
			'n_delta': [], 'n_top': [], 'healthy_reward': [],
			'mean_reward': [], 'reward_std': [], 'seed': [], 'time': []
		}

		for healthy_reward, n_top, n_delta in zip(healthy_rewards, tops, deltas):
			env = gym.make('Ant-v4', healthy_reward = healthy_reward)
			results = []

			for seed in self.rng_seeds:
				ars = ARS(
					policy = 'LinearPolicy',
					env = env,
					alive_bonus_offset = -1,
					delta_std = 0.025,
					learning_rate = 0.015,
					n_delta = n_delta,
					n_top = n_top,
					device = 'cuda'
				)

				result = test_algorithm(
					ars, n_epochs, self.n_eval_episodes, seed = seed,
					save_algorithm = self.out_dir.joinpath(f'ars_d{n_delta}_t{n_top}_h{healthy_reward}')
				)
				# result = TestResult(random.random(), random.random(), random.randint(0, 3600))
				results.append(result)

				result_data['seed'].append(seed)
				result_data['time'].append(result.training_seconds)
				result_data['mean_reward'].append(result.mean_reward)
				result_data['reward_std'].append(result.reward_std)
				result_data['healthy_reward'].append(healthy_reward)
				result_data['n_delta'].append(n_delta)
				result_data['n_top'].append(n_top)

		return pd.DataFrame(data = result_data)


if __name__ == '__main__':
	target = sys.argv[1]
	out_dir = f'_out/{sys.argv[2]}'

	tester = AlgorithmTester(rng_seeds = [19, 454, 966], n_eval_episodes = 5, out_dir = f'{out_dir}/models')

	if target == 'ars-epochs':
		print('================================== Testing ARS epoch count ==================================')
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
		ars_epochs_df = tester.test_epochs(ars, n_min = 3_000_000, step = 1_000_000, n_steps = 3, alg_desc = 'ars')
		ars_epochs_df.to_csv(f'{out_dir}/ars-epochs.csv', index = False)

	elif target == 'sac-epochs':
		print('\n================================== Testing SAC epoch count ==================================')
		sac_env = gym.make('Ant-v4', healthy_reward = 0.01)
		sac = SAC("MlpPolicy", sac_env, learning_starts = 10_000)
		sac_epochs_df = tester.test_epochs(sac, n_min = 100_000, step = 100_000, n_steps = 3, alg_desc = 'sac')
		sac_epochs_df.to_csv(f'{out_dir}/sac-epochs.csv', index = False)

	elif target == 'ars-params':
		print('\n===================================== Testing ARS params =====================================')

		ars_params_df = tester.test_ars_params(
			tops = [10, 20, 30], deltas = [60, 40, 60], healthy_rewards = [0.95] * 3, n_epochs = 5_000_000
		)
		ars_params_df.to_csv(f'{out_dir}/ars-params.csv', index = False)

	elif target == 'healthy-reward':
		print('\n================================= Testing ARS healthy reward =================================')
		ars_params_df = tester.test_ars_params(
			tops = [20], deltas = [60], healthy_rewards = [1.0], n_epochs = 5_000_000
		)
		ars_params_df.to_csv(f'{out_dir}/ars-healthy-1.csv', index = False)
