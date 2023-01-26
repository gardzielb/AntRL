import gymnasium as gym
import gymnasium
import sys
from typing import Union

sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC


def train_sac(healthy_reward: float, ent_coef: Union[int, str], n_steps: int, out_file: str, seed: int):
	env = gym.make('Ant-v4', healthy_reward = healthy_reward)
	env.reset(seed = seed)

	model = SAC("MlpPolicy", env, learning_starts = 10_000, ent_coef=ent_coef)
	model.learn(total_timesteps = n_steps, progress_bar = True)
	model.save(out_file)


def prepare_sac_models(out_dir: str):
	configs = [
		(1.0, 'auto', 200_000),
		(0.1, 'auto', 200_000),
		(0.01, 'auto', 200_000),
		(0.1, 'auto', 1_000_000),
		(1.0, 'auto', 1_000_000),
		(1.0, 0.5, 1_000_000),
		(1.0, 0.1, 1_000_000),
		(1.0, 0.01, 1_000_000)
	]

	for healthy_reward, ent_coef, n_steps in configs:
		print(f'Training SAC: healthy_reward = {healthy_reward}, ent_coef = {ent_coef}, n_steps = {n_steps}')
		file_name = f'{out_dir}/sac_e{ent_coef}_h{healthy_reward}_{n_steps // 1_000_000}M.zip'

		train_sac(
			healthy_reward = healthy_reward, ent_coef = ent_coef, n_steps = n_steps, seed=42, out_file = file_name
		)
