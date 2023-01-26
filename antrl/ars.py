from datetime import datetime

import gymnasium as gym
import gymnasium
import sys

sys.modules["gym"] = gymnasium
from sb3_contrib import ARS


def train_ars(healthy_reward: float, n_delta: int, n_top: int, n_epochs: int, out_file: str, seed: int):
	env = gym.make('Ant-v4', healthy_reward = healthy_reward)
	env.reset(seed = seed)

	model = ARS(
		'LinearPolicy', env,
		alive_bonus_offset = -1,
		delta_std = 0.025,
		learning_rate = 0.015,
		n_delta = n_delta,
		n_top = n_top,
		device = 'cuda'
	)

	model.learn(total_timesteps = n_epochs, progress_bar = True)
	model.save(out_file)


def prepare_ars_models(out_dir: str):
	configs = [
		(1.0, 60, 30, 5_000_000),
		(1.0, 60, 10, 5_000_000),
		(1.0, 40, 20, 5_000_000),
		(1.0, 60, 20, 2_000_000),
		(1.0, 60, 20, 3_500_000),
		(0.95, 60, 20, 5_000_000),
		(0.9, 60, 20, 5_000_000)
	]

	for healthy_reward, delta, top, epochs in configs:
		print(f'Training ARS: healthy_reward = {healthy_reward}, delta = {delta}, top = {top}, epoch count = {epochs}')
		file_name = f'{out_dir}/ars_d{delta}_t{top}_h{healthy_reward}_{epochs // 1_000_000}M.zip'

		start_time = datetime.now()
		train_ars(
			healthy_reward = 1.0, n_delta = delta, n_top = top, n_epochs = epochs, out_file = file_name, seed = 2137
		)

		training_time = datetime.now() - start_time
		print(f'Training time: {training_time.seconds}')
