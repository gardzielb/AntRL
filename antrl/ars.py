import gymnasium as gym
import gymnasium
import sys

sys.modules["gym"] = gymnasium
from sb3_contrib import ARS


def train_ars(healthy_reward: float):
	env = gym.make('Ant-v4', healthy_reward = healthy_reward)
	env.reset(seed = 2137)

	model = ARS(
		'LinearPolicy', env,
		alive_bonus_offset = -1,
		delta_std = 0.025,
		learning_rate = 0.015,
		n_delta = 60,
		n_top = 20,
		device = 'cuda'
	)

	model.learn(total_timesteps = 5_000_000, progress_bar = True)
	model.save(f'results/ARS/ant_ars_5M_h{healthy_reward}.zip')