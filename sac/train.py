import gymnasium as gym
import gymnasium
import sys

sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC

if __name__ == '__main__':
	env = gym.make('Ant-v4', healthy_reward = 0.01)
	observation, info = env.reset(seed = 42)

	model = SAC("MlpPolicy", env, learning_starts = 10_000)
	model.learn(total_timesteps = 200_000, progress_bar = True)
	model.save("results/SAC/ant_sac_1.zip")
