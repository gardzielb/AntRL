import gymnasium as gym
import gymnasium
import sys

sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# ant_3: ctrl_cost_weight=0.25, healthy_reward=0.5, use_contact_forces=True

if __name__ == '__main__':
	env = gym.make('Ant-v4', render_mode = 'human')
	observation, info = env.reset(seed = 42)

	model = SAC.load("results/SAC/ant_big.zip", env = env, device = 'cuda', verbose = 1)
	obs, _ = env.reset()

	for i in range(1000):
		action, _states = model.predict(obs)
		obs, rewards, dones, info, _ = env.step(action)
		env.render()

	env.close()
