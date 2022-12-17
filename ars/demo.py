import gymnasium as gym
import gymnasium
import sys

sys.modules["gym"] = gymnasium
from sb3_contrib import ARS
from stable_baselines3.common.evaluation import evaluate_policy

# ant_3: ctrl_cost_weight=0.25, healthy_reward=0.5, use_contact_forces=True

if __name__ == '__main__':
	env = gym.make('Ant-v4', render_mode = 'human')
	observation, info = env.reset(seed = 42)

	model = ARS.load("results/ARS/ant_ars_5M_h0.85", env = env, device = 'cuda', verbose = 1)
	obs, _ = env.reset()

	for i in range(1000):
		action, _states = model.predict(obs)
		obs, rewards, dones, info, _ = env.step(action)
		env.render()

	env.close()
