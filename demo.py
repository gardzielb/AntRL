import gymnasium as gym
import gymnasium
import sys

sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# ant_3: ctrl_cost_weight=0.25, healthy_reward=0.5, use_contact_forces=True

if __name__ == '__main__':
	env = gym.make('Ant-v4')
	observation, info = env.reset(seed = 42)

	model = SAC.load("ant_big", env=env)
	obs, _ = env.reset()

	mean, std = evaluate_policy(model, env = model.get_env(), n_eval_episodes = 10, render = False)
	print(mean)
	print(std)

	# for i in range(1000):
	# 	action, _states = model.predict(obs)
	# 	obs, rewards, dones, info, _ = env.step(action)
	# 	env.render()

	env.close()
