import argparse

import gymnasium as gym
import gymnasium
import sys

sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC
from sb3_contrib.ars import ARS
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('alg_path', type = str)
	arg_parser.add_argument('-a', '--algorithm', type = str)
	arg_parser.add_argument('-n', '--n-episodes', type = int, default = 1)
	arg_parser.add_argument('-r', '--render', action = 'store_true')
	args = arg_parser.parse_args()

	alg_class = SAC if args.algorithm == 'sac' else ARS
	env = gym.make('Ant-v4', render_mode = 'human')
	algorithm = alg_class.load(args.alg_path, env = env, device = 'cuda', verbose = 1)

	mean, std = evaluate_policy(algorithm, env = env, n_eval_episodes = args.n_episodes, render = args.render)
	print(f'Mean reward = {mean}, std = {std}')
