import argparse

import gymnasium
import sys

import matplotlib.pyplot as plt

sys.modules["gym"] = gymnasium
from stable_baselines3 import SAC
from sb3_contrib.ars import ARS

from antrl.experiments import evaluate_model
from antrl.plot import plot_mean_and_std, plot_deaths
from antrl.ars import train_ars, prepare_ars_models
from antrl.sac import train_sac


def prepare(args):
	if args.algorithm == 'ars':
		prepare_ars_models(args.out_dir)
	elif args.algorithm == 'sac':
		pass
	else:
		print(f'Unrecognized algorithm: {args.algorithm}')
		exit(1)


def train(args):
	if args.algorithm == 'ars':
		train_ars(
			healthy_reward = args.healthy_reward, n_delta = 40, n_top = 20,
			n_epochs = args.n_timestamps, out_file = args.out_path, seed = args.seed
		)
	elif args.algorithm == 'sac':
		train_sac(
			healthy_reward = args.healthy_reward, n_epochs = args.n_timestamps,
			seed = args.seed, out_file = args.out_path
		)
	else:
		print(f'Unrecognized algorithm: {args.algorithm}')
		exit(1)


def evaluate(args):
	algorithm_cls = ARS if args.algorithm == 'ars' else SAC
	result_df = evaluate_model(
		args.alg_path, algorithm_cls, seed = args.seed, n_eval_episodes = args.n_episodes, render = args.render
	)

	_, ax = plt.subplots()
	plot_mean_and_std(result_df, ax, color = 'blue')
	plot_deaths(result_df, ax, color = 'blue')
	plt.show()


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-a', '--algorithm', type = str, choices = ['ars', 'sac'])
	arg_parser.add_argument('-s', '--seed', type = int, default = 2137)
	sub_parsers = arg_parser.add_subparsers()

	eval_parser = sub_parsers.add_parser('eval')
	eval_parser.add_argument('alg_path', type = str)
	eval_parser.add_argument('-n', '--n-episodes', type = int, default = 1)
	eval_parser.add_argument('-r', '--render', action = 'store_true')
	eval_parser.set_defaults(func = evaluate)

	train_parser = sub_parsers.add_parser('train')
	train_parser.add_argument('-o', '--out-path', type = str)
	train_parser.add_argument('-n', '--n-timestamps', type = int)
	train_parser.add_argument('-r', '--healthy-reward', type = float)
	train_parser.set_defaults(func = train)

	prepare_parser = sub_parsers.add_parser('prepare')
	prepare_parser.add_argument('-o', '--out-dir', type = str)
	prepare_parser.set_defaults(func = prepare)

	prog_args = arg_parser.parse_args()
	prog_args.func(prog_args)
