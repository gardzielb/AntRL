import math

import gymnasium as gym

if __name__ == '__main__':
	env = gym.make('Ant-v4', healthy_z_range = (0.2, 1.5), render_mode = 'human')
	observation, info = env.reset(seed = 42)

	for i in range(1000):
		action = env.action_space.sample()
		observation, reward, terminated, truncated, info = env.step(action)
		print(info['x_position'], info['y_position'], observation[0])
		env.render()

		if terminated:
			print('Terminated')
			observation, info = env.reset()

		if truncated:
			print('Truncated')
			observation, info = env.reset()

		if math.fabs(info['x_position']) > 3 or math.fabs(info['y_position']) > 3:
			observation, info = env.reset()

	env.close()
