import gymnasium as gym
import gymnasium
import sys

sys.modules["gym"] = gymnasium
from sb3_contrib import ARS

from pathlib import Path

models = [
	'ars_d60_t20_h1_5M',
	'ars_d60_t10_h1_5M',
	'ars_d60_t10_h1_5M',
	'ars_d40_t20_h1_5M',
	'ars_d60_t20_h095_5M',
	'ars_d60_t20_h090_5M'
]
model_paths = [Path(f'_out/{m}.zip') for m in models]

for model_path in model_paths:
	env = gym.make('Ant-v4', render_mode = 'rgb_array_list')
	env = gym.wrappers.RecordVideo(
		env, './_out/videos', episode_trigger = lambda e: True, name_prefix = model_path.stem
	)

	observation, info = env.reset(seed = 19)

	model = ARS.load(model_path, env = env)
	obs, _ = env.reset()

	for i in range(1000):
		action, _states = model.predict(obs)
		obs, rewards, dones, info, _ = env.step(action)
		env.render()

	env.close()
