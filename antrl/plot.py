import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_deaths(df, ax: plt.Axes, color = 'red'):
	sns.set_theme()
	with sns.axes_style("darkgrid"):
		death_x = []
		death_y = []
		death_index = df.apply(pd.Series.last_valid_index)
		death_index = death_index[death_index < 990]
		for i, el in death_index.iteritems():
			death_y.append(df.iloc[el, i])
		ax.plot(death_index, death_y, marker = 'x', color = color, linestyle = '')


def plot_mean_and_std(df, ax: plt.Axes = None, color = 'blue', label = None):
	sns.set_theme()
	with sns.axes_style("darkgrid"):
		df2 = df.fillna(method = 'ffill')
		df2['mean'] = df2.mean(axis = 1)
		df2['std'] = df2['mean'] + df2.std(axis = 1)
		df2['nstd'] = df2['mean'] - df2.std(axis = 1)
		if ax is None:
			_, ax = plt.subplots()
		ax.plot(df2['mean'], color = color, label = label)
		ax.set_xlabel('Liczba iteracji')
		ax.set_ylabel('Średnia wartość nagrody')
		ax.grid(True)
		ax.fill_between(x = df2.index, y1 = 'nstd', y2 = 'std', data = df2, color = mpl.colors.to_rgba(color, 0.1))
	return ax
