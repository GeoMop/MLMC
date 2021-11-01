import numpy as np
import seaborn
from mlmc.plot.plots import _show_and_save
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt


class ViolinPlotter(seaborn.categorical._ViolinPlotter):
    def draw_quartiles(self, ax, data, support, density, center, split=False):
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        mean = np.mean(data)

        self.draw_to_density(ax, center, mean, support, density, split,
                             linewidth=self.linewidth)

        self.draw_to_density(ax, center, q25, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 1.5] * 2)
        self.draw_to_density(ax, center, q50, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 3] * 2)
        self.draw_to_density(ax, center, q75, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 1.5] * 2)


def violinplot(
    *,
    x=None, y=None,
    hue=None, data=None,
    order=None, hue_order=None,
    bw="scott", cut=2, scale="area", scale_hue=True, gridsize=100,
    width=.8, inner="box", split=False, dodge=True, orient=None,
    linewidth=None, color=None, palette=None, saturation=.75,
    ax=None, **kwargs,):

    plotter = ViolinPlotter(x, y, hue, data, order, hue_order,
                             bw, cut, scale, scale_hue, gridsize,
                             width, inner, split, dodge, orient, linewidth,
                             color, palette, saturation)

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)
    return ax


def fine_coarse_violinplot(data_frame):
    fig, axes = plt.subplots(1, 1, figsize=(22, 10))

    # mean with confidence interval
    # sns.pointplot(x='level', y='samples', hue='type', data=data_frame, estimator=np.mean,
    #               palette="Set2", join=False, ax=axes)

    # line is not suitable for our purpose
    # sns.lineplot(x="level", y="samples", hue="type",# err_style="band", ci='sd'
    #              estimator=np.median, data=data_frame, ax=axes)

    violinplot(x="level", y="samples", hue='type', data=data_frame, palette="Set2",
                           split=True, scale="area", inner="quartile", ax=axes)

    axes.set_yscale('log')
    axes.set_ylabel('')
    axes.set_xlabel('')
    axes.legend([], [], frameon=False)

    _show_and_save(fig, "violinplot", "violinplot")
    _show_and_save(fig, None, "violinplot")

