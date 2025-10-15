import numpy as np
import seaborn
from mlmc.plot.plots import _show_and_save
import matplotlib

# Set default font size for all plots
matplotlib.rcParams.update({'font.size': 22})

import matplotlib.pyplot as plt


class ViolinPlotter(seaborn.categorical._ViolinPlotter):
    """
    Custom subclass of seaborn's internal _ViolinPlotter to modify how quartiles
    and mean lines are drawn inside a violin plot.

    This class extends the default behavior by drawing the 25th, 50th, and 75th
    percentiles as dashed lines, and the mean as a solid line across the violin body.
    """

    def draw_quartiles(self, ax, data, support, density, center, split=False):
        """
        Draw quartile and mean lines on the violin plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to draw on.
        data : array-like
            Input data for a single violin.
        support : array-like
            Grid over which the kernel density was evaluated.
        density : array-like
            Corresponding kernel density values.
        center : float
            Position of the violin on the categorical axis.
        split : bool, default=False
            Whether the violin is split by hue (two sides).

        Notes
        -----
        - The mean is drawn as a solid line.
        - Quartiles (25%, 50%, 75%) are drawn as dashed lines.
        - The density scaling follows seaborn’s internal behavior.
        """
        # Compute quartiles and mean of the data
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        mean = np.mean(data)

        # Draw mean line (solid)
        self.draw_to_density(ax, center, mean, support, density, split,
                             linewidth=self.linewidth)

        # Draw quartile lines (dashed)
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
    ax=None, **kwargs,
):
    """
    Wrapper around the custom ViolinPlotter class to generate a violin plot.

    Parameters
    ----------
    x, y, hue : str, optional
        Variable names for the categorical axis, numeric axis, and hue grouping.
    data : DataFrame, optional
        Dataset containing the variables.
    order, hue_order : list, optional
        Order of categories for x and hue variables.
    bw : str or float, default="scott"
        Bandwidth method or scalar for kernel density estimation.
    cut : float, default=2
        How far the violin extends beyond extreme data points.
    scale : {"area", "count", "width"}, default="area"
        Method for scaling the width of each violin.
    scale_hue : bool, default=True
        Whether to scale by hue levels within each category.
    gridsize : int, default=100
        Number of points in the KDE grid.
    width : float, default=0.8
        Width of each violin.
    inner : {"box", "quartile", "point", "stick", None}, default="box"
        Representation inside each violin.
    split : bool, default=False
        Draw half-violins when hue is used.
    dodge : bool, default=True
        Separate violins for each hue level.
    orient : {"v", "h"}, optional
        Plot orientation; inferred if not specified.
    linewidth : float, optional
        Width of the line used for drawing violins and quartiles.
    color : matplotlib color, optional
        Color for all violins.
    palette : str or sequence, optional
        Color palette for hue levels.
    saturation : float, default=0.75
        Saturation for colors.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on; created if None.
    **kwargs :
        Additional arguments passed to seaborn’s internal methods.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the drawn violin plot.
    """
    # Initialize a custom violin plotter instance
    plotter = ViolinPlotter(
        x, y, hue, data, order, hue_order,
        bw, cut, scale, scale_hue, gridsize,
        width, inner, split, dodge, orient, linewidth,
        color, palette, saturation
    )

    # Create a new axes if none provided
    if ax is None:
        ax = plt.gca()

    # Draw the plot using the seaborn-based custom plotter
    plotter.plot(ax)
    return ax


def fine_coarse_violinplot(data_frame):
    """
    Generate a split violin plot comparing fine and coarse simulation samples per level.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        Must contain the columns:
        - 'level' : int, simulation level
        - 'samples' : float, sample values
        - 'type' : str, either 'fine' or 'coarse'

    Notes
    -----
    - Uses log scale on the y-axis.
    - Calls `_show_and_save` to display and save the resulting plot.
    - Produces a split violin plot (fine vs coarse) for each level.
    """
    # Create a single subplot for the violin plot
    fig, axes = plt.subplots(1, 1, figsize=(22, 10))

    # Draw split violin plot for 'fine' and 'coarse' samples per level
    violinplot(
        x="level", y="samples", hue='type', data=data_frame,
        palette="Set2", split=True, scale="area",
        inner="quartile", ax=axes
    )

    # Use logarithmic y-scale (typical for MLMC variance/cost visualizations)
    axes.set_yscale('log')
    axes.set_ylabel('')
    axes.set_xlabel('')

    # Remove legend frame and content
    axes.legend([], [], frameon=False)

    # Display and save plot using utility function
    _show_and_save(fig, "violinplot", "violinplot")
    _show_and_save(fig, None, "violinplot")
