import numpy as np
import scipy.stats as st
from scipy import interpolate
import matplotlib.pyplot as plt

def create_color_bar(range, label, ax = None):
    """
    Create colorbar for a variable with given range and add it to given axes.
    :param range: single value as high bound or tuple (low bound, high bound)
    :param label: Label of the colorbar.
    :param ax:
    :return: Function to map values to colors. (normalize + cmap)
    """
    # Create colorbar
    colormap = plt.cm.gist_ncar
    try:
        min_r, max_r = range
    except TypeError:
        min_r, max_r = 0, range
    normalize = plt.Normalize(vmin=min_r, vmax=max_r)
    scalar_mappable = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)
    if type(max_r) is int:
        cb_values = np.arange(min_r, max_r)
        #ticks = np.linspace(min_r, int(size / 10) * 10, 9)
    else:
        cb_values = np.linspace(min_r, max_r, 100)
        #ticks = np.linspace(min_r, int(size / 10) * 10, 9)
    ticks = None
    scalar_mappable.set_array(cb_values)
    clb = plt.colorbar(scalar_mappable, ticks=ticks, aspect=50, pad=0.01, ax=ax)
    clb.set_label(label)
    return lambda v: colormap(normalize(v))

def moments_subset(n_moments, moments=None):
    """
    Return subset of range(n_moments) for ploting.
    :param n_moments: Actual number of moments.
    :param moments: Type of subset:
        None - all moments
        int - size of subset, formed by geometrical sequence
    :return:
    """
    if moments is None:
        subset =  np.arange(1, n_moments)
    else:
        assert type(moments) is int
        subset =  np.round(np.geomspace(1, n_moments-1, moments)).astype(int)
        # make indices unique by increasing
        for i in range(1,len(subset)):
            subset[i] = max(subset[i], subset[i-1]+1)
    return subset


def _show_and_save(fig, file, title):
    """
    Internal function to show the figure and/or save it into a file.
    """
    if file is None:
        fig.show()
    else:
        if file == "":
            file = title
        if file[-3:] != "pdf":
            file = file + ".pdf"
        fig.savefig(file)


def make_monotone(X, Y):

    sX, iX = np.unique(X, return_index=True)
    sY = np.array(Y)[iX]
    return sX, sY


class Distribution:
    """
    mlmc.plot.Distribution

    Class for plotting distribution approximation: PDF and CDF (optional)
    Provides methods to: add more plots, add exact PDF, add ECDF/histogram from single level MC
    """
    def __init__(self, exact_distr=None, title="", quantity_name="X", legend_title="",
                 log_density=False, cdf_plot=True, log_x=False, error_plot='l2'):
        """
        Plot configuration
        :param exact_distr:  Optional exact domain (for adding to plot and computing error)
        :param title: Figure title.
        :param quantity_name: Quantity for X axis label.
        :param log_density: Plot logarithm of density value.
        :param cdf_plot: Plot CDF as well (default)
        :param log_x: Use logarithmic scale for X axis.
        :param error_plot: None, 'diff', 'kl. Plot error of pdf using either difference or
        integrand of KL divergence: exact_pdf * log(exact_pdf / approx_pdf).
        Simple difference is used for CDF for both options.
        """
        self._exact_distr = exact_distr
        self._log_density = log_density
        self._log_x = log_x
        self._error_plot = error_plot
        self._domain = None
        self._title = title
        self._legend_title = legend_title
        self.plot_matrix = []
        self.i_plot = 0

        if cdf_plot:
            self.fig, axes = plt.subplots(1, 2, figsize=(22, 10))
            self.fig_cdf = None
            self.ax_pdf = axes[0]
            self.ax_cdf = axes[1]
        else:
            self.fig, self.ax_pdf = plt.subplots(1, 1, figsize=(12, 10))
            self.fig_cdf, self.ax_cdf = plt.subplots(1, 1, figsize=(12, 10))

        self.fig.suptitle(title)
        x_axis_label = quantity_name

        # PDF axes
        self.ax_pdf.set_title("PDF approximations")
        self.ax_pdf.set_ylabel("probability density")
        self.ax_pdf.set_xlabel(x_axis_label)
        if self._log_x:
            self.ax_pdf.set_xscale('log')
            x_axis_label = "log " + x_axis_label
        if self._log_density:
            self.ax_pdf.set_yscale('log')

        # CDF axes
        self.ax_cdf.set_title("CDF approximations")
        self.ax_cdf.set_ylabel("probability")
        self.ax_cdf.set_xlabel(x_axis_label)
        if self._log_x:
            self.ax_cdf.set_xscale('log')

        if error_plot:
            self.ax_pdf_err = self.ax_pdf.twinx()
            self.ax_pdf.set_zorder(10)
            self.ax_pdf.patch.set_visible(False)

            pdf_err_title = "error - dashed"
            if error_plot == 'kl':
                pdf_err_title = "KL-error - dashed"
            self.ax_pdf_err.set_ylabel(pdf_err_title)
            self.ax_cdf_err = self.ax_cdf.twinx()
            self.ax_cdf.set_zorder(10)
            self.ax_cdf.patch.set_visible(False)

            self.ax_cdf_err.set_ylabel("error - dashed")
            self.ax_pdf_err.set_yscale('log')
            self.ax_cdf_err.set_yscale('log')

    def add_raw_samples(self, samples):
        """
        Add histogram and ecdf for raw samples.
        :param samples:
        """
        # Histogram
        domain = (np.min(samples), np.max(samples))
        self.adjust_domain(domain)
        N = len(samples)
        bins = self._grid(0.5 * np.sqrt(N))
        self.ax_pdf.hist(samples, density=True, bins=bins, alpha=0.3, label='samples', color='red')

        # Ecdf
        X = np.sort(samples)
        Y = (np.arange(len(X)) + 0.5) / float(len(X))
        X, Y = make_monotone(X, Y)
        self.ax_cdf.plot(X, Y, 'red')

        # PDF approx as derivative of Bspline CDF approx
        size_8 = int(N / 8)
        w = np.ones_like(X)
        w[:size_8] = 1 / (Y[:size_8])
        w[N - size_8:] = 1 / (1 - Y[N - size_8:])
        spl = interpolate.UnivariateSpline(X, Y, w, k=3, s=1)
        sX = np.linspace(domain[0], domain[1], 1000)
        self.ax_pdf.plot(sX, spl.derivative()(sX), color='red', alpha=0.4)


    def add_distribution(self, distr_object, label=None):
        """
        Add plot for distribution 'distr_object' with given label.
        :param distr_object: Instance of Distribution, we use methods: density, cdf and attribute domain
        :param label: string label for legend
        :return:
        """
        if label is None:
            label = "size {}".format(distr_object.moments_fn.size)
        domain = distr_object.domain
        self.adjust_domain(domain)
        d_size = domain[1] - domain[0]
        slack = 0  # 0.05
        extended_domain = (domain[0] - slack * d_size, domain[1] + slack * d_size)
        X = self._grid(1000, domain=domain)
        color = 'C{}'.format(self.i_plot)

        plots = []
        Y_pdf = distr_object.density(X)
        self.ax_pdf.plot(X, Y_pdf, label=label, color=color)
        self._plot_borders(self.ax_pdf, color, domain)

        Y_cdf = distr_object.cdf(X)
        self.ax_cdf.plot(X, Y_cdf)
        self._plot_borders(self.ax_cdf, color, domain)

        if self._error_plot and self._exact_distr is not None:
            if self._error_plot == 'kl':
                exact_pdf = self._exact_distr.pdf(X)
                eY_pdf = exact_pdf * np.log(exact_pdf / Y_pdf) - exact_pdf + Y_pdf
                #eY_pdf = exact_pdf / Y_pdf #* np.log(exact_pdf / Y_pdf) / Y_pdf
            else:
                eY_pdf = Y_pdf - self._exact_distr.pdf(X)
            self.ax_pdf_err.plot(X, eY_pdf, linestyle="--", color=color, linewidth=0.5)
            eY_cdf = Y_cdf - self._exact_distr.cdf(X)
            self.ax_cdf_err.plot(X, eY_cdf, linestyle="--", color=color, linewidth=0.5)

        self.i_plot += 1

    def show(self, file=""):
        """
        Set colors according to the number of added plots.
        Set domain from all plots.
        Plot exact distribution.
        show, possibly save to file.
        :param file: None, or filename, default name is same as plot title.
        """
        self._add_exact_distr()
        self.ax_pdf.legend(title=self._legend_title, loc = 1)
        _show_and_save(self.fig, file, self._title)

    def reset(self):
        plt.close()
        self._domain = None

    def _plot_borders(self, ax, color, domain=None):
        """
        Add vertical lines to the plot for endpoints of the 'domain'.
        :return: Pair of line objects.
        """
        if domain is None:
            domain = self._domain
        l1 = ax.axvline(x=domain[0], ymin=0, ymax=0.1, color=color)
        l2 = ax.axvline(x=domain[1], ymin=0, ymax=0.1, color=color)
        return [l1, l2]

    def adjust_domain(self, domain):
        """
        Enlarge common domain by given bounds.
        :param value: [lower_bound, upper_bound]
        """
        if self._domain is None:
            self._domain = domain
        else:
            self._domain = [min(self._domain[0], domain[0]), max(self._domain[1], domain[1])]

    def _add_exact_distr(self):
        """
        Plot exact PDF and CDF.
        :return:
        """
        if self._exact_distr is None:
            return

        # with np.printoptions(precision=2):
        #    lab = str(np.array(self._domain))
        X = self._grid(1000)
        Y = self._exact_distr.pdf(X)
        self.ax_pdf.plot(X, Y, c='black', label="exact")

        Y = self._exact_distr.cdf(X)
        self.ax_cdf.plot(X, Y, c='black')

    def _grid(self, size, domain=None):
        """
        X values grid for given domain. Optionally use the log scale.
        """
        if domain is None:
            domain = self._domain
        if self._log_x:
            X = np.geomspace(domain[0], domain[1], size)
        else:
            X = np.linspace(domain[0], domain[1], size)
        return X




class Eigenvalues:
    """
    Plot of eigenvalues (of the covariance matrix), several sets of eigenvalues can be added
    together with error bars and cut-tresholds.
    Colors are chosen automatically. Slight X shift is used to avoid point overlapping.
    For log Y scale only positive values are plotted.
    """
    def __init__(self, log_y=True, title="Eigenvalues"):
        self._ylim = None
        self.log_y = log_y
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.suptitle(title)
        self.i_plot = 0
        self.title = title
        # index of eignevalues dataset
        if self.log_y:
            self.ax.set_yscale('log')

    def add_values(self, values, errors=None, threshold=None, label=""):
        """
        Add set of eigenvalues into the plot.
        :param values: array (n,); eigen values in increasing or decreasing ordred, automatically flipped to decreasing.
        :param errors: array (n,); corresponding std errors
        :param threshold: horizontal line marking noise level or cut-off eigen value
        :return:
        """
        assert not errors or len(values) == len(errors)
        if values[0] < values[-1]:
            values = np.flip(values)
            if errors is not None:
                errors = np.flip(errors)
            threshold = len(values) - 1 - threshold

        if self.log_y:
            # plot only positive values
            i_last_positive = len(values) - np.argmax(np.flip(values) > 0)
            values = values[:i_last_positive + 1]
            a, b = np.min(values), np.max(values)
            self.adjust_ylim( (a / ((b/a)**0.05), b * (b/a)**0.05) )
        else:
            a, b = np.min(values), np.max(values)
            self.adjust_ylim( (a - 0.05 * (b - a), b + 0.05 * (b - a)) )

        color = 'C{}'.format(self.i_plot)
        X = np.arange(len(values)) + self.i_plot * 0.1
        if errors is None:
            self.ax.scatter(X, values, label=label, color=color)
        else:
            self.ax.errorbar(X, values, yerr=errors, fmt='o', color=color, ecolor=color, capthick=2, label=label)
        if threshold is not None:
            self.ax.axhline(y=threshold, color=color)
        self.i_plot += 1

    def add_linear_fit(self, values):
        pass

    def show(self, file=""):
        """
        Show the plot or save to file.
        :param file: filename base, None for show.
        :return:
        """
        self.ax.legend(title="Noise level")
        _show_and_save(self.fig, file, self.title)

    def adjust_ylim(self, ylim):
        """
        Enlarge common domain by given bounds.
        :param value: [lower_bound, upper_bound]
        """
        if self._ylim is None:
            self._ylim = ylim
        else:
            self._ylim = [min(self._ylim[0], ylim[0]), max(self._ylim[1], ylim[1])]


def moments(moments_fn, size=None, title="", file=""):
    """
    Plot moment functions.
    :param moments_fn:
    :param size:
    :param title:
    :param file:
    :return:
    """
    if size == None:
        size = max(moments_fn.size, 21)
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(title)
    ax = fig.add_subplot(1, 1, 1)
    cmap = create_color_bar(size, 'moments', ax)
    n_pt = 1000
    X = np.linspace(moments_fn.domain[0], moments_fn.domain[1], n_pt)
    Y = moments_fn._eval_all(X, size=size)
    central_band = Y[int(n_pt*0.1):int(n_pt*0.9), :]
    ax.set_ylim((np.min(central_band), np.max(central_band)))
    for m, y in enumerate(Y.T):
        color = cmap(m)
        ax.plot(X, y, color=color, linewidth=0.5)
    _show_and_save(fig, file, title)





class VarianceBreakdown:
    """
    Plot total variance average over moments and variances of individual moments,
    Brake down to contribution of individual levels and optionally comparison to the reference level variances using
    error bars for the (signed) difference: ref_level_vars - level_vars
    """
    def __init__(self, moments=None):
        """
        :param moments: Size or type of moments subset, see moments_subset function.
        """
        self.fig =  plt.figure(figsize=(15, 8))
        self.title = "Variance brakedown"
        self.fig.suptitle(self.title)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.X_list = []
        self.X_labels = []
        self.x_shift = 0
        self.n_moments = None
        self.subset_type = moments

    def add_variances(self, level_vars, n_samples, ref_level_vars=None):
        """
        Add plot of variances for single MLMC instance.

        :param level_vars: Array (n_levels, n_moments) of level variances.
        :param n_samples: Array (n_levels,) of numberf of samples on levels
        :param ref_level_vars: reference level vars (e.g. from bootstrapping)
        :return:
        """
        n_levels, n_moments = level_vars.shape
        assert n_samples.shape == (n_levels, )

        if self.n_moments is None:
            self.n_moments = n_moments
            self.moments_subset = moments_subset(n_moments, self.subset_type)
        else:
            assert self.n_moments == n_moments

        level_vars = level_vars[:, self.moments_subset]
        n_levels, n_moments = level_vars.shape

        width=0.1
        X = self.x_shift + (width*1.1)*np.arange(n_moments)
        self.x_shift = X[-1] + 3*width
        self.X_list.append(X)
        X_labels = ['avg'] + [str(self.moments_subset[m]) for m in range(1, n_moments)]
        self.X_labels.extend(X_labels)
        #plots = []
        sum_Y = np.zeros(n_moments)
        yerr = None
        total_Y0 = np.sum(np.mean(level_vars[:, :] / n_samples[:, None], axis=1))
        for il in reversed(range(n_levels)):
            vars = level_vars[il, 1:]
            Y = np.concatenate(( [np.mean(vars)], vars))
            Y /= n_samples[il]

            if ref_level_vars is not None:
                rvars = ref_level_vars[il, 1:]
                rY = np.concatenate(([np.mean(rvars)], rvars))
                rY /= n_samples[il]
                yerr = np.array([(-diff, 0) if diff < 0 else (0, diff) for diff in rY - Y]).T
            level_col = plt.cm.tab20(il)
            self.ax.bar(X, Y, width, bottom=sum_Y, yerr=yerr,
                        color=level_col)
            level_label = "L{} {:5}".format(il, n_samples[il])
            aX = X[0] - width/2
            aY = Y[0]/2 + sum_Y[0]
            tY = total_Y0 * ((n_levels - 0.5 - il) / n_levels)
            self.ax.annotate(level_label,
                xy=(aX, aY), xytext=(-50, tY),
                arrowprops=dict(arrowstyle='->', color=level_col),
                textcoords=('offset pixels', 'data'),
                )

            sum_Y += Y


    def show(self, file=""):
        """
        Show the plot or save to file.
        :param filename: filename base, None for show.
        :return:
        """
        self.ax.set_xlabel("moments")
        self.ax.set_xticks(np.concatenate(self.X_list))
        self.ax.set_xticklabels(self.X_labels)
        _show_and_save(self.fig, file, self.title)


class Variance:
    """
    Plot level variances, i.e. Var X^l as a function of the mesh step.
    Selected moments are plotted.
    """
    def __init__(self, moments=None):
        """
        :param moments: Size or type of moments subset, see moments_subset function.
        """
        self.fig = plt.figure(figsize=(15, 8))
        self.title = "Level variances"
        self.fig.suptitle(self.title)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("$h$ - mesh step")
        self.ax.set_ylabel("Var $X^h$")
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

        self.n_moments = None
        self.subset_type = moments
        self.min_step = 1e300
        self.max_step = 0
        self.data = {}


    def add_level_variances(self, steps, variances):
        """
        Add variances for single MLMC instance.
        :param steps, variances : as returned by Estimate.estimate_level_vars
        :param n_levels:
        """
        n_levels, n_moments = variances.shape
        if self.n_moments is None:
            self.n_moments = n_moments
            self.moments_subset = moments_subset(n_moments, self.subset_type)
        else:
            assert self.n_moments == n_moments

        variances = variances[:, self.moments_subset]
        self.min_step = min(self.min_step, steps[-1])
        self.max_step = max(self.max_step, steps[0])
        for m, vars in enumerate(variances.T):
            X, Y = self.data.get(m, ([], []))
            X.extend(steps.tolist())
            Y.extend(vars.tolist())
            self.data[m] = (X, Y)




    # def add_diff_variances(self, step, variances):
    #     pass

    def show(self, file=""):
        step_range = self.max_step / self.min_step
        log_scale = step_range ** 0.001 - 1
        rv = st.lognorm(scale=1, s=log_scale)
        for m, (X, Y) in self.data.items():
            col = plt.cm.tab20(m)
            label = "M{}".format(self.moments_subset[m])
            XX = np.array(X) * rv.rvs(size=len(X))
            self.ax.scatter(XX, Y, color=col, label=label)
            #f = interpolate.interp1d(X, Y, kind='cubic')

            XX, YY = make_monotone(X, Y)

            #f = interpolate.PchipInterpolator(XX[1:], YY[1:])
            m = len(XX)-1
            spl = interpolate.splrep(XX[1:], YY[1:], k=3, s=m - np.sqrt(2*m))
            xf = np.geomspace(self.min_step, self.max_step, 100)
            yf = interpolate.splev(xf, spl)
            self.ax.plot(xf, yf, color=col)
        self.fig.legend()
        _show_and_save(self.fig, file, self.title)










class Aux:
    def _scatter_level_moment_data(self, ax, values, i_moments=None, marker='o'):
        """
        Scatter plot of given table of data for moments and levels.
        X coordinate is given by level, and slight shift is applied to distinguish the moments.
        Moments are colored using self._moments_cmap.
        :param ax: Axis where to add the scatter.
        :param values: data to plot, array n_levels x len(i_moments)
        :param i_moments: Indices of moments to use, all moments grater then 0 are used.
        :param marker: Scatter marker to use.
        :return:
        """
        cmap = self._moments_cmap
        if i_moments is None:
            i_moments = range(1, self.n_moments)
        values = values[:, i_moments[:]]
        n_levels = values.shape[0]
        n_moments = values.shape[1]

        moments_x_step = 0.5/n_moments
        for m in range(n_moments):
            color = cmap(i_moments[m])
            X = np.arange(n_levels) + moments_x_step * m
            Y = values[:, m]
            col = np.ones(n_levels)[:, None] * np.array(color)[None, :]
            ax.scatter(X, Y, c=col, marker=marker, label="var, m=" + str(i_moments[m]))

    def plot_bootstrap_variance_compare(self):
        """
        Plot fraction (MLMC var est) / (BS var set) for the total variance and level variances.
        :param moments_fn:
        :return:
        """
        moments_fn = self.moments
        mean, var, l_mean, l_var = self._bs_get_estimates(moments_fn)
        l_var = l_var / self.n_samples[: , None]
        est_variances = np.concatenate((var[None, 1:], l_var[:, 1:]), axis=0)

        bs_var = self._bs_mean_variance
        bs_l_var = self._bs_level_mean_variance / self.n_samples[:, None]
        bs_variances = np.concatenate((bs_var[None, 1:], bs_l_var[:, 1:]), axis=0)

        fraction = est_variances / bs_variances

        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1)

        #self._scatter_level_moment_data(ax, bs_variances, marker='.')
        #self._scatter_level_moment_data(ax, est_variances, marker='d')
        self._scatter_level_moment_data(ax, fraction, marker='o')

        #ax.legend(loc=6)
        lbls = ['Total'] + [ 'L{:2d}'.format(l+1) for l in range(self.n_levels)]
        ax.set_xticks(ticks = np.arange(self.n_levels + 1))
        ax.set_xticklabels(lbls)
        ax.set_yscale('log')
        ax.set_ylim((0.3, 3))

        self.color_bar(moments_fn.size, 'moments')

        fig.savefig('bs_var_vs_var.pdf')
        plt.show()

    def plot_bs_variances(self, variances, y_label=None, log=True, y_lim=None):
        """
        Plot BS estimate of error of variances of other related quantities.
        :param variances: Data, shape: (n_levels + 1, n_moments).
        :return:
        """
        if y_lim is None:
            y_lim = (np.min(variances[:, 1:]), np.max(variances[:, 1:]))
        if y_label is None:
            y_label = "Error of variance estimates"

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        self.set_moments_color_bar(ax)
        self._scatter_level_moment_data(ax, variances, marker='.')

        lbls = ['Total'] + ['L{:2d}\n{}\n{}'.format(l + 1, nsbs, ns)
                            for l, (nsbs, ns) in enumerate(zip(self._bs_n_samples, self.n_samples))]
        ax.set_xticks(ticks = np.arange(self.n_levels + 1))
        ax.set_xticklabels(lbls)
        if log:
            ax.set_yscale('log')
        ax.set_ylim(y_lim)
        ax.set_ylabel(y_label)

        fig.savefig('bs_var_var.pdf')
        plt.show()

    def plot_bs_var_error_contributions(self):
        """
        MSE of total variance and contribution of individual levels.
        """
        bs_var_var = self._bs_var_variance[:]
        bs_l_var_var = self._bs_level_var_variance[:, :]
        bs_l_var_var[:, 1:] /= self._bs_n_samples[:, None]**2

        bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
        self.plot_bs_variances(bs_variances, log=True,
                               y_label="MSE of total variance and contributions from individual levels.",
                               )

    def plot_bs_level_variances_error(self):
        """
        Plot error of estimates of V_l. Scaled as V_l^2 / N_l
        """
        l_var = self._ref_level_var

        l_var_var_scale = l_var[:, 1:] ** 2 * 2 / (self._bs_n_samples[:, None] - 1)
        total_var_var_scale = np.sum(l_var_var_scale[:, :] / self._bs_n_samples[:, None]**2, axis=0 )

        bs_var_var = self._bs_var_variance[:]
        bs_var_var[1:] /= total_var_var_scale

        bs_l_var_var = self._bs_level_var_variance[:, :]
        bs_l_var_var[:, 1:] /= l_var_var_scale

        bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
        self.plot_bs_variances(bs_variances, log=True,
                               y_label="MSE of level variances estimators scaled by $V_l^2/N_l$.")

    def plot_bs_var_log_var(self):
        """
        Test that  MSE of log V_l scales as variance of log chi^2_{N-1}, that is approx. 2 / (n_samples-1).
        """
        #vv = 1/ self.mlmc._variance_of_variance(self._bs_n_samples)
        vv = self._bs_n_samples
        bs_l_var_var = np.sqrt((self._bs_level_var_variance[:, :]) * vv[:, None])
        bs_var_var = self._bs_var_variance[:]  # - np.log(total_var_var_scale)
        bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
        self.plot_bs_variances(bs_variances, log=True,
                               y_label="BS est. of var. of $\hat V^r$, $\hat V^r_l$ estimators.",
                               )#y_lim=(0.1, 20))

    # def plot_bs_var_reg_var(self):
    #     """
    #     Test that  MSE of log V_l scales as variance of log chi^2_{N-1}, that is approx. 2 / (n_samples-1).
    #     """
    #     vv = self.mlmc._variance_of_variance(self._bs_n_samples)
    #     bs_l_var_var = (self._bs_level_var_variance[:, :]) / vv[:, None]
    #     bs_var_var = self._bs_var_variance[:]  # - np.log(total_var_var_scale)
    #     bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
    #     self.plot_bs_variances(bs_variances, log=True,
    #                            y_label="BS est. of var. of $\hat V^r$, $\hat V^r_l$ estimators.",
    #                            y_lim=(0.1, 20))


    def plot_means_and_vars(self, moments_mean, moments_var, n_levels, exact_moments):
        """
        Plot means with variance whiskers to given axes.
        :param moments_mean: array, moments mean
        :param moments_var: array, moments variance
        :param n_levels: array, number of levels
        :param exact_moments: array, moments from distribution
        :param ex_moments: array, moments from distribution samples
        :return:
        """
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(moments_mean) + 1)))
        # print("moments mean ", moments_mean)
        # print("exact momentss ", exact_moments)

        x = np.arange(0, len(moments_mean[0]))
        x = x - 0.3
        default_x = x

        for index, means in enumerate(moments_mean):
            if index == int(len(moments_mean) / 2) and exact_moments is not None:
                plt.plot(default_x, exact_moments, 'ro', label="Exact moments")
            else:
                x = x + (1 / (len(moments_mean) * 1.5))
                plt.errorbar(x, means, yerr=moments_var[index], fmt='o', capsize=3, color=next(colors),
                             label = "%dLMC" % n_levels[index])
        if ex_moments is not None:
                plt.plot(default_x - 0.125, ex_moments, 'ko', label="Exact moments")
        plt.legend()
        #plt.show()
        #exit()








    def plot_var_regression(self, i_moments = None):
        """
        Plot total and level variances and their regression and errors of regression.
        :param i_moments: List of moment indices to plot. If it is an int M, the range(M) is used.
                       If None, self.moments.size is used.
        """
        moments_fn = self.moments

        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 2, 1)
        ax_err = fig.add_subplot(1, 2, 2)

        if i_moments is None:
            i_moments = moments_fn.size
        if type(i_moments) is int:
            i_moments = list(range(i_moments))
        i_moments = np.array(i_moments, dtype=int)

        self.set_moments_color_bar(ax=ax)

        est_diff_vars, n_samples = self.mlmc.estimate_diff_vars(moments_fn)
        reg_diff_vars = self.mlmc.estimate_diff_vars_regression(moments_fn) #/ self.n_samples[:, None]
        ref_diff_vars = self._ref_level_var #/ self.n_samples[:, None]

        self._scatter_level_moment_data(ax,  ref_diff_vars, i_moments, marker='o')
        self._scatter_level_moment_data(ax, est_diff_vars, i_moments, marker='d')
        # add regression curves
        moments_x_step = 0.5 / self.n_moments
        for m in i_moments:
            color = self._moments_cmap(m)
            X = np.arange(self.n_levels) + moments_x_step * m
            Y = reg_diff_vars[1:, m]
            ax.plot(X[1:], Y, c=color)
            ax_err.plot(X[:], reg_diff_vars[:, m]/ref_diff_vars[:,m], c=color)

        ax.set_yscale('log')
        ax.set_ylabel("level variance $V_l$")
        ax.set_xlabel("step h_l")

        ax_err.set_yscale('log')
        ax_err.set_ylabel("regresion var. / reference var.")

        #ax.legend(loc=2)
        fig.savefig('level_vars_regression.pdf')
        plt.show()
