import numpy as np
import scipy.stats as st
from scipy import interpolate
import matplotlib as mpl
# font = {'family': 'normal',
#         'weight': 'bold',
#         'size': 22}
#
# matplotlib.rc('font', **font)

mpl.use("pgf")
pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        ##r"\usepackage{cmbright}",
    ],
}
mpl.rcParams.update(pgf_with_pdflatex)
# mpl.rcParams['xtick.labelsize']=12
# mpl.rcParams['ytick.labelsize']=12

#matplotlib.rcParams.update({'font.size': 22})


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator

def create_color_bar(range, label, ax=None):
    """
    Create colorbar for a variable with given range and add it to given axes.
    :param range: single value as high bound or tuple (low bound, high bound)
    :param label: Label of the colorbar.
    :param ax:
    :return: Function to map values to colors. (normalize + cmap)
    """
    # Create colorbar
    colormap = plt.cm.bone#plt.cm.gist_ncar
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


class SimpleDistribution:
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

        self.colormap = plt.cm.tab20

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
        #self.ax_pdf.set_title("PDF approximations")
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

    def add_distribution(self, X, Y_pdf, Y_cdf, domain, label=None):
        """
        Add plot for distribution 'distr_object' with given label.
        :param distr_object: Instance of Distribution, we use methods: density, cdf and attribute domain
        :param label: string label for legend
        :return:
        """
        # if label is None:
        #     label = "size {}".format(distr_object.moments_fn.size)
        #domain = distr_object.domain
        self.adjust_domain(domain)
        d_size = domain[1] - domain[0]
        slack = 0  # 0.05
        extended_domain = (domain[0] - slack * d_size, domain[1] + slack * d_size)
        #X = self._grid(1000, domain=domain)
        color = self.colormap(self.i_plot)

        plots = []
        #Y_pdf = distr_object.density(X)
        self.ax_pdf.plot(X, Y_pdf, label=label, color=color)
        self._plot_borders(self.ax_pdf, color, domain)

        #Y_cdf = distr_object.cdf(X)
        self.ax_cdf.plot(X, Y_cdf, color=color)
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
        #self._add_exact_distr()
        self.ax_pdf.legend(title=self._legend_title, loc=1)

        #_show_and_save(self.fig_kl, file, self._title)

        self.fig.show()
        file = self._title
        if file[-3:] != "pdf":
            file = file + ".pdf"
        self.fig.savefig(file)

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

    def _add_exact_distr(self, X, Y_pdf, Y_cdf):
        """
        Plot exact PDF and CDF.
        :return:
        """

        self.ax_pdf.set_ylim([np.min(Y_pdf) - (np.max(Y_pdf) - np.min(Y_pdf))*0.1, np.max(Y_pdf) + (np.max(Y_pdf) - np.min(Y_pdf))*0.1])
        self.ax_cdf.set_ylim([np.min(Y_cdf) - (np.max(Y_cdf) - np.min(Y_cdf)) * 0.1, np.max(Y_cdf) + (np.max(Y_cdf) - np.min(Y_cdf)) * 0.1])

        self.ax_pdf.plot(X, Y_pdf, c='black', label="exact")
        self.ax_cdf.plot(X, Y_cdf, c='black')

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


class Distribution:
    """
    mlmc.plot.Distribution

    Class for plotting distribution approximation: PDF and CDF (optional)
    Provides methods to: add more plots, add exact PDF, add ECDF/histogram from single level MC
    """
    def __init__(self, exact_distr=None, title="", quantity_name="X", legend_title="",
                 log_density=False, cdf_plot=True, log_x=False, error_plot='l2', reg_plot=False, multipliers_plot=True):
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

        self.ax_cdf = None
        self.ax_log_density = None
        self.ax_mult_mom_der = None
        self.ax_mult_mom_der_2 = None

        self._reg_param = 0

        self.colormap = plt.cm.tab20

        self.reg_plot = reg_plot

        if cdf_plot:
            self.fig, axes = plt.subplots(1, 2, figsize=(22, 10))
            self.fig_cdf = None
            self.ax_pdf = axes[0]
            self.ax_cdf = axes[1]
        else:
            if multipliers_plot:
                self.fig, axes = plt.subplots(2, 2, figsize=(22, 14))

                self.ax_pdf = axes[0, 0]
                self.ax_log_density = axes[0, 1]
                self.ax_mult_mom_der = axes[1, 0]
                self.ax_mult_mom_der_2 = axes[1, 1]

                self.ax_log_density.set_title(r'$\ln(\rho)$')
                self.ax_mult_mom_der.set_title(r'$\lambda \phi^\prime $')
                self.ax_mult_mom_der_2.set_title(r'$\lambda \phi^{\prime \prime} $')

                #self.ax_log_density, self.ax_mult_mom_der, self.ax_mult_mom_der_2]
                # print("self ax pdf ", self.ax_pdf)
                # print("self ax pdf type ", type(self.ax_pdf))
                # self.fig_ax_mult_mom, self.ax_log_density = plt.subplots(2, 2, figsize=(12, 10))
                #
                # print("self.ax_log_density ", self.ax_log_density)
                # print("self.ax_log_density ", type(self.ax_log_density))
                # exit()
                #
                # self.fig_ax_mult_mom_der, self.ax_mult_mom_der = plt.subplots(1, 3, figsize=(12, 10))
                # self.fig_ax_mult_mom_2, self.ax_mult_mom_2 = plt.subplots(1, 4, figsize=(12, 10))

        # if reg_plot:
        #     self.fig, axes = plt.subplots(1, 3, figsize=(22, 10))
        #     self.fig_reg_term = None
        #     self.ax_pdf = axes[0]
        #     self.ax_reg_term = axes[2]
        # else:
        #     self.fig, self.ax_pdf = plt.subplots(1, 1, figsize=(12, 10))
        #     self.fig_reg_term, self.ax_reg_term = plt.subplots(1, 1, figsize=(12, 10))

        self.fig.suptitle(title, y=0.99)
        x_axis_label = quantity_name

        # PDF axes
        self.ax_pdf.set_title(r'$\rho$')
        #self.ax_pdf.set_ylabel("probability density")
        self.ax_pdf.set_xlabel(x_axis_label)
        if self._log_x:
            self.ax_pdf.set_xscale('log')
            x_axis_label = "log " + x_axis_label
        # if self._log_density:
        #     self.ax_pdf.set_yscale('log')

        if cdf_plot:
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
            self.ax_pdf_err.set_yscale('log')

            if cdf_plot:
                self.ax_cdf_err = self.ax_cdf.twinx()
                self.ax_cdf.set_zorder(10)
                self.ax_cdf.patch.set_visible(False)
                self.ax_cdf_err.set_ylabel("error - dashed")
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
        print("N samples ", N)
        # bins = self._grid(0.5 * np.sqrt(N))
        # self.ax_pdf.hist(samples, density=True, bins=bins, alpha=0.3, label='samples', color='red')

        # Ecdf
        X = np.sort(samples)
        Y = (np.arange(len(X)) + 0.5) / float(len(X))
        X, Y = make_monotone(X, Y)
        if self.ax_cdf is not None:
            self.ax_cdf.plot(X, Y, 'red', label="ecdf")

        # PDF approx as derivative of Bspline CDF approx
        size_8 = int(N / 8)
        w = np.ones_like(X)
        w[:size_8] = 1 / (Y[:size_8])
        w[N - size_8:] = 1 / (1 - Y[N - size_8:])
        spl = interpolate.UnivariateSpline(X, Y, w, k=3, s=1)
        sX = np.linspace(domain[0], domain[1], 1000)
        # if self._reg_param == 0:
        #     self.ax_pdf.plot(sX, spl.derivative()(sX), color='red', alpha=0.4, label="derivative of Bspline CDF")

    def add_distribution(self, distr_object, label=None, size=0, mom_indices=None, reg_param=0):
        """
        Add plot for distribution 'distr_object' with given label.
        :param distr_object: Instance of Distribution, we use methods: density, cdf and attribute domain
        :param label: string label for legend
        :return:
        """
        self._reg_param = reg_param

        if label is None:
            label = "size {}".format(distr_object.moments_fn.size)
        domain = distr_object.domain
        self.adjust_domain(domain)
        d_size = domain[1] - domain[0]
        slack = 0  # 0.05
        extended_domain = (domain[0] - slack * d_size, domain[1] + slack * d_size)
        X = self._grid(1000, domain=domain)
        color = self.colormap(self.i_plot)#'C{}'.format(self.i_plot)

        line_styles = ['-', ':', '-.', '--']
        plots = []

        Y_pdf = distr_object.density(X)
        self.ax_pdf.plot(X, Y_pdf, label=label, color=color)
        #self._plot_borders(self.ax_pdf, color, domain)

        # if self.i_plot >= len(line_styles):
        #     raise Exception("Number of line styles is insufficient")

        if self.ax_log_density is not None:
            if self.i_plot == 0:
                pass
                #self.cmap_1 = create_color_bar(size, 'i-th moment', self.ax_log_density)
                #self.cmap_2 = create_color_bar(size, 'i-th moment', self.ax_mult_mom_der)
                #self.cmap_3 = create_color_bar(size, 'i-th moment', self.ax_mult_mom_der_2)

            #Y = distr_object.mult_mom(X)

            # if mom_indices is not None:
            #     indices = mom_indices
            # else:
            #     indices = range(len(Y))
            #
            # print(indices)

            Y = distr_object.density_log(X)
            self.ax_log_density.plot(X, Y, color=color)
            self._plot_borders(self.ax_log_density, color, domain)

            if self.ax_mult_mom_der is not None:
                Y = distr_object.mult_mom_der(X, degree=1)
                self.ax_mult_mom_der.plot(X, Y, color=color)
                self._plot_borders(self.ax_mult_mom_der, color, domain)

            if self.ax_mult_mom_der_2 is not None:
                Y = distr_object.mult_mom_der(X, degree=2)
                self.ax_mult_mom_der_2.plot(X, Y, color=color)
                self._plot_borders(self.ax_mult_mom_der_2, color, domain)

        #self.ax_pdf.plot(X, distr_object.plot_regularization(X), label="regularization")

        if self.reg_plot is True and distr_object.reg_param != 0:
            X, Y_cdf = reg = distr_object.regularization(X)
            #pdf = distr_object.density(X)

            #print("Y_cdf ", Y_cdf)
            if self.ax_cdf is not None:
                self.ax_cdf.scatter(X, Y_cdf, color=color, label="reg term")

            beta_reg = []
            #for x in X:
            #X, beta_reg = distr_object.beta_regularization(X)

            # Y_cdf = beta_reg
            # print("X", X)
            # print("beta reg ", beta_reg)
            # self.ax_cdf.plot(X, beta_reg, color=color, label="beta reg", linestyle="-")

            # self.i_plot += 1
            # color = 'C{}'.format(self.i_plot)
            # print("reg + beta color ", color)
            # self.ax_cdf.plot(X, reg + beta_reg, color=color, label="reg + beta reg")

            #self.ax_cdf.plot(X, distr_object.multipliers_dot_phi(X), label="\lambda * \phi", color=color)
        else:
            Y_cdf = distr_object.cdf(X)

        if self.ax_cdf is not None:
            self.ax_cdf.plot(X, Y_cdf, color=color, label=label)
            self._plot_borders(self.ax_cdf, color, domain)

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
        self.ax_pdf.legend(title=self._legend_title)#, loc='upper right', bbox_to_anchor=(0.5, -0.05))

        if self.ax_cdf is not None:
            self.ax_cdf.legend()

        if self.ax_log_density is not None:
            self.ax_mult_mom_der.legend()
            self.ax_log_density.legend()
            self.ax_mult_mom_der_2.legend()

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
        print("self exact distr ", self._exact_distr)
        if self._exact_distr is None:
            return

        # with np.printoptions(precision=2):
        #    lab = str(np.array(self._domain))
        X = self._grid(1000)
        Y = self._exact_distr.pdf(X)#[self.distr_object.density_mask])
        # if self._log_density:
        #     Y = np.log(Y)
        self.ax_pdf.set_ylim([np.min(Y) - (np.max(Y) - np.min(Y)) * 0.1, np.max(Y) + (np.max(Y) - np.min(Y)) * 0.1])



        self.ax_pdf.plot(X, Y, c='black', label="exact", linestyle=":")

        if self.ax_log_density is not None:
            self.ax_log_density.plot(X, np.log(Y), c='black', linestyle=":")

        if self.reg_plot is False and self.ax_cdf is not None:
            Y = self._exact_distr.cdf(X)#self.distr_object.distr_mask])
            self.ax_cdf.plot(X, Y, c='black')

    def _grid(self, size, domain=None):
        """
        X values grid for given domain. Optionally use the log scale.
        """
        if domain is None:
            domain = self._domain
        print("domain ", domain)
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
        self.colormap = plt.cm.tab20

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

        color = self.colormap(self.i_plot)#'C{}'.format(self.i_plot)
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

#
# class KL_divergence:
#     """
#     Plot of eigenvalues (of the covariance matrix), several sets of eigenvalues can be added
#     together with error bars and cut-tresholds.
#     Colors are chosen automatically. Slight X shift is used to avoid point overlapping.
#     For log Y scale only positive values are plotted.
#     """
#     def __init__(self, log_y=True, title="Kullback-Leibler divergence"):
#         self._ylim = None
#         self.log_y = log_y
#         self.fig = plt.figure(figsize=(15, 10))
#         self.ax = self.fig.add_subplot(1, 1, 1)
#         self.fig.suptitle(title)
#         self.i_plot = 0
#         self.title = title
#         self.colormap = plt.cm.tab20
#
#         # index of eignevalues dataset
#         if self.log_y:
#             self.ax.set_yscale('log')
#
#     def add_values(self, values, errors=None, threshold=None, label=""):
#         """
#         Add set of eigenvalues into the plot.
#         :param values: array (n,); eigen values in increasing or decreasing ordred, automatically flipped to decreasing.
#         :param errors: array (n,); corresponding std errors
#         :param threshold: horizontal line marking noise level or cut-off eigen value
#         :return:
#         """
#         assert not errors or len(values) == len(errors)
#         if values[0] < values[-1]:
#             values = np.flip(values)
#             if errors is not None:
#                 errors = np.flip(errors)
#             threshold = len(values) - 1 - threshold
#
#         if self.log_y:
#             # plot only positive values
#             i_last_positive = len(values) - np.argmax(np.flip(values) > 0)
#             values = values[:i_last_positive + 1]
#             a, b = np.min(values), np.max(values)
#             self.adjust_ylim( (a / ((b/a)**0.05), b * (b/a)**0.05) )
#         else:
#             a, b = np.min(values), np.max(values)
#             self.adjust_ylim( (a - 0.05 * (b - a), b + 0.05 * (b - a)) )
#
#         color = self.colormap(self.i_plot)#'C{}'.format(self.i_plot)
#         X = np.arange(len(values))# + self.i_plot * 0.1
#         print("X ", X)
#         if errors is None:
#             self.ax.scatter(X, values, label=label, color=color)
#         else:
#             self.ax.errorbar(X, values, yerr=errors, fmt='o', color=color, ecolor=color, capthick=2, label=label)
#         if threshold is not None:
#             self.ax.axhline(y=threshold, color=color)
#         self.i_plot += 1
#
#     def show(self, file=""):
#         """
#         Show the plot or save to file.
#         :param file: filename base, None for show.
#         :return:
#         """
#         self.ax.legend(title="Noise level")
#         _show_and_save(self.fig, file, self.title)
#
#     def adjust_ylim(self, ylim):
#         """
#         Enlarge common domain by given bounds.
#         :param value: [lower_bound, upper_bound]
#         """
#         if self._ylim is None:
#             self._ylim = ylim
#         else:
#             self._ylim = [min(self._ylim[0], ylim[0]), max(self._ylim[1], ylim[1])]


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
    #ax.set_ylim((np.min(central_band), np.max(central_band)))
    for m, y in enumerate(Y.T):
        color = cmap(m)
        ax.plot(X, y, color=color, linewidth=0.5)
    _show_and_save(fig, file, title)


class Spline_plot:
    """
    Plot of KL divergence
    """
    def __init__(self, bspline=False, title="Spline approximation", density=False):
        self._ylim = None
        self.i_plot = 0
        self.title = title
        self.colormap = plt.cm.tab20

        self.indicator_ax = None
        self.smooth_ax = None
        self.bspline_ax = None

        self.indicator_density_ax = None
        self.smooth_density_ax = None
        self.bspline_density_ax = None

        self.interpolation_points = None

        if density:
            if bspline:
                self.fig_spline, axes = plt.subplots(2, 3, figsize=(22, 10))
                self.fig_iter = None

                self.indicator_ax = axes[0][0]
                self.smooth_ax = axes[0][1]
                self.bspline_ax = axes[0][2]
                self.bspline_ax.set_title("Bspline")

                self.indicator_density_ax = axes[1][0]
                self.smooth_density_ax = axes[1][1]
                self.bspline_density_ax = axes[1][2]

            else:
                self.fig_spline, axes = plt.subplots(2, 2, figsize=(22, 10))
                self.fig_iter = None
                self.indicator_ax = axes[0][0]
                self.smooth_ax = axes[0][1]
                self.indicator_density_ax = axes[1][0]
                self.smooth_density_ax = axes[1][1]

        else:
            if bspline:
                self.fig_spline, axes = plt.subplots(2, 3, figsize=(22, 10))
                self.fig_iter = None
                self.indicator_ax = axes[0][0]
                self.smooth_ax = axes[0][1]
                self.bspline_ax = axes[0][2]
                self.bspline_ax.set_title("Bspline")
            else:
                self.fig_spline, axes = plt.subplots(2, 2, figsize=(22, 10))
                self.fig_iter = None
                self.indicator_ax = axes[0]
                self.smooth_ax = axes[1]

        self.fig_spline.suptitle(self.title)

        self.indicator_ax.set_title("Indicator")
        self.smooth_ax.set_title("Smooth")

        # Display integers on x axes
        self.indicator_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.indicator_x = []
        self.indicator_y = []
        self.smooth_x = []
        self.smooth_y = []
        self.bspline_x = []
        self.bspline_y = []
        self.exact_x = None
        self.exact_y = None
        self.ecdf_x = None
        self.ecdf_y = None

        self.indicator_density_x = []
        self.indicator_density_y = []
        self.smooth_density_x = []
        self.smooth_density_y = []
        self.bspline_density_x = []
        self.bspline_density_y = []
        self.exact_density_x = None
        self.exact_density_y = None


    def add_indicator(self, values):
        """
        Add one KL div value
        :param values: tuple
        :return:
        """
        self.indicator_x.append(values[0])
        self.indicator_y.append(values[1])

    def add_smooth(self, values):
        self.smooth_x.append(values[0])
        self.smooth_y.append(values[1])

    def add_bspline(self, values):
        self.bspline_x.append(values[0])
        self.bspline_y.append(values[1])

    def add_indicator_density(self, values):
        """
        Add one KL div value
        :param values: tuple
        :return:
        """
        self.indicator_density_x.append(values[0])
        self.indicator_density_y.append(values[1])

    def add_smooth_density(self, values):
        self.smooth_density_x.append(values[0])
        self.smooth_density_y.append(values[1])

    def add_bspline_density(self, values):
        self.bspline_density_x.append(values[0])
        self.bspline_density_y.append(values[1])

    def _plot_values(self):
        if self.exact_x is not None:
            self.indicator_ax.plot(self.exact_x, self.exact_y, color="black", label="exact")
            self.smooth_ax.plot(self.exact_x, self.exact_y, color="black", label="exact")
            if self.bspline_ax is not None:
                self.bspline_ax.plot(self.exact_x, self.exact_y, color="black", label="exact")

        color = 'C{}'.format(0)
        if self.ecdf_x is not None:
            self.indicator_ax.plot(self.ecdf_x, self.ecdf_y, color=color, label="ECDF")
            self.smooth_ax.plot(self.ecdf_x, self.ecdf_y, color=color, label="ECDF")
            if self.bspline_ax is not None:
                self.bspline_ax.plot(self.ecdf_x, self.ecdf_y, color=color, label="ECDF")

        for i in range(1, len(self.indicator_x)+1):
            color ='C{}'.format(i) # 'C{}'.format(self.i_plot)

            print("self.indicator_x[i-1] ", len(self.indicator_x[i-1]))
            print("self.indicator_y[i-1] ", len(self.indicator_y[i-1]))

            self.indicator_ax.plot(self.indicator_x[i-1], self.indicator_y[i-1], color=color, linestyle="--",
                                   label="{}".format(self.interpolation_points[i-1]))

            self.smooth_ax.plot(self.smooth_x[i-1], self.smooth_y[i-1], color=color, linestyle="--",
                                label="{}".format(self.interpolation_points[i-1]))

            if self.bspline_ax is not None:
                self.bspline_ax.plot(self.bspline_x[i - 1], self.bspline_y[i - 1], color=color, linestyle="--",
                                    label="{}".format(self.interpolation_points[i - 1]))

        if self.exact_density_x is not None:
            self.indicator_density_ax.plot(self.exact_density_x, self.exact_density_y, color="black", label="exact")
            self.smooth_density_ax.plot(self.exact_density_x, self.exact_density_y, color="black", label="exact")

            if self.bspline_density_ax is not None:
                self.bspline_density_ax.plot(self.exact_density_x, self.exact_density_y, color="black", label="exact")

        color = 'C{}'.format(0)

        for i in range(1, len(self.indicator_density_x)+1):
            color ='C{}'.format(i) # 'C{}'.format(self.i_plot)

            print("self.indicator_x[i-1] ", len(self.indicator_density_x[i-1]))
            print("self.indicator_y[i-1] ", len(self.indicator_density_y[i-1]))

            self.indicator_density_ax.plot(self.indicator_density_x[i-1], self.indicator_density_y[i-1], color=color, linestyle="--",
                                   label="{}".format(self.interpolation_points[i-1]))

            self.smooth_density_ax.plot(self.smooth_density_x[i-1], self.smooth_density_y[i-1], color=color, linestyle="--",
                                label="{}".format(self.interpolation_points[i-1]))

            if self.bspline_density_ax is not None:
                self.bspline_density_ax.plot(self.bspline_density_x[i - 1], self.bspline_density_y[i - 1], color=color,
                                             linestyle="--", label="{}".format(self.interpolation_points[i - 1]))

    def add_exact_values(self, x, y):
        self.exact_x = x
        self.exact_y = y

    def add_ecdf(self, x, y):
        self.ecdf_x = x
        self.ecdf_y = y

    def add_density_exact_values(self, x, y):
        self.exact_density_x = x
        self.exact_density_y = y


    def show(self, file=""):
        """
        Show the plot or save to file.
        :param file: filename base, None for show.
        :return:
        """
        self._plot_values()
        self.indicator_ax.legend()
        self.smooth_ax.legend()
        if self.indicator_density_ax is not None:
            self.indicator_density_ax.legend()
            self.smooth_density_ax.legend()

        if self.bspline_ax is not None:
            self.bspline_ax.legend()
            if self.bspline_density_ax is not None:
                self.bspline_density_ax.legend()

        self.fig_spline.show()
        # file = self.title
        # if file[-3:] != "pdf":
        #     file = file + ".pdf"
        #
        # self.fig_spline.savefig(file)


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
        X = self.x_shift + (width*1.1)*np.arange(n_moments+1)
        self.x_shift = X[-1] + 3*width
        self.X_list.append(X)
        X_labels = ['avg'] + [str(m) for m in self.moments_subset]
        self.X_labels.extend(X_labels)
        #plots = []
        sum_Y = np.zeros(n_moments+1)
        yerr = None
        total_Y0 = np.sum(np.mean(level_vars[:, :] / n_samples[:, None], axis=1))
        for il in reversed(range(n_levels)):
            vars = level_vars[il, :]
            Y = np.concatenate(( [np.mean(vars)], vars))
            Y /= n_samples[il]

            if ref_level_vars is not None:
                ref_vars = ref_level_vars[il, self.moments_subset]
                ref_Y = np.concatenate(([np.mean(ref_vars)], ref_vars))
                ref_Y /= n_samples[il]
                diff_Y = ref_Y - Y
                yerr_lower_lim = -np.minimum(diff_Y, 0)
                yerr_upper_lim = np.maximum(diff_Y, 0)
                yerr = np.stack((yerr_lower_lim, yerr_upper_lim), axis=0)
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

label_fontsize = 12
class KL_div_mom_err():

    def __init__(self, title=""):

        self.kl_divs = []
        self.mom_errs = []
        self.densities = []
        self.data = []
        self.title = title
        self.colormap = plt.cm.tab20
        self.i_plot = 0
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 10))

        self.markers = ["o", "v", "s", "p", "X", "D"]

        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

        self.ax.set_xlabel(r'$|\mu - \hat{\mu}|^2$', size=label_fontsize)
        self.ax.set_ylabel(r'$D(\rho_35 \Vert \hat{\rho}_35)$', size=label_fontsize)

    def add_values(self, kl_div, mom_err, density):
        self.data.append((kl_div, mom_err, density))

    def plot_values(self):
        for kl_div, mom_err, density in self.data:
            print("kl div ", kl_div)
            print("mom erro ", mom_err)
            self.ax.plot(mom_err, kl_div, color=self.colormap(self.i_plot), marker=self.markers[self.i_plot], label=density)
            self.i_plot += 1

    def show(self):
        self.plot_values()
        self.ax.legend()
        file = self.title + "_kl_mom_diff.pdf"
        self.fig.show()
        self.fig.savefig(file)


class KL_divergence:
    """
    Plot of KL divergence
    """
    def __init__(self, log_y=True, log_x=False, iter_plot=False, kl_mom_err=True, title="", xlabel="number of moments", ylabel="KL divergence", label="", truncation_err_label=""):
        self._ylim = None
        self.log_y = log_y
        self.i_plot = 0
        self.title = title
        self.colormap = plt.cm.tab20

        if iter_plot:
            self.fig_kl, axes = plt.subplots(1, 2, figsize=(22, 10))
            self.fig_iter = None
            self.ax_kl = axes[0]
            self.ax_iter = axes[1]
        else:
            self.fig_kl, self.ax_kl = plt.subplots(1, 1, figsize=(12, 10))
            self.fig_iter, self.ax_iter = plt.subplots(1, 1, figsize=(12, 10))

        if kl_mom_err:
            self.fig_mom_err, self.ax_mom_err = plt.subplots(1, 1, figsize=(12, 10))


        self.ax_kl.set_title("Kullback-Leibler divergence")
        self.ax_iter.set_title("Optimization iterations")

        # Display integers on x axes
        self.ax_kl.xaxis.set_major_locator(MaxNLocator(integer=True))

        self.ax_kl.set_xlabel(xlabel)
        self.ax_kl.set_ylabel(ylabel)
        self.ax_iter.set_xlabel(xlabel)
        self.ax_iter.set_ylabel("number of iterations")

        self._plot_kl_mom_err = kl_mom_err
        self._x = []
        self._y = []
        self._mom_err_x = []
        self._mom_err_y = []
        self._iter_x = []
        self._failed_iter_x = []
        self._iterations = []
        self._failed_iterations = []
        self._truncation_err = None
        self._label = label
        self._truncation_err_label = truncation_err_label

        if self.log_y:
            self.ax_kl.set_yscale('log')


            #self.ax_mom_err.set_yscale('log')
        if log_x:
            self.ax_kl.set_xscale('log')
            self.ax_iter.set_xscale('log')
            #self.ax_mom_err.set_xscale('log')

    @property
    def truncation_err(self):
        """
        KL divergence between exact density and density produced by certain number of exact moments (is it the first part of overall KL divergence)
        It is used just for inexact moments KL div as a "threshold" value
        :return:
        """
        return self._truncation_err

    @truncation_err.setter
    def truncation_err(self, trunc_err):
        self._truncation_err = trunc_err

    def add_value(self, values):
        """
        Add one KL div value
        :param values: tuple
        :return:
        """
        self._x.append(values[0])
        self._y.append(values[1])

    def add_iteration(self, x, n_iter, failed=False):
        """
        Add number of iterations
        :param x:
        :param n_iter: number of iterations
        :param failed: bool
        :return: None
        """
        if failed:
            self._failed_iter_x.append(x)
            self._failed_iterations.append(n_iter)
        else:
            self._iter_x.append(x)
            self._iterations.append(n_iter)

    def add_moments_l2_norm(self, values):
        self._mom_err_x.append(values[0])
        self._mom_err_y.append(values[1])

    def add_values(self, values):
        """
        Allow add more values
        :param values: array (n,); kl divergences
        :return:
        """
        self._x = values[0]
        self._y = values[1]

        if len(values) == 3:
            self._iterations = values[2]


    def _plot_values(self):
        if self.log_y:
            # plot only positive values
            i_last_positive = len(self._y) - np.argmax(np.flip(self._y) > 0)
            self._y = self._y[:i_last_positive + 1]
            a, b = np.min(self._y), np.max(self._y)
            #self.adjust_ylim((a / ((b / a) ** 0.05), b * (b / a) ** 0.05))
        else:
            a, b = np.min(self._y), np.max(self._y)
            #self.adjust_ylim((a - 0.05 * (b - a), b + 0.05 * (b - a)))

        color = self.colormap(self.i_plot)  # 'C{}'.format(self.i_plot)

        if self._mom_err_y:
            self.ax_kl.plot(self._mom_err_x, self._mom_err_y, ls='solid', color="red", marker="v", label=r'$|\mu - \hat{\mu}|^2$')
            self.ax_kl.plot(self._x, self._y, ls='solid', color=color, marker='o', label="KL div")
        else:
            self.ax_kl.plot(self._x, self._y, ls='solid', color=color, marker='o')

        if self._iterations:
            self.ax_iter.scatter(self._iter_x, self._iterations, color=color, marker="p", label="successful")

        if self._failed_iterations:
            self.ax_iter.scatter(self._failed_iter_x, self._failed_iterations, color="red", marker="p", label="failed")

        if self._plot_kl_mom_err:
            self.ax_mom_err.plot(self._mom_err_y, self._y, ls='solid', color="red", marker="v",
                            label=r'$|\mu - \hat{\mu}|^2$')

        self.i_plot += 1

        if self._truncation_err is not None:
            color = self.colormap(self.i_plot)
            self.ax_kl.axhline(y=self._truncation_err, color=color, label=self._truncation_err_label)
        self.i_plot += 1

    def show(self, file=""):
        """
        Show the plot or save to file.
        :param file: filename base, None for show.
        :return:
        """
        self._plot_values()
        self.ax_kl.legend()
        self.ax_iter.legend()


        self.fig_kl.show()
        file = self.title
        if file[-3:] != "pdf":
            file = file + ".pdf"

        self.fig_kl.savefig(file)

        if self.fig_iter is not None:
            file = self.title + "_iterations.pdf"
            self.fig_iter.show()
            self.fig_kl.savefig(file)

        if self._plot_kl_mom_err:
            file = self.title + "_kl_mom_diff.pdf"
            self.ax_mom_err.legend()
            self.fig_mom_err.show()
            self.fig_mom_err.savefig(file)



###########################################
# test.fixture.mlmc_test_run plot methods #
###########################################

def plot_n_sample_est_distributions(cost, total_std, n_samples):

    fig = plt.figure(figsize=(30,10))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.hist(cost, normed=1)
    ax1.set_xlabel("cost")
    cost_99 = np.percentile(cost, [99])
    ax1.axvline(x=cost_99, label=str(cost_99), c='red')
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(total_std, normed=1)
    ax2.set_xlabel("total var")
    tstd_99 = np.percentile(total_std, [99])
    ax2.axvline(x=tstd_99, label=str(tstd_99), c='red')
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(n_samples, normed=1)
    ax3.set_xlabel("n_samples")
    ns_99 = np.percentile(n_samples, [99])
    ax3.axvline(x=ns_99, label=str(ns_99), c='red')
    ax3.legend()
    plt.show()


def plot_diff_var_subsample(level_variance_diff, n_levels):
    """
    Plot diff between V* and V
    :param level_variance_diff: array of moments sqrt(V/V*)
    :param n_levels: array, number of levels
    :return: None
    """
    import matplotlib.cm as cm
    if len(level_variance_diff) > 0:
        colors = iter(cm.rainbow(np.linspace(0, 1, len(level_variance_diff) + 1)))
        x = np.arange(0, len(level_variance_diff[0]))
        [plt.plot(x, var_diff, 'o', label="%dLMC" % n_levels[index], color=next(colors)) for index, var_diff in
         enumerate(level_variance_diff)]
        plt.legend()
        plt.ylabel(r'$ \sqrt{\frac{V}{V^{*}}}$', rotation=0)
        plt.xlabel("moments")
        plt.show()

        # Levels on x axes
        moments = []
        level_variance_diff = np.array(level_variance_diff)
        for index in range(len(level_variance_diff[0])):
            moments.append(level_variance_diff[:, index])

        colors = iter(cm.rainbow(np.linspace(0, 1, len(moments) + 1)))
        [plt.plot(n_levels, moment, 'o', label=index+1, color=next(colors)) for index, moment in enumerate(moments)]
        plt.ylabel(r'$ \sqrt{\frac{V}{V^{*}}}$', rotation=0)
        plt.xlabel("levels method")
        plt.legend(title="moments")
        plt.show()


def plot_vars(moments_mean, moments_var, n_levels, exact_moments=None, ex_moments=None):
    """
    Plot means with variance whiskers
    :param moments_mean: array, moments mean
    :param moments_var: array, moments variance
    :param n_levels: array, number of levels
    :param exact_moments: array, moments from distribution
    :param ex_moments: array, moments from distribution samples
    :return: None
    """
    import matplotlib.cm as cm
    colors = iter(cm.rainbow(np.linspace(0, 1, len(moments_mean) + 1)))

    x = np.arange(0, len(moments_mean[0]))
    x = x - 0.3
    default_x = x

    for index, means in enumerate(moments_mean):
        if index == int(len(moments_mean)/2) and exact_moments is not None:
            plt.plot(default_x, exact_moments, 'ro', label="Exact moments")
        else:
            x = x + (1 / (len(moments_mean)*1.5))
            plt.errorbar(x, means, yerr=moments_var[index], fmt='o', capsize=3, color=next(colors), label="%dLMC" % n_levels[index])

    if ex_moments is not None:
        plt.plot(default_x-0.125, ex_moments, 'ko', label="Exact moments")

    plt.legend()
    plt.show()


def plot_convergence(quantiles, conv_val, title):
    """
    Plot convergence with moment size for various quantiles.
    :param quantiles: iterable with quantiles
    :param conv_val: matrix of ConvResult, n_quantiles x n_moments
    :param title: plot title and filename used to save
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for iq, q in enumerate(quantiles):
        results = conv_val[iq]
        #X = [r.size for r in results]
        X = np.arange(len(results))
        kl = [r.kl for r in results]
        l2 = [r.l2 for r in results]
        col = plt.cm.tab10(plt.Normalize(0,10)(iq))
        ax.plot(X, kl, ls='solid', c=col, label="kl_q="+str(q), marker='o')
        ax.plot(X, l2, ls='dashed', c=col, label="l2_q=" + str(q), marker='d')
    ax.set_yscale('log')
    ax.set_xscale('log')
    fig.legend()
    fig.suptitle(title)
    fname = title + ".pdf"
    fig.savefig(fname)


def plot_diff_var(ref_mc_diff_vars, n_moments, steps):
    """
    Plot level diff vars
    """
    fig = plt.figure(figsize=(10, 20))
    ax = fig.add_subplot(1, 1, 1)

    error_power = 2.0
    for m in range(1, n_moments):
        color = 'C' + str(m)

        Y = ref_mc_diff_vars[:, m] / (steps ** error_power)

        ax.plot(steps[1:], Y[1:], c=color, label=str(m))
        ax.plot(steps[0], Y[0], 'o', c=color)

        # Y = np.percentile(self.vars_est[:, :, m],  [10, 50, 90], axis=1)
        # ax.plot(target_var, Y[1,:], c=color)
        # ax.plot(target_var, Y[0,:], c=color, ls='--')
        # ax.plot(target_var, Y[2, :], c=color, ls ='--')
        # Y = (self.exact_mean[m] - self.means_est[:, :, m])**2
        # Y = np.percentile(Y, [10, 50, 90], axis=1)
        # ax.plot(target_var, Y[1,:], c='gray')
        # ax.plot(target_var, Y[0,:], c='gray', ls='--')
        # ax.plot(target_var, Y[2, :], c='gray', ls ='--')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.set_ylabel("observed var. of mean est.")
    plt.show()


def plot_var_regression(ref_level_vars, reg_vars, n_levels, n_moments):
    """
    Plot levels variance regression
    """
    # Plot variance regression for exact level variances
    X = np.outer(np.arange(n_levels), np.ones(n_moments - 1)) + 0.1 * np.outer(np.ones(n_levels),
                                                                                         np.arange(n_moments - 1))
    col = np.outer(np.ones(n_levels), np.arange(n_moments - 1))
    plt.scatter(X.ravel(), ref_level_vars[:, 1:].ravel(), c=col.ravel(), cmap=plt.cm.tab10,
                norm=plt.Normalize(0, 10), marker='o')
    for i_mom in range(n_moments - 1):
        col = plt.cm.tab10(plt.Normalize(0, 10)(i_mom))
        plt.plot(X[:, i_mom], reg_vars[:, i_mom + 1], c=col)
    plt.legend()
    plt.yscale('log')
    plt.ylim(1e-10, 1)
    plt.show()


def plot_regression_diffs(all_diffs, n_moments):
    """
    Plot level variance difference regression
    :param all_diffs: list, difference between Estimate._variance_regression result and Estimate.estimate_diff_var result
    :param n_moments: number of moments
    :return:
    """
    for i_mom in range(n_moments - 1):
        diffs = [sample[:, i_mom] for sample in all_diffs]
        diffs = np.array(diffs)
        N, L = diffs.shape
        X = np.outer(np.ones(N), np.arange(L)) + i_mom * 0.1
        col = np.ones_like(diffs) * i_mom
        plt.scatter(X, diffs, c=col, cmap=plt.cm.tab10, norm=plt.Normalize(0, 10), marker='o', label=str(i_mom))
    plt.legend()
    plt.yscale('log')
    plt.ylim(1e-10, 1)
    plt.show()


def plot_mlmc_conv(n_moments, vars_est, exact_mean, means_est, target_var):

    fig = plt.figure(figsize=(10,20))
    for m in range(1, n_moments):
        ax = fig.add_subplot(2, 2, m)
        color = 'C' + str(m)
        Y = np.var(means_est[:,:,m], axis=1)
        ax.plot(target_var, Y, 'o', c=color, label=str(m))

        Y = np.percentile(vars_est[:, :, m],  [10, 50, 90], axis=1)
        ax.plot(target_var, Y[1,:], c=color)
        ax.plot(target_var, Y[0,:], c=color, ls='--')
        ax.plot(target_var, Y[2, :], c=color, ls ='--')
        Y = (exact_mean[m] - means_est[:, :, m])**2
        Y = np.percentile(Y, [10, 50, 90], axis=1)
        ax.plot(target_var, Y[1,:], c='gray')
        ax.plot(target_var, Y[0,:], c='gray', ls='--')
        ax.plot(target_var, Y[2, :], c='gray', ls ='--')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax.set_ylabel("observed var. of mean est.")
    plt.show()


def plot_n_sample_est_distributions(title, cost, total_std, n_samples, rel_moments):
    fig = plt.figure(figsize=(30,10))
    ax1 = fig.add_subplot(2, 2, 1)
    plot_error(cost, ax1, "cost err")

    ax2 = fig.add_subplot(2, 2, 2)
    plot_error(total_std, ax2, "total std err")

    ax3 = fig.add_subplot(2, 2, 3)
    plot_error(n_samples, ax3, "n. samples err")

    ax4 = fig.add_subplot(2, 2, 4)
    plot_error(rel_moments, ax4, "moments err")
    fig.suptitle(title)
    plt.show()


def plot_error(arr, ax, label):
    ax.hist(arr, normed=1)
    ax.set_xlabel(label)
    prc = np.percentile(arr, [99])
    ax.axvline(x=prc, label=str(prc), c='red')
    ax.legend()

# @staticmethod
# def box_plot(ax, X, Y):
#     bp = boxplot(column='age', by='pclass', grid=False)
#     for i in [1, 2, 3]:
#         y = titanic.age[titanic.pclass == i].dropna()
#         # Add some random "jitter" to the x-axis
#         x = np.random.normal(i, 0.04, size=len(y))
#         plot(x, y, 'r.', alpha=0.2)

