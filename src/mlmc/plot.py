import numpy as np
import matplotlib.pyplot as plt

def create_color_bar(size, label, ax = None):
    # Create colorbar
    colormap = plt.cm.gist_ncar
    normalize = plt.Normalize(vmin=0, vmax=size)
    scalar_mappable = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalar_mappable.set_array(np.arange(size))
    ticks = np.linspace(0, int(size/10)*10, 9)
    clb = plt.colorbar(scalar_mappable, ticks=ticks, aspect=50, pad=0.01, ax=ax)
    clb.set_label(label)
    return lambda v: colormap(normalize(v))


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
            pdf_err_title = "error - dashed"
            if error_plot:
                pdf_err_title = "kl-error - dashed"
            self.ax_pdf_err.set_ylabel(pdf_err_title)
            self.ax_cdf_err = self.ax_cdf.twinx()
            self.ax_cdf_err.set_ylabel("error - dashed")


    def add_raw_samples(self, samples):
        """
        Add histogram and ecdf for raw samples.
        :param samples:
        """
        bins = self._grid(np.sqrt(len(samples)))
        self.ax_pdf.hist(samples, density=True, bins=bins, alpha=0.3, label='samples', color='red')
        X = np.sort(samples)
        Y = np.arange(1, len(xs) + 1) / float(len(xs))
        self.ax_cdf.plot(X, Y, 'red')


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
        print("pdf max: ", distr_object.density(0.0))
        Y_pdf = distr_object.density(X)
        self.ax_pdf.plot(X, Y_pdf, label=label, color=color)
        self._plot_borders(self.ax_pdf, color, domain)

        Y_cdf = distr_object.cdf(X)
        self.ax_cdf.plot(X, Y_cdf)
        self._plot_borders(self.ax_cdf, color, domain)

        if self._error_plot and self._exact_distr is not None:
            if self._error_plot == 'kl':
                exact_pdf = self._exact_distr.pdf(X)
                eY_pdf = exact_pdf * np.log(exact_pdf / Y_pdf)
            else:
                eY_pdf = Y_pdf - self._exact_distr.pdf(X)
            self.ax_pdf_err.plot(X, eY_pdf, linestyle="--", color=color)
            eY_cdf = Y_cdf - self._exact_distr.cdf(X)
            self.ax_cdf_err.plot(X, eY_cdf, linestyle="--", color=color)

        self.i_plot += 1

    def show(self, save=""):
        """
        Set colors according to the number of added plots.
        Set domain from all plots.
        Plot exact distribution.
        show, possibly save to file.
        :param save: None, or filename, default name is same as plot title.
        """
        if save == "":
            save = self._title
        self._add_exact_distr()
        self.fig.legend(title=self._legend_title)
        if save is not None:
            self.fig.savefig(save)
        else:
            self.fig.show()

    def reset(self):
        plt.close()
        self._domain = None

    def _plot_borders(self, ax, col, domain=None):
        if domain is None:
            domain = self._domain
        l1 = ax.axvline(x=domain[0], ymin=0, ymax=0.1, color=col)
        l2 = ax.axvline(x=domain[1], ymin=0, ymax=0.1, color=col)
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
        if domain is None:
            domain = self._domain
        if self._log_x:
            X = np.exp(np.linspace(np.log(domain[0]), np.log(domain[1]), size))
        else:
            X = np.linspace(domain[0], domain[1], size)
        return X


# def plot_pdf_approx(ax1, ax2, mc0_samples, mlmc_wrapper, domain, est_domain):
#     """
#     Plot density and distribution, plot contains density estimation from MLMC and histogram created from One level MC
#     TODO: merge with similar method in clss estimate
#     :param ax1: First figure subplot
#     :param ax2: Second figure subplot
#     :param mc0_samples: One level MC samples
#     :param mlmc_wrapper: Object with mlmc instance, must contains distribution object
#     :param domain: Domain from one level MC
#     :param est_domain: Domain from MLMC
#     :return: None
#     """
#     # X = np.exp(np.linspace(np.log(domain[0]), np.log(domain[1]), 1000))
#     # bins = np.exp(np.linspace(np.log(domain[0]), np.log(10), 60))
#     X = np.linspace(domain[0], domain[1], 1000)
#     bins = np.linspace(domain[0], domain[1], len(mc0_samples)/15)
#
#     distr_obj = mlmc_wrapper.distr_obj
#
#     n_levels = mlmc_wrapper.mc.n_levels
#     color = "C{}".format(n_levels)
#     label = "l {}".format(n_levels)
#     Y = distr_obj.density(X)
#     ax1.plot(X, Y, c=color, label=label)
#
#     Y = distr_obj.cdf(X)
#     ax2.plot(X, Y, c=color, label=label)
#
#     if n_levels == 1:
#         ax1.hist(mc0_samples, normed=True, bins=bins, alpha=0.3, label='full MC', color=color)
#         X, Y = ecdf(mc0_samples)
#         ax2.plot(X, Y, 'red')
#
#     ax1.axvline(x=domain[0], c=color)
#     ax1.axvline(x=domain[1], c=color)


class Eigenvalues:
    """
    Plot of eigenvalues (of the covariance matrix), several sets of eigenvalues can be added
    together with error bars and cut-tresholds.
    Colors are chosen automatically. Slight X shift is used to avoid point overlapping.
    For log Y scale only positive values are plotted.
    """
    def __init__(self, log_y = True, title = "eigenvalues"):
        self._ylim = None
        self.log_y = log_y
        self.fig = plt.figure(figsize=(30, 10))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.suptitle(title)
        self.i_plot = 0
        # index of eignevalues dataset
        if self.log_y:
            self.ax.set_yscale('log')

    def add_values(self, values, errors=None, threshold=None, label=""):
        """
        Add set of eigenvalues into the plot.
        :param values: eigen values in increasing or decreasing ordred, automatically flipped to decreasing.
        :param errors: corresponding std errors
        :param threshold: index to mark cut-off
        :return:
        """
        assert not errors or len(values) == len(errors)
        if values[0] < values[-1]:
            values = np.flip(values)
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
            tx = threshold + self.i_plot * 0.1 + 0.05
            self.ax.axvline(x=tx, color=color)
        self.i_plot += 1

    def add_linear_fit(self, values):
        pass

    def show(self, file=""):
        """
        Show the plot or save to file.
        :param filename: filename base, None for show.
        :return:
        """
        self.fig.legend()
        if file == "":
            file = self.title
        if file[-3:] != "pdf":
            file = file + ".pdf"
        if file is None:
            self.fig.show()
        else:
            self.fig.savefig(file)

    def adjust_ylim(self, ylim):
        """
        Enlarge common domain by given bounds.
        :param value: [lower_bound, upper_bound]
        """
        if self._ylim is None:
            self._ylim = ylim
        else:
            self._ylim = [min(self._ylim[0], ylim[0]), max(self._ylim[1], ylim[1])]


