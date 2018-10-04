import os
import sys
import numpy as np

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', '..', 'src'))
import mlmc.mlmc
from mlmc.distribution import Distribution


class ProcessMLMC:
    """
    Base of future class dedicated to all kind of processing of the collected samples
    MLMC should only collect the samples.
    """

    def __init__(self, mlmc):
        self.mlmc = mlmc

        # Distribution aproximation, created by method 'construct_density'
        self._distribution = None

        # Bootstrap estimates of variances of MLMC estimators.
        # Created by method 'construct_bootstrap_estimates'.
        # BS estimate of variance of MLMC mean estimate. For every moment.
        self._bs_mean_variance = None
        # BS estimate of variance of MLMC variance estimate. For every moment.
        self._bs_var_variance = None
        # BS estimate of variance of MLMC level mean estimate. For every level, every moment,
        self._bs_level_mean_variance = None
        # BS estimate of variance of MLMC level variance estimate. For every level, every moment,
        self._bs_level_var_variance = None


    @property
    def n_levels(self):
        return self.mlmc.n_levels

    @property
    def n_samples(self):
        return self.mlmc.n_samples

    @property
    def levels(self):
        return self.mlmc.levels

    @property
    def distribution(self):
        assert self._distribution is not None, "Need to call construct_density before."
        return self._distribution

    @property
    def sim_steps(self):
        return self.mlmc.sim_steps

    def approx_pdf(self, x):
        return self.distribution.density(x)

    def approx_cdf(self, x):
        return self.distribution.cdf(x)

    def estimate_domain(self):
        return self.mlmc.estimate_domain()

    def construct_density(self, moments_fn, tol=1.95, reg_param=0.01):
        """
        Construct approximation of the density using given moment functions.
        Args:
            moments_fn: Moments object, determines also domain and n_moments.
            tol: Tolerance of the fitting problem, with account for variances in moments.
                 Default value 1.95 corresponds to the two tail confidency 0.95.
            reg_param: Regularization parameter.
        """
        # [print("integral density ", integrate.simps(densities[index], x[index])) for index, density in
        # enumerate(densities)]

        domain = moments_fn.domain

        # t_var = 1e-5
        # ref_diff_vars, _ = mlmc.estimate_diff_vars(moments_fn)
        # ref_moments, ref_vars = mc.estimate_moments(moments_fn)
        # ref_std = np.sqrt(ref_vars)
        # ref_diff_vars_max = np.max(ref_diff_vars, axis=1)
        # ref_n_samples = mc.set_target_variance(t_var, prescribe_vars=ref_diff_vars)
        # ref_n_samples = np.max(ref_n_samples, axis=1)
        # ref_cost = mc.estimate_cost(n_samples=ref_n_samples)
        # ref_total_std = np.sqrt(np.sum(ref_diff_vars / ref_n_samples[:, None]) / n_moments)
        # ref_total_std_x = np.sqrt(np.mean(ref_vars))

        est_moments, est_vars = self.mlmc.estimate_moments(moments_fn)

        # def describe(arr):
        #     print("arr ", arr)
        #     q1, q3 = np.percentile(arr, [25, 75])
        #     print("q1 ", q1)
        #     print("q2 ", q3)
        #     return "{:f8.2} < {:f8.2} | {:f8.2} | {:f8.2} < {:f8.2}".format(
        #         np.min(arr), q1, np.mean(arr), q3, np.max(arr))

        print("n_levels: ", self.n_levels)
        moments_data = np.stack((est_moments, est_vars), axis=1)
        distr_obj = Distribution(moments_fn, moments_data, domain=domain)
        distr_obj.estimate_density_minimize(tol, reg_param)  # 0.95 two side quantile
        # distr_obj.estimate_density_minimize(0.1)  # 0.95 two side quantile
        self._distribution = distr_obj


    def construct_bootstrap_estimates(self, moments_fn, n_subsamples=100, sample_vector=None):
        """
        Estimate error of the MLMC mean value estimator and of the MLMC variance estimator
        using a bootstrapping. Bootstrapping use subsampling with replacements !!
        :param n_subsamples: Number of subsamples to perform. Default is 1000. This should guarantee at least
                             first digit to be correct.
        :param sample_vector: By default same as the original sampling.
        :return:
        """
        if sample_vector is None:
            sample_vector = self.mlmc.n_samples
        n_moments = moments_fn.size
        mean_samples = np.empty((n_subsamples, n_moments))
        var_samples = np.empty((n_subsamples, n_moments))
        level_var_samples = np.empty((n_subsamples, self.n_levels, n_moments))
        level_mean_samples = np.empty((n_subsamples, self.n_levels, n_moments))

        self.mlmc.update_moments(moments_fn)
        for i in range(n_subsamples):
            self.mlmc.subsample(sample_vector)
            means, vars = self.mlmc.estimate_moments(moments_fn)
            mean_samples[i, : ] = means
            var_samples[i, :] = vars
            level_vars, _ = self.mlmc.estimate_diff_vars(moments_fn)
            level_var_samples[i, :, : ] = level_vars
            level_means = self.mlmc.estimate_level_means(moments_fn)
            level_mean_samples[i, :, :] = level_means
        self._bs_mean_variance = np.var(mean_samples, axis=0, ddof=1)
        self._bs_var_variance = np.var(mean_samples, axis=0, ddof=1)
        self._bs_level_mean_variance = np.var(level_mean_samples, axis=0, ddof=1)
        self._bs_level_var_variance = np.var(level_var_samples, axis=0, ddof=1)
        self.mlmc.clean_subsamples()
    # def _plot_var_fraction(self, vars_frac):
    #     """
    #     :param vars_frac:
    #     :return:
    #     """

    def _plot_level_moment_value(self, ax, value, marker='o'):
        import matplotlib.pyplot as plt
        n_levels = value.shape[0]
        n_moments = value.shape[1]
        for m in range(n_moments):
            color = plt.cm.rainbow(plt.Normalize(0, n_moments)(m))

            X = np.arange(n_levels) + 0.01 * m
            Y = value[:, m]
            col = np.ones(n_levels)[:, None] * np.array(color)[None, :]
            ax.scatter(X, Y, c=col, marker=marker, label="var, m=" + str(m+1))

    def plot_bootstrap_variance_variance(self, moments_fn):
        import matplotlib.pyplot as plt
        self.mlmc.clean_subsamples()
        means, vars = self.mlmc.estimate_moments(moments_fn)
        level_var_var = self.mlmc._variance_of_variance()
        level_var_frac = self._bs_level_var_variance[:, 1:] #/ self.n_samples[:, None]
        total_var_frac = self._bs_var_variance[1:]
        bs_variances = np.concatenate((total_var_frac[None, :], level_var_frac), axis=0)

        vars, _  = self.mlmc.estimate_diff_vars(moments_fn)
        tot_mean, tot_var = self.mlmc.estimate_moments(moments_fn)
        level_var_frac = vars[:, 1:] #/ self.n_samples[:, None]
        total_var_frac = tot_var[1:]
        est_variances = np.concatenate((total_var_frac[None, :], level_var_frac), axis=0)


        #n_moments = self._bs_var_variance.shape[0]
        #assert n_moments == moments_fn.size



        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1)

        self._plot_level_moment_value(ax, bs_variances, marker='.')
        self._plot_level_moment_value(ax, est_variances, marker='d')

        ax.legend()
        ax.set_yscale('log')
        ax.set_ylim((1e-17, 0.1))
        plt.show()


    def plot_means_and_vars(self, ax, moments_fn, ):
        """
        Plot means with variance whiskers to given axes.
        :param moments_mean: array, moments mean
        :param moments_var: array, moments variance
        :param n_levels: array, number of levels
        :param exact_moments: array, moments from distribution
        :param ex_moments: array, moments from distribution samples
        :return:
        """

        moment_fn
        colors = iter(cm.rainbow(np.linspace(0, 1, len(moments_mean) + 1)))

        x = np.arange(0, len(moments_mean[0]))
        x = x - 0.3
        default_x = x

        for index, means in enumerate(moments_mean):
            if index == int(len(moments_mean) / 2) and exact_moments is not None:
                plt.plot(default_x, exact_moments, 'ro', label="Exact moments")
            else:
                x = x + (1 / (len(moments_mean) * 1.5))
                plt.errorbar(x, means, yerr=moments_var[index], fmt='o', capsize=3, color=next(colors),
                             label="%dLMC" % n_levels[index])

        if ex_moments is not None:
            plt.plot(default_x - 0.125, ex_moments, 'ko', label="Exact moments")

        plt.legend()
        plt.show()
        exit()

    def plot_diff_vars(self, ax, moments_fn, i_moments=[], i_style=0):
        import matplotlib.pyplot as plt

        est_diff_vars, n_samples = self.mlmc.estimate_diff_vars(moments_fn)
        reg_diff_vars = self.mlmc.estimate_diff_vars_regression(moments_fn)

        marker = ['o', 'd', ',', '^'][i_style]
        line_style = ['-', '--', '-.', ':'][i_style]

        if not i_moments:
            i_moments = range(1, moments_fn.size)

        for m in i_moments:
            color = plt.cm.rainbow(plt.Normalize(0, len(i_moments))(m))
            Y = est_diff_vars[:, m]
            col = np.ones_like(Y)[:, None] * np.array(color)[None, :]
            ax.scatter(self.sim_steps, Y, c=col, marker=marker, label="var, m=" + str(m))
        Y = reg_diff_vars[1:]
        ax.plot(self.sim_steps[1:], Y, c=color, linestyle=line_style, label="reg")
        # ax.clim(0, len(i_moments))
        # ax.colorbar()

        #
        #
        #     Y = np.percentile(self.vars_est[:, :, m],  [10, 50, 90], axis=1)
        #     ax.plot(target_var, Y[1,:], c=color)
        #     ax.plot(target_var, Y[0,:], c=color, ls='--')
        #     ax.plot(target_var, Y[2, :], c=color, ls ='--')
        #     Y = (self.exact_mean[m] - self.means_est[:, :, m])**2
        #     Y = np.percentile(Y, [10, 50, 90], axis=1)
        #     ax.plot(target_var, Y[1,:], c='gray')
        #     ax.plot(target_var, Y[0,:], c='gray', ls='--')
        #     ax.plot(target_var, Y[2, :], c='gray', ls ='--')
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.set_ylabel("level variance $V_l$")
        ax.set_xlabel("step h_l")


class CompareLevels:
    """
    Class to compare MLMC for different number of levels.
    """

    def __init__(self, mlmc_list, **kwargs):
        """
        Args:
            List of MLMC instances with collected data.
        """
        self.mlmc = [ProcessMLMC(mc) for mc in mlmc_list]
        self.mlmc_dict = {mc.n_levels: mc for mc in self.mlmc}

        # Directory for plots.
        self.output_dir = kwargs.get('output_dir', "")
        # Default moments, type, number.
        self.log_scale = kwargs.get('log_scale', False)
        self.moment_class = kwargs.get('moment_class', mlmc.moments.Legendre)
        self.n_moments = kwargs.get('n_meoments', 21)
        # Optional quantity name used in plots
        self.quantity_name = kwargs.get('quantity_name', 'X')

        # Set domain to union of domains  of all mlmc:
        self.domain = self.common_domain()

    def common_domain(self):
        L = +np.inf
        U = -np.inf
        for mc in self.mlmc:
            l, u = mc.estimate_domain()
            L = min(l, L)
            U = max(u, U)
        return (L, U)


    def __getitem__(self, n_levels):
        return self.mlmc_dict[n_levels]


    @property
    def moments(self):
        return self.moment_class(self.n_moments, self.domain, log=self.log_scale)

    @property
    def moments_uncut(self):
        return self.moment_class(self.n_moments, self.domain, log=self.log_scale, safe_eval=False)

    def collected_report(self):
        """
        Print a record about existing levels, their collected samples, etc.
        """

        print("\n#Levels |     N collected samples")
        for mlmc in self.mlmc:
            tab_fields = ["{:8}".format(n) for n in mlmc.n_samples]
            print("{:7} | {}".format(mlmc.n_levels, " ".join(tab_fields)))
        print("\n")

    def set_common_domain(self, i_mlmc, domain=None):
        if domain is not None:
            self._domain = domain
        self._domain = self.mlmc[i_mlmc].estimate_domain()

    def plot_means(self, moments_fn):
        pass

    def construct_densities(self, tol=1.95, reg_param=0.01):
        for mc in self.mlmc:
            mc.construct_density(self.moments, tol, reg_param)

    @staticmethod
    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / float(len(xs))
        return xs, ys

    def plot_densities(self, i_sample_mlmc=0):
        """
        Plot constructed densities (see construct densities)
        Args:
            i_sample_mlmc: Index of MLMC used to construct histogram and ecdf.

        Returns:
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(30, 10))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        x_axis_label = self.quantity_name
        if self.log_scale:
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            x_axis_label = "log " + x_axis_label
        # ax1.set_yscale('log')

        ax1.set_title("PDF approximations")
        ax2.set_title("CDF approximations")
        ax1.set_ylabel("probability density")
        ax2.set_ylabel("probability")
        ax1.set_xlabel(x_axis_label)
        ax2.set_xlabel(x_axis_label)

        # Add histogram and ecdf
        if i_sample_mlmc is not None:
            mc0_samples = self.mlmc[i_sample_mlmc].levels[0].sample_values[:, 0]
            domain = self.mlmc[i_sample_mlmc].estimate_domain()
            if self.log_scale:
                bins = np.exp(np.linspace(np.log(domain[0]), np.log(domain[1]), np.sqrt(len(mc0_samples))))
            else:
                bins = np.linspace(domain[0], domain[1], np.sqrt(len(mc0_samples)))
            ax1.hist(mc0_samples, normed=True, bins=bins, alpha=0.3, label='full MC', color='red')
            X, Y = self.ecdf(mc0_samples)
            ax2.plot(X, Y, 'red')
        #
        for mc in self.mlmc:
            domain = mc.distribution.domain
            if self.log_scale:
                X = np.exp(np.linspace(np.log(domain[0]), np.log(domain[1]), 1000))
            else:
                X = np.linspace(domain[0], domain[1], 1000)
            color = "C{}".format(mc.n_levels)
            label = "L = {}".format(mc.n_levels)
            Y = mc.approx_pdf(X)
            ax1.plot(X, Y, c=color, label=label)

            Y = mc.approx_cdf(X)
            ax2.plot(X, Y, c=color, label=label)

            ax1.set_ylim(0, 2)
            ax1.axvline(x=domain[0], ymin=0, ymax=0.1, c=color)
            ax1.axvline(x=domain[1], ymin=0, ymax=0.1, c=color)

        ax1.legend()
        ax2.legend()
        fig.savefig('compare_distributions.pdf')
        plt.show()

    def plot_level_vars(self, l_mlmc, n_moments):
        """
        For given i_mlmc plot level variances and regression curve.
        Args:
            n_levels: List of  i_mlmc to plot.
            n_moments: n_moments to plot
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 2, 1)
        if type(n_moments) is int:
            n_moments = list(range(n_moments))

        for i, l in enumerate(l_mlmc):
            mc = self.mlmc_dict[l]
            mc.plot_diff_vars(ax, self.moments, n_moments, i_style=i)
        ax.legend()
        fig.savefig('level_vars_regression.pdf')
        plt.show()

    def construct_bootstrap_estimates(self, n_samples):
        for mc in self.mlmc:
            mc.construct_bootstrap_estimates(self.moments, n_subsamples=n_samples)

    def plot_var_var(self, nl):
        self[nl].plot_bootstrap_variance_variance(self.moments)
