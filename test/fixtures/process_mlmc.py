import os
import sys
import numpy as np
import matplotlib.pyplot as plt

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', '..', 'src'))
import mlmc.mlmc
from mlmc.distribution import Distribution


def create_color_bar(size, label, ax = None):
    # Create colorbar
    colormap = plt.cm.viridis
    normalize = plt.Normalize(vmin=0, vmax=size)
    scalarmappaple = plt.cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(np.arange(size))
    ticks = np.linspace(0, int(size/10)*10, 9)
    clb = plt.colorbar(scalarmappaple, ticks=ticks, aspect=50, pad=0.01, ax=ax)
    clb.set_label(label)
    return lambda v: colormap(normalize(v))

class ProcessMLMC:
    """
    Base of future class dedicated to all kind of processing of the collected samples
    MLMC should only collect the samples.
    """

    def __init__(self, mlmc, moments):
        self.mlmc = mlmc
        self.moments = moments

        # Distribution aproximation, created by method 'construct_density'
        self._distribution = None

        # Bootstrap estimates of variances of MLMC estimators.
        # Created by method 'ref_estimates_bootstrap'.
        # BS estimate of variance of MLMC mean estimate. For every moment.
        self._bs_mean_variance = None
        # BS estimate of variance of MLMC variance estimate. For every moment.
        self._bs_var_variance = None
        # BS estimate of variance of MLMC level mean estimate. For every level, every moment,
        self._bs_level_mean_variance = None
        # BS estimate of variance of MLMC level variance estimate. For every level, every moment,
        self._bs_level_var_variance = None

    @property
    def n_moments(self):
        return self.moments.size


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

    def construct_density(self, tol=1.95, reg_param=0.01):
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
        moments_fn = self.moments
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

    def _bs_get_estimates(self):
        moments_fn = self.moments
        mean_est, var_est = self.mlmc.estimate_moments(moments_fn)
        level_var_est, _ = self.mlmc.estimate_diff_vars(moments_fn)
        level_mean_est = self.mlmc.estimate_level_means(moments_fn)
        return mean_est, var_est, level_mean_est, level_var_est

    def _bs_get_estimates_log(self):
        moments_fn = self.moments
        mean_est, var_est = self.mlmc.estimate_moments(moments_fn)
        level_var_est, _ = self.mlmc.estimate_diff_vars(moments_fn)
        level_mean_est = self.mlmc.estimate_level_means(moments_fn)
        return mean_est, var_est, level_mean_est, level_var_est


    def _bs_get_estimates_regression(self):
        moments_fn = self.moments
        mean_est, var_est = self.mlmc.estimate_moments(moments_fn)
        level_var_est, _ = self.mlmc.estimate_diff_vars(moments_fn)
        level_mean_est = self.mlmc.estimate_level_means(moments_fn)
        level_var_est = self.mlmc.estimate_diff_vars_regression(moments_fn, level_var_est)
        var_est = np.sum(level_var_est[:, :]/self.n_samples[:,  None], axis=0)
        return mean_est, var_est, level_mean_est, level_var_est


    def ref_estimates_bootstrap(self, n_subsamples=100, sample_vector=None, regression=False, log=False):
        """
        Use current MLMC sample_vector to compute reference estimates for: mean, var, level_means, leval_vars.

        Estimate error of these MLMC estimators using a bootstrapping (with replacements).
        Bootstrapping samples MLMC estimators using provided 'sample_vector'.
        Reference estimates are used as mean values of the estimators.
        :param n_subsamples: Number of subsamples to perform. Default is 1000. This should guarantee at least
                             first digit to be correct.
        :param sample_vector: By default same as the original sampling.
        :return: None. Set reference and BS estimates.
        """
        moments_fn = self.moments
        if sample_vector is None:
            sample_vector = self.mlmc.n_samples
        if len(sample_vector) > self.n_levels:
            sample_vector = sample_vector[:self.n_levels]


        estimate_fn = self._bs_get_estimates
        if regression:
            estimate_fn = self._bs_get_estimates_regression
        if log:
            _est_fn = estimate_fn
            def estimate_fn():
                (m, v, lm, lv) = _est_fn()
                return (m, np.log(np.maximum(v, 1e-10)), lm, np.log(np.maximum(lv, 1e-10)))


        self.mlmc.update_moments(moments_fn)
        estimates = estimate_fn()
        est_samples = [ np.zeros(n_subsamples) * est[..., None]  for est in estimates]
        for i in range(n_subsamples):
            self.mlmc.subsample(sample_vector)
            sub_estimates = estimate_fn()
            for es, se, e in zip(est_samples, sub_estimates, estimates):
                es[..., i] = se - e

        bs_estimates = [np.sum(est ** 2, axis=-1) / n_subsamples for est in est_samples]
        mvar, vvar, lmvar, lvvar = bs_estimates
        m, v, lm, lv = estimates

        self._ref_mean = m
        self._ref_var = v
        self._ref_level_mean = lm
        self._ref_level_var = lv

        self._bs_n_samples = self.n_samples
        self._bs_mean_variance = mvar
        self._bs_var_variance = vvar

        # Var dX_l =  n * Var[ mean dX_l ] = n * (1 / n^2) * n * Var dX_l
        self._bs_level_mean_variance = lmvar * self.n_samples[:, None]
        self._bs_level_var_variance = lvvar
        self.mlmc.clean_subsamples()




    # def _plot_var_fraction(self, vars_frac):
    #     """
    #     :param vars_frac:
    #     :return:
    #     """

    def _plot_level_moment_value(self, ax, cmap, value, marker='o'):
        n_levels = value.shape[0]
        n_moments = value.shape[1]
        moments_x_step = 0.5/n_moments
        for m in range(n_moments):
            color = cmap(m)
            X = np.arange(n_levels) + moments_x_step * m
            Y = value[:, m]
            col = np.ones(n_levels)[:, None] * np.array(color)[None, :]
            ax.scatter(X, Y, c=col, marker=marker, label="var, m=" + str(m+1))

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

        #self._plot_level_moment_value(ax, bs_variances, marker='.')
        #self._plot_level_moment_value(ax, est_variances, marker='d')
        self._plot_level_moment_value(ax, fraction, marker='o')

        #ax.legend(loc=6)
        lbls = ['Total'] + [ 'L{:2d}'.format(l+1) for l in range(self.n_levels)]
        ax.set_xticks(ticks = np.arange(self.n_levels + 1))
        ax.set_xticklabels(lbls)
        ax.set_yscale('log')
        ax.set_ylim((0.3, 3))

        self.color_bar(moments_fn.size, 'moments')

        fig.savefig('bs_var_vs_var.pdf')
        plt.show()



    def plot_bootstrap_var_var(self):
        """
        Plot fraction (MLMC var est) / (BS var set) for the total variance and level variances.
        :param moments_fn:
        :return:
        """
        moments_fn = self.moments
        mean, var, l_mean, l_var = self._bs_get_estimates(moments_fn)
        print("n samples:", self.n_samples)
        l_var = l_var
        est_variances = np.concatenate((var[None, 1:], l_var[:, 1:]), axis=0)

        bs_var_var = self._bs_var_variance[1:]
        #bs_l_var_var = self._bs_level_var_variance / (self.n_samples[:, None])**2
        vv  =  self.mlmc._variance_of_variance()[:, None]


        #print(vv)
        #bs_l_var_var = - np.log(self._bs_level_var_variance[:, 1:]) / self.mlmc._variance_of_variance()[:, None]  / self.n_samples[:, None]
        bs_l_var_var = self._bs_level_var_variance[:, 1:] / l_var[:, 1:]**2
        bs_l_var_var = bs_l_var_var * (self.n_samples[:, None] - 1) / 2
        #bs_l_var_var = bs_l_var_var ** (1 / self.mlmc._variance_of_variance()[:, None])
        bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)

        #fraction = est_variances / bs_variances

        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1)
        cmap = create_color_bar(moments_fn.size, 'moments')
        self._plot_level_moment_value(ax, cmap, bs_variances, marker='.')
        #self._plot_level_moment_value(ax, est_variances, marker='d')
        #self._plot_level_moment_value(ax, fraction, marker='o')

        #ax.legend(loc=6)
        lbls = ['Total'] + [ 'L{:2d}'.format(l+1) for l in range(self.n_levels)]
        ax.set_xticks(ticks = np.arange(self.n_levels + 1))
        ax.set_xticklabels(lbls)
        ax.set_yscale('log')
        ax.set_ylim((0.01, 1000))


        fig.savefig('bs_var_vs_var.pdf')
        plt.show()


    def plot_means_and_vars(self, ax, ):
        """
        Plot means with variance whiskers to given axes.
        :param moments_mean: array, moments mean
        :param moments_var: array, moments variance
        :param n_levels: array, number of levels
        :param exact_moments: array, moments from distribution
        :param ex_moments: array, moments from distribution samples
        :return:
        """
        moments_fn = self.moments
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

        if not i_moments:
            i_moments = moments_fn.size
        if type(i_moments) is int:
            i_moments = list(range(i_moments))
        i_moments = np.array(i_moments)

        cmap = create_color_bar(moments_fn.size, 'moments', ax=ax)

        #est_diff_vars, n_samples = self.mlmc.estimate_diff_vars(moments_fn)
        reg_diff_vars = self.mlmc.estimate_diff_vars_regression(moments_fn) / self.n_samples[:, None]
        ref_diff_vars = self._ref_level_var / self.n_samples[:, None]


        self._plot_level_moment_value(ax, cmap(i_moments), ref_diff_vars[:, i_moments], marker='o')
        # add regression curves
        moments_x_step = 0.5 / self.n_moments
        for m in i_moments:
            color = cmap(m)
            X = np.arange(self.n_levels) + moments_x_step * m
            Y = reg_diff_vars[1:, m]
            ax.plot(X[1:], Y, c=color, label="reg")

            ax_err.plot(X[:], reg_diff_vars[:, m]/ref_diff_vars[:, m], c=color, label="reg")

        ax.set_yscale('log')
        ax.set_ylabel("level variance $V_l$")
        ax.set_xlabel("step h_l")

        ax_err.set_ylabel("regresion var. / reference var.")

        #ax.legend(loc=2)
        fig.savefig('level_vars_regression.pdf')
        plt.show()


class CompareLevels:
    """
    Class to compare MLMC for different number of levels.
    """

    def __init__(self, mlmc_list, **kwargs):
        """
        Args:
            List of MLMC instances with collected data.
        """
        self._mlmc_list = mlmc_list
        # Directory for plots.
        self.output_dir = kwargs.get('output_dir', "")
        # Optional quantity name used in plots
        self.quantity_name = kwargs.get('quantity_name', 'X')

        self.reinit(**kwargs)

    def reinit(self, **kwargs):
        """
        Re-create new ProcessMLMC objects from same original MLMC list.
        Set new parameters in particular for moments.
        :return:
        """
        # Default moments, type, number.
        self.log_scale = kwargs.get('log_scale', False)
        # Name of Moments class to use.
        self.moment_class = kwargs.get('moment_class', mlmc.moments.Legendre)
        # Number of moments.
        self.n_moments = kwargs.get('n_moments', 21)

        # Set domain to union of domains  of all mlmc:
        self.domain = self.common_domain()

        self._moments = self.moment_class(self.n_moments, self.domain, self.log_scale)

        self.mlmc = [ProcessMLMC(mc, self._moments) for mc in self._mlmc_list]
        self.mlmc_dict = {mc.n_levels: mc for mc in self.mlmc}



        self._moments_params = None



    def common_domain(self):
        L = +np.inf
        U = -np.inf
        for mc in self._mlmc_list:
            l, u = mc.estimate_domain()
            L = min(l, L)
            U = max(u, U)
        return (L, U)


    def __getitem__(self, n_levels):
        return self.mlmc_dict[n_levels]


    @property
    def moments(self):
        return self._moments

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


    def ref_estimates_bootstrap(self, n_samples, sample_vector=None):
        for mc in self.mlmc:
            mc.ref_estimates_bootstrap(self.moments, n_subsamples=n_samples, sample_vector=sample_vector)

    def plot_var_compare(self, nl):
        self[nl].plot_bootstrap_variance_compare(self.moments)

    def plot_var_var(self, nl):
        self[nl].plot_bootstrap_var_var(self.moments)