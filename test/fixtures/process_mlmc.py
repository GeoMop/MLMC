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

    def set_moments_color_bar(self, ax):
        self._moments_cmap = create_color_bar(self.n_moments, "Moments", ax)

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
        #mean_est, var_est = self.mlmc.estimate_moments(moments_fn)
        level_var_est, _ = self.mlmc.estimate_diff_vars(moments_fn)
        level_mean_est = self.mlmc.estimate_level_means(moments_fn)
        return level_mean_est, level_var_est


    def _bs_get_estimates_regression(self):
        moments_fn = self.moments
        #mean_est, var_est = self.mlmc.estimate_moments(moments_fn)
        level_var_est, _ = self.mlmc.estimate_diff_vars(moments_fn)
        level_mean_est = self.mlmc.estimate_level_means(moments_fn)
        reg_level_var_est = self.mlmc.estimate_diff_vars_regression(moments_fn, level_var_est)

        print("ns: ", self.n_samples)
        raw_diff = (np.log(level_var_est[:,1:]) - np.log(self._ref_level_var[:,1:])) * np.sqrt(self.n_samples)[:, None]
        reg_diff = (np.log(reg_level_var_est[:, 1:]) - np.log(self._ref_level_var[:, 1:])) * np.sqrt(self.n_samples)[:, None]
        print("raw - ref: ", np.linalg.norm(raw_diff))
        print("reg - ref: ", np.linalg.norm(reg_diff))
        #self.plot_var_regression([1,2,4,8,16,20])
        #var_est = np.sum(level_var_est[:, :]/self.n_samples[:,  None], axis=0)
        return level_mean_est, reg_level_var_est


    def check_bias(self, a, b, var, label):
        diff = np.abs(a - b)
        tol = 2*np.sqrt(var) + 1e-20
        if np.any( diff > tol):
            print("Bias ", label)
            it = np.nditer(diff, flags=['multi_index'])
            while not it.finished:
                midx = it.multi_index
                sign = "<" if diff[midx] < tol[midx] else ">"
                print("{:6} {:8.2g} {} {:8.2g} {:8.3g} {:8.3g}".format(
                    str(midx), diff[midx], sign, tol[midx], a[midx], b[midx]))
                it.iternext()

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
        sample_vector = np.array(sample_vector)

        level_estimate_fn = self._bs_get_estimates
        if regression:
            level_estimate_fn = self._bs_get_estimates_regression

        def full_fn(est_fn):
            if log:
                def estimate_fn():
                    lm, lv = est_fn()
                    m, v, lm, lv = (np.sum(lm, axis=0), np.sum(lv[:, :] / sample_vector[:, None], axis=0), lm, lv)
                    v[1:] = np.log(np.maximum(v[1:], 1e-10))
                    lv[:, 1:] = np.log(np.maximum(lv[:, 1:], 1e-10))
                    return (m, v, lm, lv)
            else:
                def estimate_fn():
                    lm, lv = est_fn()
                    return (np.sum(lm, axis=0), np.sum(lv[:, :] / sample_vector[:, None], axis=0), lm, lv)
            return estimate_fn

        estimate_fn = full_fn(level_estimate_fn)

        self.mlmc.update_moments(moments_fn)
        estimates = full_fn(self._bs_get_estimates)()
        # if log:
        #     m, v, lm, lv = estimates
        #     v[1:] = np.log(np.maximum(v[1:], 1e-10))
        #     lv[:, 1:] = np.log(np.maximum(lv[:, 1:], 1e-10))
        #     estimates = (m, v, lm, lv)
        m, v, lm, lv = estimates
        self._ref_mean = m
        self._ref_var = v
        self._ref_level_mean = lm
        self._ref_level_var = lv


        est_samples = [ np.zeros(n_subsamples) * est[..., None]  for est in estimates]
        for i in range(n_subsamples):
            self.mlmc.subsample(sample_vector)
            sub_estimates = estimate_fn()
            for es, se in zip(est_samples, sub_estimates):
                es[..., i] = se

        bs_mean_est = [np.mean(est, axis=-1) for est in est_samples]
        bs_err_est = [np.var(est, axis=-1, ddof=1) for est in est_samples]
        mvar, vvar, lmvar, lvvar = bs_err_est

        self._bs_n_samples = self.n_samples
        self._bs_mean_variance = mvar
        self._bs_var_variance = vvar

        # ???
        # Err[ dX_l ] =  n * Var[ mean dX_l ] = n * (1 / n^2) * n * Var dX_l
        self._bs_level_mean_variance = lmvar * self.n_samples[:, None]
        self._bs_level_var_variance = lvvar

        # Check bias
        mmean, vmean, lmmean, lvmean = bs_mean_est
        self._bs_mean_mean = mmean
        self._bs_var_mean = vmean
        self._bs_level_mean_mean = lmmean
        self._bs_level_var_mean = lvmean
        self.mlmc.clean_subsamples()
        return

        self.check_bias(m,   mmean, mvar,   "Mean")
        self.check_bias(v,   vmean, vvar,   "Variance")
        self.check_bias(lm, lmmean, lmvar,  "Level Mean")
        self.check_bias(lv, lvmean, lvvar,  "Level Varaince")





    # def _plot_var_fraction(self, vars_frac):
    #     """
    #     :param vars_frac:
    #     :return:
    #     """

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


    def plot_bs_var_reg_var(self):
        """
        Test that  MSE of log V_l scales as variance of log chi^2_{N-1}, that is approx. 2 / (n_samples-1).
        """
        vv = self.mlmc._variance_of_variance(self._bs_n_samples)
        bs_l_var_var = (self._bs_level_var_variance[:, :]) / vv[:, None]
        bs_var_var = self._bs_var_variance[:]  # - np.log(total_var_var_scale)
        bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
        self.plot_bs_variances(bs_variances, log=True,
                               y_label="BS est. of var. of $\hat V^r$, $\hat V^r_l$ estimators.",
                               y_lim=(0.1, 20))



    def plot_var_error_contributions(self):
        """
        Plot error in the total variance and contributions form individual levels.
        """
        #l_var = self._ref_level_var

        # l_var_var_scale = l_var[:, 1:] ** 2 * 2 / (self._bs_n_samples[:, None] - 1)
        # total_var_var_scale = np.sum(l_var_var_scale[:, :] / self._bs_n_samples[:, None]**2, axis=0 )


        l_cost = self.mlmc.estimate_level_cost()
        l_var, n_samples = self.mlmc.estimate_diff_vars(self.moments)
        ref_cost = np.sum(l_cost)
        ref_var = np.sum(l_var)

        l_var_scale = np.sqrt(self._bs_level_var_mean[:, :] / ref_var * l_cost[:, None] / ref_cost) / np.sqrt(self._bs_n_samples[:, None])
        #total_var_var_scale = np.sum(l_var_var_scale[:, :] / self._bs_n_samples[:, None]**2, axis=0 )

        target_var = 0.0001
        bs_var_var = np.sqrt(self._bs_var_variance[:]) / ref_var / target_var * np.sqrt(np.sum(l_var[:,:] * l_cost[:, None], axis=0))
        #bs_var_var[1:] /= total_var_var_scale


        bs_l_var_var = self._bs_level_var_variance[:, :]
        bs_l_var_var[:, 1:] = l_var_scale[:, 1:]
        avg_err = np.sum(bs_l_var_var) / self.n_levels
        print(np.sum(bs_l_var_var, axis=1) - avg_err)

        var_errors = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
        self.plot_bs_variances(var_errors, log=True,
                               y_label="BS estimate of MSE of $\hat V^r$, $\hat V^r_l$ estimators.",
                               )#y_lim=(1e-10, 1e10))


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
            mc.construct_density(tol, reg_param)

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