import mlmc
import numpy as np
import scipy.stats as st
import scipy.integrate as integrate
from mlmc.tool import simple_distribution
from mlmc.tool import plot
import matplotlib.pyplot as plt


def compute_results(mlmc_l0, n_moments, mlmc_wrapper):
    """
    Compute density and moments domains
    TODO: remove completely or move into test_distribution
    :param mlmc_l0: One level Monte-Carlo method
    :param n_moments: int, Number of moments
    :param mc_wrapper: Object with mlmc instance, must contains also moments function object
    :return: domain - tuple, domain from 1LMC
             est_domain - tuple, domain estimated by mlmc instance
             mc_wrapper - with current distr_obj (object estimating distribution)
    """
    mlmc = mlmc_wrapper.mc
    estimator = Estimate(mlmc)
    moments_fn = mlmc_wrapper.moments_fn
    domain = mlmc_l0.ref_domain
    est_domain = estimator.estimate_domain()

    t_var = 1e-5
    ref_diff_vars, _ = estimator.estimate_diff_vars(moments_fn)
    # ref_moments, ref_vars = mc.estimate_moments(moments_fn)
    # ref_std = np.sqrt(ref_vars)
    # ref_diff_vars_max = np.max(ref_diff_vars, axis=1)
    # ref_n_samples = mc.set_target_variance(t_var, prescribe_vars=ref_diff_vars)
    # ref_n_samples = np.max(ref_n_samples, axis=1)
    # ref_cost = mc.estimate_cost(n_samples=ref_n_samples)
    # ref_total_std = np.sqrt(np.sum(ref_diff_vars / ref_n_samples[:, None]) / n_moments)
    # ref_total_std_x = np.sqrt(np.mean(ref_vars))

    est_moments, est_vars = estimator.estimate_moments(moments_fn)

    # def describe(arr):
    #     print("arr ", arr)
    #     q1, q3 = np.percentile(arr, [25, 75])
    #     print("q1 ", q1)
    #     print("q2 ", q3)
    #     return "{:f8.2} < {:f8.2} | {:f8.2} | {:f8.2} < {:f8.2}".format(
    #         np.min(arr), q1, np.mean(arr), q3, np.max(arr))

    moments_data = np.stack((est_moments, est_vars), axis=1)

    distr_obj = simple_distribution.SimpleDistribution(moments_fn, moments_data)
    distr_obj.domain = domain
    distr_obj.estimate_density_minimize(1)
    mlmc_wrapper.distr_obj = distr_obj

    return domain, est_domain, mlmc_wrapper


class Estimate:
    """
    Base of future class dedicated to all kind of processing of the collected samples
    MLMC should only collect the samples.

    TODO: try to move plotting methods into separate file, allowing independent usage of the plots for
    explicitely provided datasets.
    """
    def __init__(self, mlmc, moments=None):
        self.mlmc = mlmc
        self.moments = moments

        self.moments_2_integral = None
        self.cov_mat = None

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

    def estimate_level_vars(self, moments_fn=None):
        """
        Estimate variances for moments of X approximations on individual levels.
        i.t. Var \phi_r( X^l ).
        :param moments_fn:
        :return:
        """
        if moments_fn is None:
            moments_fn = self.moments

        sim_steps = self.sim_steps
        #n_samples = mlmc.n_samples
        vars = []
        steps = []
        for il, level in enumerate(self.levels):
            var_coarse, var_fine = level.estimate_level_var(moments_fn)
            if il > 0:
                vars.append(var_coarse)
                steps.append(sim_steps[il-1])
            vars.append(var_fine)
            steps.append(sim_steps[il])
        return np.array(steps), np.array(vars)

    def estimate_diff_vars_regression(self, moments_fn=None, raw_vars=None):
        """
        Estimate variances using linear regression model.
        Assumes increasing variance with moments, use only two moments with highest average variance.
        :param moments_fn: Moment evaluation function
        :return: array of variances, shape  L
        """
        # vars shape L x R
        if raw_vars is None:
            assert moments_fn is not None
            raw_vars, n_samples = self.estimate_diff_vars(moments_fn)

        raw_vars = np.squeeze(raw_vars)
        sim_steps = self.sim_steps
        #vars = self._varinace_regression(raw_vars, sim_steps)
        vars = self._all_moments_variance_regression(raw_vars, sim_steps)
        return vars

    def _variance_regression(self, raw_vars, sim_steps):
        """
        Estimate level variance by regression from model:

        log(var_l,r) = A_r + B * log(h_l) + C * log^2(hl),
                                            for l = 0, .. L-1


        :param raw_vars: moments variances raws, shape (L, R)
        :param sim_steps: simulation steps, shape L
        :return: np.array  (L, R)
        """
        L, R = raw_vars.shape
        L1 = L - 1
        if L < 3:
            return raw_vars

        # estimate of variances of variances, compute scaling
        W = 1.0 / np.sqrt(self._variance_of_variance())
        W = W[1:]  # ignore level 0
        # W = np.ones((L - 1,))

        # Use linear regresion to improve estimate of variances V1, ...
        # model log var_{r,l} = a_r  + b * log step_l
        # X_(r,l), j = dirac_{r,j}

        K = R + 1  # number of parameters
        R1 = R - 1
        X = np.zeros((L1, R1, K))
        X[:, :, :-2] = np.eye(R1)[None, :, :]
        log_step = np.log(sim_steps[1:])
        # X[:, :, -1] = np.repeat(log_step ** 2, R1).reshape((L1, R1))[:, :, None] * np.eye(R1)[None, :, :]
        X[:, :, -2] = np.repeat(log_step ** 2, R1).reshape((L1, R1))
        X[:, :, -1] = np.repeat(log_step, R1).reshape((L1, R1))

        WX = X * W[:, None, None]  # scale
        WX.shape = (-1, K)
        X.shape = (-1, K)
        # solve X.T * X = X.T * V

        log_vars = np.log(raw_vars[1:, 1:])  # omit first variance, and first moment that is constant 1.0
        log_vars = W[:, None] * log_vars  # scale RHS

        params, res, rank, sing_vals = np.linalg.lstsq(WX, log_vars.ravel())
        new_vars = raw_vars.copy()
        assert np.allclose(raw_vars[:, 0], 0.0)
        new_vars[1:, 1:] = np.exp(np.dot(X, params)).reshape(L - 1, -1)
        return new_vars

    def _variance_of_variance(self, n_samples = None):
        """
        Approximate variance of log(X) where
        X is from ch-squared with df=n_samples - 1.
        Return array of variances for actual n_samples array.

        :param n_samples: Optional array with n_samples.
        :return: array of variances of variance estimate.
        """
        if n_samples is None:
            n_samples = self.n_samples
        if hasattr(self, "_saved_var_var"):
            ns, var_var = self._saved_var_var
            if np.sum(np.abs(ns - n_samples)) == 0:
                return var_var

        vars = []
        for ns in n_samples:
            df = ns - 1

            def log_chi_pdf(x):
                return np.exp(x) * df * st.chi2.pdf(np.exp(x) * df, df=df)

            def compute_moment(moment):
                std_est = np.sqrt(2 / df)
                fn = lambda x, m=moment: x ** m * log_chi_pdf(x)
                return integrate.quad(fn, -100 * std_est, 100 * std_est)[0]

            mean = compute_moment(1)
            second = compute_moment(2)
            vars.append(second - mean ** 2)

        self._saved_var_var = (n_samples, np.array(vars))
        return np.array(vars)

    def _moment_variance_regression(self, raw_vars, sim_steps):
        """
        Estimate level variance using separate model for every moment.

        log(var_l) = A + B * log(h_l) + C * log^2(hl),
                                            for l = 0, .. L-1
        :param raw_vars: moments variances raws, shape (L,)
        :param sim_steps: simulation steps, shape (L,)
        :return: np.array  (L, )
        """
        L, = raw_vars.shape

        L1 = L - 1
        if L < 3:
            return raw_vars

        # estimate of variances of variances, compute scaling
        W = 1.0 / np.sqrt(self._variance_of_variance())
        W = W[1:]   # ignore level 0
        W = np.ones((L - 1,))

        # Use linear regresion to improve estimate of variances V1, ...
        # model log var_{r,l} = a_r  + b * log step_l
        # X_(r,l), j = dirac_{r,j}

        K = 3 # number of parameters

        X = np.zeros((L1, K))
        log_step = np.log(sim_steps[1:])
        X[:, 0] = np.ones(L1)
        X[:, 1] = np.full(L1, log_step)
        X[:, 2] = np.full(L1, log_step ** 2)

        WX = X * W[:, None]    # scale

        log_vars = np.log(raw_vars[1:])     # omit first variance
        log_vars = W * log_vars       # scale RHS

        params, res, rank, sing_vals = np.linalg.lstsq(WX, log_vars)
        new_vars = raw_vars.copy()

        new_vars[1:] = np.exp(np.dot(X, params))
        return new_vars

    def _all_moments_variance_regression(self, raw_vars, sim_steps):
        reg_vars = raw_vars.copy()
        if len(raw_vars.shape) == 1:
            raw_vars = np.array([[raw_vars]])
        n_moments = raw_vars.shape[1]

        for m in range(1, n_moments):
            reg_vars[:, m] = self._moment_variance_regression(raw_vars[:, m], sim_steps)

        if len(reg_vars.shape) == 1:
            reg_vars = np.array([reg_vars])

        assert np.allclose(reg_vars[:, 0], 0.0)
        return reg_vars

    def estimate_diff_vars(self, moments_fn=None):
        """
        Estimate moments variance from samples
        :param moments_fn: Moment evaluation functions
        :return: (diff_variance, n_samples);
            diff_variance - shape LxR, variances of diffs of moments
            n_samples -  shape L, num samples for individual levels.

            Returns simple variance for level 0.
        """
        if moments_fn is None:
            moments_fn = self.moments
        vars = []
        n_samples = []

        for level in self.levels:
            v, n = level.estimate_diff_var(moments_fn)
            vars.append(v)
            n_samples.append(n)

        return np.array(vars), np.array(n_samples)

    def estimate_level_means(self, moments_fn):
        """
        Estimate means on individual levels.
        :param moments_fn: moments object of size R
        :return: shape (L, R)
        """
        means = []
        for level in self.mlmc.levels:
            means.append(level.estimate_diff_mean(moments_fn))
        return np.array(means)

    def estimate_n_samples_for_target_variance(self, target_variance, moments_fn=None, prescribe_vars=None):
        """
        Estimate optimal number of samples for individual levels that should provide a target variance of
        resulting moment estimate. Number of samples are directly set to levels.
        This also set given moment functions to be used for further estimates if not specified otherwise.
        TODO: separate target_variance per moment
        :param target_variance: Constrain to achieve this variance.
        :param moments_fn: moment evaluation functions
        :param prescribe_vars: vars[ L, M] for all levels L and moments M safe the (zeroth) constant moment with zero variance.
        :return: np.array with number of optimal samples for individual levels and moments, array (LxR)
        """
        _, n_samples_estimate_safe = self.n_sample_estimate_moments(target_variance, moments_fn, prescribe_vars)
        n_samples = np.max(n_samples_estimate_safe, axis=1).astype(int)

        return n_samples

    def n_sample_estimate_moments(self, target_variance, moments_fn=None, prescribe_vars=None):
        if moments_fn is None:
            moments_fn = self.moments
        if prescribe_vars is None:
            vars = self.estimate_diff_vars_regression(moments_fn)
        else:
            vars = prescribe_vars

        n_ops = np.array([lvl.n_ops_estimate for lvl in self.levels])

        sqrt_var_n = np.sqrt(vars.T * n_ops)  # moments in rows, levels in cols
        total = np.sum(sqrt_var_n, axis=1)  # sum over levels
        n_samples_estimate = np.round((sqrt_var_n / n_ops).T * total / target_variance).astype(int)  # moments in cols

        # Limit maximal number of samples per level
        n_samples_estimate_safe = np.maximum(np.minimum(n_samples_estimate, vars * self.n_levels / target_variance), 2)

        return n_samples_estimate, n_samples_estimate_safe

    def target_var_adding_samples(self, target_var, moments_fn, pbs=None, sleep=20, add_coef=0.1):
        """
        Set level target number of samples according to improving estimates.
        We assume set_initial_n_samples method was called before.
        :param target_var: float, whole mlmc target variance
        :param moments_fn: Object providing calculating moments
        :param pbs: Pbs script generator object
        :param sleep: Sample waiting time
        :param add_coef: Coefficient for adding samples
        :return: None
        """
        # New estimation according to already finished samples
        n_estimated = self.estimate_n_samples_for_target_variance(target_var, moments_fn)

        # Loop until number of estimated samples is greater than the number of scheduled samples
        while not self.mlmc.process_adding_samples(n_estimated, pbs, sleep, add_coef):
            # New estimation according to already finished samples
            n_estimated = self.estimate_n_samples_for_target_variance(target_var, moments_fn)

    def estimate_moments(self, moments_fn):
        """
        Use collected samples to estimate moments and variance of this estimate.
        :param moments_fn: Vector moment function, gives vector of moments for given sample or sample vector.
        :return: estimate_of_moment_means, estimate_of_variance_of_estimate ; arrays of length n_moments
        """
        means = []
        vars = []
        n_samples = []
        for level in self.levels:
            means.append(level.estimate_diff_mean(moments_fn))
            l_vars, ns = level.estimate_diff_var(moments_fn)
            vars.append(l_vars)
            n_samples.append(ns)

        means = np.sum(np.array(means), axis=0)
        n_samples = np.array(n_samples, dtype=int)
        vars = np.sum(vars / n_samples[:, None, None], axis=0)

        return np.array(means), np.array(vars)

    def estimate_level_cost(self):
        """
        For every level estimate of cost of evaluation of a single coarse-fine simulation pair.
        TODO: Estimate simulation cost from collected times + regression similar to variance
        :return:
        """
        return np.array([lvl.n_ops_estimate for lvl in self.mlmc.levels])

    def estimate_cost(self, level_times=None, n_samples=None):
        """
        Estimate total cost of mlmc
        :param level_times: Cost estimate for single simulation for every level.
        :param n_samples: Number of samples on each level
        :return: total cost
        """
        if level_times is None:
            level_times = self.estimate_level_cost()
        if n_samples is None:
            n_samples = self.mlmc.n_samples

        return np.sum(level_times * n_samples)

    def estimate_covariance(self, moments_fn, levels, stable=False, mse=False):
        """
        MLMC estimate of covariance matrix of moments.
        :param stable: use formula with better numerical stability
        :param mse: Mean squared error
        :return:
        """
        if self.cov_mat is None:

            cov_mat = np.zeros((moments_fn.size, moments_fn.size))

            for level in levels:
                cov_mat += level.estimate_covariance(moments_fn, stable)
            if mse:
                mse_diag = np.zeros(moments_fn.size)
                for level in levels:
                    mse_diag += level.estimate_cov_diag_err(moments_fn)/level.n_samples
                self.cov_mat = cov_mat
                return cov_mat, mse_diag
            else:
                self.cov_mat = cov_mat
                return cov_mat

        return self.cov_mat

    def quad_regularization(self, tol):
        a, b = self.moments.domain
        m = self.moments.size - 1
        gauss_degree = 150

        integral = np.zeros((self.moments.size, self.moments.size))
        for i in range(self.moments.size):
            for j in range(i + 1):
                def fn_moments(x):
                    all_moments = self.moments.eval_all_der(x, degree=2)
                    return all_moments[:, i] * all_moments[:, j]

                [x, w] = np.polynomial.legendre.leggauss(gauss_degree)
                x = (x[None, :] + 1) / 2 * (b - a) + a
                w = w[None, :] * 0.5 * (b - a)
                x = x.flatten()
                w = w.flatten()
                integ = (np.sum(w * fn_moments(x)))
                #integ = integrate.quad(fn_moments, self.moments.domain[0], self.moments.domain[1], epsabs=tol)[0]
                integral[i][j] = integral[j][i] = integ

        return integral

    def regularization(self, tol):
        """

        Args:
            tol:

        Returns:

        """
        integral = np.zeros((self.moments.size, self.moments.size))
        for i in range(self.moments.size):
            for j in range(i + 1):
                def fn_moments(x):
                    moments = self.moments.eval_all_der(x, degree=2)[0, :]
                    return moments[i] * moments[j]

                integ = integrate.quad(fn_moments, self.moments.domain[0], self.moments.domain[1], epsabs=tol)[0]
                integral[i][j] = integral[j][i] = integ
        return integral

    def construct_density(self, tol=1.95, reg_param=1e-7*5, orth_moments_tol=1e-2, exact_pdf=None, orth_method=2):
        """
        Construct approximation of the density using given moment functions.
        Args:
            moments_fn: Moments object, determines also domain and n_moments.
            tol: Tolerance of the fitting problem, with account for variances in moments.
                 Default value 1.95 corresponds to the two tail confidency 0.95.
            reg_param: Regularization parameter.
        """
        import pandas as pd
        cov = self.estimate_covariance(self.moments, self.mlmc.levels)
        # print("cov")
        # print(pd.DataFrame(cov))
        reg_term = np.zeros(cov.shape)
        if reg_param != 0:
            reg_term = self.quad_regularization(tol)
            #reg_term = self.regularization(tol)
            print("reg term ", reg_term)

        cov += 2 * reg_param * reg_term

        moments_obj, info, cov_centered = simple_distribution.construct_orthogonal_moments(self.moments, cov,
                                                                                           tol=orth_moments_tol,
                                                                                            orth_method=orth_method
                                                                                        )
        print("n levels: ", self.n_levels, "size: ", moments_obj.size)

        est_moments, est_vars = self.estimate_moments(moments_obj)
        est_moments = np.squeeze(est_moments)
        est_vars = np.squeeze(est_vars)
        exact_moments = mlmc.simple_distribution.compute_exact_moments(moments_obj, exact_pdf)


        from src.mlmc.moments import TransformedMomentsDerivative
        moments_obj_derivative = TransformedMomentsDerivative(moments_obj._origin, moments_obj._transform)

        samples = self.mlmc.levels[0].sample_values[:, 0]
        moments = np.squeeze(moments_obj_derivative(samples))

        var_vec = []
        for i in range(len(moments[0])):
            mask = np.isfinite(moments[:, i])
            values = moments[:, i][mask]
            var_vec.append(np.var(values, axis=0, ddof=1)/len(values))

        #der_moments, der_vars = self.estimate_moments(moments_obj)

        #est_moments = np.zeros(moments_obj.size)
        est_moments[0] = 1.0
        #est_vars[0] = 1
        est_vars = np.ones(moments_obj.size)
        min_var, max_var = np.min(est_vars[1:]), np.max(est_vars[1:])
        print("min_err: {} max_err: {} ratio: {}".format(min_var, max_var, max_var / min_var))
        moments_data = np.stack((est_moments, est_vars), axis=1)

        m = np.zeros(len(exact_moments))
        m[0] = 1

        # moments_data = np.empty((len(exact_moments), 2))
        # moments_data[:, 0] = exact_moments
        moments_data[:, 1] = 1.0

        regularization = mlmc.simple_distribution.Regularization2ndDerivation()
        distr_obj = simple_distribution.SimpleDistribution(moments_obj, moments_data, domain=moments_obj.domain,
                                                           reg_param=reg_param, regularization=regularization)
        result = distr_obj.estimate_density_minimize(tol)  # 0.95 two side quantile
        self._distribution = distr_obj

        return info, result

    def _bs_get_estimates(self):
        moments_fn = self.moments
        #mean_est, var_est = self.mlmc.estimate_moments(moments_fn)
        level_var_est, _ = self.estimate_diff_vars(moments_fn)
        level_mean_est = self.estimate_level_means(moments_fn)
        return level_mean_est, level_var_est

    def _bs_get_estimates_regression(self):
        moments_fn = self.moments
        #mean_est, var_est = self.mlmc.estimate_moments(moments_fn)
        level_var_est, _ = self.estimate_diff_vars(moments_fn)
        level_mean_est = self.estimate_level_means(moments_fn)
        level_var_est = self.estimate_diff_vars_regression(moments_fn, level_var_est)
        #var_est = np.sum(level_var_est[:, :]/self.n_samples[:,  None], axis=0)
        return level_mean_est, level_var_est

    def check_bias(self, a, b, var, label):
        diff = np.abs(a - b)
        tol = 2*np.sqrt(var) + 1e-20
        if np.any(diff > tol):
            print("Bias ", label)
            it = np.nditer(diff, flags=['multi_index'])
            while not it.finished:
                midx = it.multi_index
                sign = "<" if diff[midx] < tol[midx] else ">"
                print("{:6} {:8.2g} {} {:8.2g} {:8.3g} {:8.3g}".format(
                    str(midx), diff[midx], sign, tol[midx], a[midx], b[midx]))
                it.iternext()

    def ref_estimates_bootstrap(self, n_subsamples=100, sample_vector=None, regression=False, log=False, moments_fn=None):
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
        if moments_fn is not None:
            self.moments = moments_fn
        else:
            moments_fn = self.moments

        if sample_vector is None:
            sample_vector = self.mlmc.n_samples
        if len(sample_vector) > self.n_levels:
            sample_vector = sample_vector[:self.n_levels]
        sample_vector = np.array(sample_vector)

        level_estimate_fn = self._bs_get_estimates
        if regression:
            level_estimate_fn = self._bs_get_estimates_regression

        def _estimate_fn():
            lm, lv =level_estimate_fn()
            return (np.sum(lm, axis=0), np.sum(lv[:, :] / sample_vector[:, None], axis=0), lm, lv)
        if log:
            def estimate_fn():
                (m, v, lm, lv) = _estimate_fn()
                return (m, np.log(np.maximum(v, 1e-10)), lm, np.log(np.maximum(lv, 1e-10)))
        else:
            estimate_fn = _estimate_fn

        self.mlmc.update_moments(moments_fn)
        estimates = estimate_fn()

        est_samples = [np.zeros(n_subsamples) * est[..., None] for est in estimates]
        for i in range(n_subsamples):
            self.mlmc.subsample(sample_vector)
            sub_estimates = estimate_fn()
            for es, se in zip(est_samples, sub_estimates):
                es[..., i] = se

        bs_mean_est = [np.mean(est, axis=-1) for est in est_samples]
        bs_err_est = [np.var(est, axis=-1, ddof=1) for est in est_samples]
        mvar, vvar, lmvar, lvvar = bs_err_est
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

        # Check bias
        mmean, vmean, lmmean, lvmean = bs_mean_est
        self.check_bias(m,   mmean, mvar,   "Mean")
        self.check_bias(v,   vmean, vvar,   "Variance")
        self.check_bias(lm, lmmean, lmvar,  "Level Mean")
        self.check_bias(lv, lvmean, lvvar,  "Level Varaince")

        self.mlmc.clean_subsamples()

        return mmean, vmean

    def estimate_exact_mean(self, distr, moments_fn, size=200000):
        """
        Calculate exact means using MC method.
        :param distr: Distribution object
        :param moments_fn: Moments object
        :param size: Number of samples
        :return: Mean of moments function
        """
        X = distr.rvs(size=size)
        return np.nanmean(moments_fn(X), axis=0)

    def estimate_diff_var(self, levels_sims, distr, moments_fn, size=10000):
        """
        Calculate variances of level differences using MC method.
        :param levels_sims: Levels simulation objects
        :param moments_fn: Moments function
        :param distr: Distribution obj
        :param size: Number of samples from distribution
        :return: means, vars ; shape = (n_levels, n_moments)
        """
        means = []
        vars = []
        sim_l = None
        # Loop through levels simulation objects
        for l in range(len(levels_sims)):
            # Previous level simulations
            sim_0 = sim_l
            # Current level simulations
            sim_l = lambda x, h=levels_sims[l].step: levels_sims[l]._sample_fn(x, h)
            # Samples from distribution
            X = distr.rvs(size=size)
            if l == 0:
                MD = moments_fn(sim_l(X))
            else:
                MD = (moments_fn(sim_l(X)) - moments_fn(sim_0(X)))

            moment_means = np.nanmean(MD, axis=0)
            moment_vars = np.nanvar(MD, axis=0, ddof=1)
            means.append(moment_means)
            vars.append(moment_vars)
        return np.array(means), np.array(vars)

    def direct_estimate_diff_var(self, level_sims, distr, moments_fn):
        """
        Used in mlmc_test_run
        Calculate variances of level differences using numerical quadrature.
        :param moments_fn:
        :return:
        """
        mom_domain = moments_fn.domain

        means = []
        vars = []
        sim_l = None
        for l in range(len(level_sims)):
            # TODO: determine integration domain as _sample_fn ^{-1} (  mom_domain )
            domain = mom_domain

            sim_0 = sim_l
            sim_l = lambda x, h=level_sims[l].step: level_sims[l]._sample_fn(x, h)
            if l == 0:
                md_fn = lambda x: moments_fn(sim_l(x))
            else:
                md_fn = lambda x: moments_fn(sim_l(x)) - moments_fn(sim_0(x))
            fn = lambda x: (md_fn(x)).T * distr.pdf(x)
            moment_means = integrate.fixed_quad(fn, domain[0], domain[1], n=100)[0]
            fn2 = lambda x: ((md_fn(x) - moment_means[None, :]) ** 2).T * distr.pdf(x)
            moment_vars = integrate.fixed_quad(fn2, domain[0], domain[1], n=100)[0]
            means.append(moment_means)
            vars.append(moment_vars)
        return means, vars

    @classmethod
    def estimate_domain(cls, mlmc, quantile=None):
        """
        Estimate density domain from MLMC samples.
        :parameter mlmc: MLMC object that provides the samples
        :parameter quantile: float in interval (0, 1), None means whole sample range.
        :return: lower_bound, upper_bound
        """
        ranges = np.array([l.sample_domain(quantile) for l in mlmc.levels])
        return np.min(ranges[:, 0]), np.max(ranges[:, 1])

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
        print("variances shape ", variances.shape)
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
        print("self._bs_n_samples ", self._bs_n_samples)
        bs_l_var_var = np.sqrt((self._bs_level_var_variance[:, :]) * vv[:, None])
        print("bs_l_var_var.shape ", bs_l_var_var.shape)
        bs_var_var = self._bs_var_variance[:]  # - np.log(total_var_var_scale)
        print("bs_var_var.shape ", bs_var_var.shape)
        bs_variances = np.concatenate((bs_var_var[None, :], bs_l_var_var[:, :]), axis=0)
        print("bs_variances.shape ", bs_variances.shape)
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
        Re-create new Estimate objects from same original MLMC list.
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
        self.domain = kwargs.get('domain', self.common_domain())

        self._moments = self.moment_class(self.n_moments, self.domain, self.log_scale)

        self.mlmc = [Estimate(mc, self._moments) for mc in self._mlmc_list]
        self.mlmc_dict = {mc.n_levels: mc for mc in self.mlmc}
        self._moments_params = None

    def common_domain(self):
        L = +np.inf
        U = -np.inf
        for mc in self._mlmc_list:
            l, u = Estimate.estimate_domain(mc)
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
        print("\n        |     Average sample time")
        for mlmc in self.mlmc:
            samples_tabs = ["{:8}".format(n) for n in mlmc.n_samples]
            times_tabs = ["{:8.2f}s".format(t) for t in mlmc.mlmc.get_sample_times()]
            print("{:7} | {}".format(mlmc.n_levels, " ".join(samples_tabs)))
            print("{:7} | {}".format(mlmc.n_levels, " ".join(times_tabs)))
        print("\n")

    def set_common_domain(self, i_mlmc, domain=None):
        if domain is not None:
            self.domain = domain
        self.domain = Estimate.estimate_domain(self._mlmc_list[i_mlmc])

    def plot_means(self, moments_fn):
        pass

    def construct_densities(self, tol=1.95, reg_param=0.01):
        for mc_est in self.mlmc:
            mc_est.construct_density(tol, reg_param)

    def plot_densities(self, i_sample_mlmc=0):
        """
        Plot constructed densities (see construct densities)
        Args:
            i_sample_mlmc: Index of MLMC used to construct histogram and ecdf.

        Returns:
        """
        distr_plot = plot.Distribution(title="Approx. density", quantity_name=self.quantity_name, legend_title="Number of levels",
                 log_density=False, cdf_plot=True, log_x=True, error_plot='kl')

        if i_sample_mlmc is not None:
            mc0_samples = np.concatenate(self.mlmc[i_sample_mlmc].levels[0].sample_values[:, 0])
            distr_plot.add_raw_samples(mc0_samples)

        for mc in self.mlmc:
            if mc._distribution is None:
                continue
            label = "L = {}".format(mc.n_levels)
            distr_plot.add_distribution(mc._distribution, label=label)

        distr_plot.show('compare_distributions.pdf')

    def plot_variances(self):
        var_plot = plot.VarianceBreakdown(5)
        for mc in self.mlmc:
            #sample_vec = [5000, 5000, 1700, 600, 210, 72, 25, 9, 3]
            sample_vec = mc.estimate_n_samples_for_target_variance(0.0001)

            mc.ref_estimates_bootstrap(300, sample_vector=sample_vec)
            #sample_vec = [10000, 10000, 3000, 1200, 400, 140, 50, 18, 6]
            mc.mlmc.subsample(sample_vec)
            mc.mlmc.update_moments(self.moments)

            vars, n_samples = mc.estimate_diff_vars()
            var_plot.add_variances(vars, n_samples, ref_level_vars=mc._bs_level_mean_variance)
        var_plot.show()

    def plot_level_variances(self):
        var_plot = plot.Variance(5)
        for mc in self.mlmc:
            steps, vars = mc.estimate_level_vars()
            var_plot.add_level_variances(steps, vars)
        var_plot.show()

    def ref_estimates_bootstrap(self, n_samples, sample_vector=None):
        for mc in self.mlmc:
            mc.ref_estimates_bootstrap(self.moments, n_subsamples=n_samples, sample_vector=sample_vector)

    def plot_var_compare(self, nl):
        self[nl].plot_bootstrap_variance_compare(self.moments)

    def plot_var_var(self, nl):
        self[nl].plot_bootstrap_var_var(self.moments)
