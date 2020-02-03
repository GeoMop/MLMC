import time
import numpy as np

import scipy.stats as st
import scipy.integrate as integrate

from typing import List
from sample_storage import SampleStorage
from sampling_pool import SamplingPool
from new_simulation import Simulation


class Sampler:

    def __init__(self, sample_storage: SampleStorage, sampling_pool: SamplingPool, sim_factory: Simulation,
                 step_range: List[float]):
        """
        :param sample_storage: store scheduled samples, results and result structure
        :param sampling_pool: calculate samples
        :param sim_factory: generate samples
        :param step_range: simulation step range
        """
        self.sample_storage = sample_storage
        self._sampling_pool = sampling_pool

        self._samples = []

        self._n_levels = len(step_range)
        self._sim_factory = sim_factory
        self._step_range = step_range

        # Number of created samples
        self._n_created_samples = np.zeros(self._n_levels)
        # Number of target samples
        self._n_target_samples = np.zeros(self._n_levels)
        self._n_finished_samples = np.zeros(self._n_levels)
        self._level_sim_objects = []
        self._create_level_sim_objects()

    @property
    def n_finished_samples(self):
        """
        Retrieve number of all finished samples
        :return:
        """
        if np.all(self._n_finished_samples) == 0:
            return self.sample_storage.n_finished()
        else:
            return self._n_finished_samples

    def _create_level_sim_objects(self):
        """
        Create LevelSimulation object for each level, use simulation factory
        :return: None
        """
        for level_id in range(self._n_levels):
            if level_id == 0:
                level_sim = self._sim_factory.level_instance([self._step_range[level_id]], [0])

            else:
                level_sim = self._sim_factory.level_instance([self._step_range[level_id]], [self._step_range[level_id-1]])

            level_sim.calculate = self._sim_factory.calculate
            level_sim.level_id = level_id
            self._level_sim_objects.append(level_sim)

    def sample_range(self, n0, nL):
        """
        Geometric sequence of L elements decreasing from n0 to nL.
        Useful to set number of samples explicitly.
        :param n0: int
        :param nL: int
        :return: np.array of length L = n_levels.
        """
        return np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), self._n_levels))).astype(int)

    def set_initial_n_samples(self, n_samples=None):
        """
        Set target number of samples for each level
        :param n_samples: array of number of samples
        :return: None
        """
        if n_samples is None:
            n_samples = [100, 3]
        # Num of samples to ndarray
        n_samples = np.atleast_1d(n_samples)

        # Just maximal number of samples is set
        if len(n_samples) == 1:
            n_samples = np.array([n_samples[0], 3])

        # Create number of samples for all levels
        if len(n_samples) == 2:
            n0, nL = n_samples
            n_samples = self.sample_range(n0, nL)

        self._n_target_samples = n_samples

    def _get_sample_tag(self, level_id):
        """
        Create sample tag
        :param level_id: identifier of current level
        :return: str
        """
        return "L{:02d}_S{:07d}".format(level_id, int(self._n_created_samples[level_id]))

    def schedule_samples(self):
        """
        Create simulation samples, loop through "levels" and its samples (given the number of target samples):
            1) generate sample tag (same for fine and coarse simulation)
            2) get LevelSimulation instance by simulation factory
            3) schedule sample via sampling pool
            4) store scheduled samples in sample storage, separately for each level
        :return: None
        """
        self.ask_sampling_pool_for_samples()
        plan_samples = self._n_target_samples - self._n_created_samples

        for level_id, n_samples in enumerate(plan_samples):
            samples = []
            for _ in range(int(n_samples)):
                # Unique sample id
                sample_id = self._get_sample_tag(level_id)
                level_sim = self._level_sim_objects[level_id]

                # Schedule current sample
                self._sampling_pool.schedule_sample(sample_id, level_sim)
                # Increment number of created samples at current level
                self._n_created_samples[level_id] += 1

                samples.append(sample_id)

                # Store scheduled samples
                self.sample_storage.save_scheduled_samples(level_id, samples)

    def ask_sampling_pool_for_samples(self, sleep=0, timeout=None):
        """
        Waiting for running simulations
        :param sleep: time for doing nothing
        :param timeout: maximum time for waiting on running simulations
        :return: int, number of running simulations
        """
        if timeout is None:
            timeout = 0
        elif timeout <= 0:
            return 1

        n_running = 1
        t0 = time.clock()
        while n_running > 0:
            successful_samples, failed_samples, n_running = self._sampling_pool.get_finished()

            for level_id, s_samples in successful_samples.items():
                self._n_finished_samples[level_id] += len(s_samples)
            for level_id, f_samples in failed_samples.items():
                self._n_finished_samples[level_id] += len(f_samples)

            # Store finished samples
            if len(successful_samples) > 0 or len(failed_samples) > 0:
                self._store_samples(successful_samples, failed_samples)

            time.sleep(sleep)
            if 0 < timeout < (time.clock() - t0):
                break

        return n_running

    def _store_samples(self, successful_samples, failed_samples):
        """
        Store finished samples
        :param successful_samples: List[Tuple[sample_id:str, Tuple[ndarray, ndarray]]]
        :param failed_samples: List[Tuple[sample_id: str, error message: str]]
        :return: None
        """
        self.sample_storage.save_samples(successful_samples, failed_samples)

    @property
    def levels_results(self):
        level_results = self.sample_storage.sample_pairs()

        # @TODO: it does not works with arrays quantities, remove ASAP
        new_level_results = []
        if level_results[0].shape[0] > 1:
            for l_res in level_results:
                new_level_results.append(l_res[0])
        else:
            new_level_results = level_results

        return new_level_results

    def target_var_adding_samples(self, target_var, moments_fn, sleep=20, add_coef=0.1):
        """
        Set level target number of samples according to improving estimates.
        We assume set_initial_n_samples method was called before.
        :param target_var: float, whole mlmc target variance
        :param moments_fn: Object providing calculating moments
        :param sleep: Sample waiting time
        :param add_coef: Coefficient for adding samples
        :return: None
        """
        # New estimation according to already finished samples
        n_estimated = self.estimate_n_samples_for_target_variance(target_var, moments_fn)
        # Loop until number of estimated samples is greater than the number of scheduled samples
        while not self.process_adding_samples(n_estimated, sleep, add_coef):
            # New estimation according to already finished samples
            n_estimated = self.estimate_n_samples_for_target_variance(target_var, moments_fn)

    def estimate_n_samples_for_target_variance(self, target_variance, moments_fn=None, prescribe_vars=None):
        """
        Estimate optimal number of samples for individual levels that should provide a target variance of
        resulting moment estimate. Number of samples are directly set to levels.
        This also set given moment functions to be used for further estimates if not specified otherwise.
        :param target_variance: Constrain to achieve this variance.
        :param moments_fn: moment evaluation functions
        :param prescribe_vars: vars[ L, M] for all levels L and moments M safe the (zeroth) constant moment with zero variance.
        :return: np.array with number of optimal samples for individual levels and moments, array (LxR)
        """
        _, n_samples_estimate_safe = self.n_sample_estimate_moments(target_variance, moments_fn, prescribe_vars)
        n_samples = np.max(n_samples_estimate_safe, axis=1).astype(int)

        return n_samples

    def process_adding_samples(self, n_estimated, sleep, add_coef=0.1):
        """
        Process adding samples
        :param n_estimated: Number of estimated samples on each level, list
        :param sleep: Sample waiting time
        :param add_coef: default value 0.1
        :return: bool, if True adding samples is complete
        """
        # Get default scheduled samples
        n_scheduled = self.l_scheduled_samples()

        # New scheduled sample will be 10 percent of difference
        # between current number of target samples and new estimated one
        # If 10 percent of estimated samples is greater than difference between estimated and scheduled samples,
        # set scheduled samples to estimated samples
        new_scheduled = np.where((n_estimated * add_coef) > (n_estimated - n_scheduled),
                                 n_estimated,
                                 n_scheduled + (n_estimated - n_scheduled) * add_coef)

        n_scheduled = np.ceil(np.where(n_estimated < n_scheduled,
                                       n_scheduled,
                                       new_scheduled))

        # Levels where estimated are greater than scheduled
        greater_items = np.where(np.greater(n_estimated, n_scheduled))[0]

        # Scheduled samples and wait until at least half of the samples are done
        self.set_scheduled_and_wait(n_scheduled, greater_items, sleep)

        return np.all(n_estimated[greater_items] == n_scheduled[greater_items])

    def set_scheduled_and_wait(self, n_scheduled, greater_items, sleep, fin_sample_coef=0.5):
        """
        Scheduled samples on each level and wait until at least half of the samples is done
        :param n_scheduled: ndarray, number of scheduled samples on each level
        :param greater_items: Items where n_estimated is greater than n_scheduled
        :param sleep: Time waiting for samples
        :param fin_sample_coef: The proportion of samples to finished for further estimate
        :return: None
        """
        # Set scheduled samples and run simulations
        self.set_level_target_n_samples(n_scheduled)
        self.schedule_samples()

        # Finished level samples
        n_finished = self.n_finished_samples

        # Wait until at least half of the scheduled samples are done on each level
        while np.any(n_finished[greater_items] < fin_sample_coef * n_scheduled[greater_items]):
            # Wait a while
            time.sleep(sleep)
            n_finished = self.n_finished_samples

    def set_level_target_n_samples(self, n_samples, fraction=1.0):
        """
        Set level number of target samples
        :param n_samples: list, each level target samples
        :param fraction: Use just fraction of total samples
        :return: None
        """
        for level, n in enumerate(n_samples):
            self._n_target_samples[level] += int(n * fraction)

    def l_scheduled_samples(self):
        """
        Get all levels target number of samples
        :return: list
        """
        return self._n_target_samples

    def n_sample_estimate_moments(self, target_variance, moments_fn, prescribe_vars=None):
        if prescribe_vars is None:
            vars = self.estimate_diff_vars_regression(moments_fn)
        else:
            vars = prescribe_vars

        # @TODO: set n ops estimate
        n_ops = np.array([lvl.task_size for lvl in self._level_sim_objects])

        sqrt_var_n = np.sqrt(vars.T * n_ops)  # moments in rows, levels in cols
        total = np.sum(sqrt_var_n, axis=1)  # sum over levels
        n_samples_estimate = np.round((sqrt_var_n / n_ops).T * total / target_variance).astype(int)  # moments in cols
        # Limit maximal number of samples per level
        n_samples_estimate_safe = np.maximum(np.minimum(n_samples_estimate, vars * self._n_levels / target_variance), 2)
        return n_samples_estimate, n_samples_estimate_safe

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
        sim_steps = self._step_range
        vars = self._all_moments_variance_regression(raw_vars, sim_steps)
        return vars

    def _all_moments_variance_regression(self, raw_vars, sim_steps):
        reg_vars = raw_vars.copy()
        n_moments = raw_vars.shape[1]
        for m in range(1, n_moments):
            reg_vars[:, m] = self._moment_variance_regression(raw_vars[:, m], sim_steps)
        assert np.allclose(reg_vars[:, 0], 0.0)
        return reg_vars

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

    def _variance_of_variance(self, n_samples = None):
        """
        Approximate variance of log(X) where
        X is from ch-squared with df=n_samples - 1.
        Return array of variances for actual n_samples array.

        :param n_samples: Optional array with n_samples.
        :return: array of variances of variance estimate.
        """
        if n_samples is None:
            n_samples = self._n_created_samples
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

    def estimate_moments(self, moments_fn):
        """
        Use collected samples to estimate moments and variance of this estimate.
        :param moments_fn: Vector moment function, gives vector of moments for given sample or sample vector.
        :return: estimate_of_moment_means, estimate_of_variance_of_estimate ; arrays of length n_moments
        """
        means = []
        vars = []
        n_samples = []

        for level_id, level_result in enumerate(self.levels_results):

            zero_level = True if level_id == 0 else False
            means.append(self.estimate_diff_mean(moments_fn, level_result, zero_level))
            l_vars, ns = self.estimate_diff_var(moments_fn, level_result, zero_level)
            vars.append(l_vars)
            n_samples.append(ns)
        means = np.sum(np.array(means), axis=0)
        n_samples = np.array(n_samples, dtype=int)
        vars = np.sum(np.array(vars) / n_samples[:, None], axis=0)

        return np.array(means), np.array(vars)

    def estimate_diff_vars(self, moments_fn=None):
        """
        Estimate moments variance from samples
        :param moments_fn: Moment evaluation functions
        :return: (diff_variance, n_samples);
            diff_variance - shape LxR, variances of diffs of moments
            n_samples -  shape L, num samples for individual levels.

            Returns simple variance for level 0.
        """
        vars = []
        n_samples = []

        for level, level_results in enumerate(self.levels_results):
            zero_level = True if level == 0 else False
            v, n = self.estimate_diff_var(moments_fn, level_results, zero_level)
            vars.append(v)
            n_samples.append(n)
        return np.array(vars), np.array(n_samples)

    ################################################
    # Level methods

    def estimate_diff_var(self, moments_fn, level_results, zero_level=False):
        """
        Estimate moments variance
        :param moments_fn: Moments evaluation function
        :return: tuple (variance vector, length of moments)
        """

        mom_fine, mom_coarse = self.evaluate_moments(moments_fn, level_results, zero_level)
        assert len(mom_fine) == len(mom_coarse)
        assert len(mom_fine) >= 2
        var_vec = np.var(mom_fine - mom_coarse, axis=0, ddof=1)
        ns = level_results.shape[1]
        return var_vec, ns

    def estimate_diff_mean(self, moments_fn, level_result, zero_level=False):
        """
        Estimate moments mean
        :param moments_fn: Function for calculating moments
        :return: np.array, moments mean vector
        """
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn, level_result, zero_level)
        assert len(mom_fine) == len(mom_coarse)
        assert len(mom_fine) >= 1
        mean_vec = np.mean(mom_fine - mom_coarse, axis=0)
        return mean_vec

    # def estimate_covariance(self, moments_fn, stable=False):
    #     """
    #     Estimate covariance matrix (non central).
    #     :param moments_fn: Moment functions object.
    #     :param stable: Use alternative formula with better numerical stability.
    #     :return: cov covariance matrix  with shape (n_moments, n_moments)
    #     """
    #     mom_fine, mom_coarse = self.evaluate_moments(moments_fn)
    #     assert len(mom_fine) == len(mom_coarse)e
    #     assert len(mom_fine) >= 2
    #     assert self.n_samples == len(mom_fine)
    #
    #     if stable:
    #         # Stable formula - however seems that we have no problem with numerical stability
    #         mom_diff = mom_fine - mom_coarse
    #         mom_sum = mom_fine + mom_coarse
    #         cov = 0.5 * (np.matmul(mom_diff.T, mom_sum) + np.matmul(mom_sum.T, mom_diff)) / self.n_samples
    #     else:
    #         # Direct formula
    #         cov_fine = np.matmul(mom_fine.T,   mom_fine)
    #         cov_coarse = np.matmul(mom_coarse.T, mom_coarse)
    #         cov = (cov_fine - cov_coarse) / self.n_samples
    #
    #     return cov

    def evaluate_moments(self, moments_fn, level_results, is_zero_level=False):
        """
        Evaluate level difference for all samples and given moments.
        :param moments_fn: Moment evaluation object.
        :param force: Reevaluate moments
        :return: (fine, coarse) both of shape (n_samples, n_moments)
        """
        # Current moment functions are different from last moment functions
        samples = np.squeeze(level_results)

        # Moments from fine samples
        moments_fine = moments_fn(samples[:, 0])

        # For first level moments from coarse samples are zeroes
        if is_zero_level:
            moments_coarse = np.zeros((len(moments_fine), moments_fn.size))
        else:
            moments_coarse = moments_fn(samples[:, 1])
        # Set last moments function
        self._last_moments_fn = moments_fn
        # Moments from fine and coarse samples
        self.last_moments_eval = moments_fine, moments_coarse

        self._remove_outliers_moments()
        return self.last_moments_eval

    def _remove_outliers_moments(self, ):
        """
        Remove moments from outliers from fine and course moments
        :return: None
        """
        # Fine and coarse moments mask
        ok_fine = np.all(np.isfinite(self.last_moments_eval[0]), axis=1)
        ok_coarse = np.all(np.isfinite(self.last_moments_eval[1]), axis=1)

        # Common mask for coarse and fine
        ok_fine_coarse = np.logical_and(ok_fine, ok_coarse)

        # New moments without outliers
        self.last_moments_eval = self.last_moments_eval[0][ok_fine_coarse, :], self.last_moments_eval[1][ok_fine_coarse, :]

    ###################################################
