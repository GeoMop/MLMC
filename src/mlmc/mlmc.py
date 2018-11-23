import time
import numpy as np
from mlmc.mc_level import Level
import scipy.stats as st
import scipy.integrate as integrate
import mlmc.hdf as hdf


class MLMC:
    """
    Multilevel Monte Carlo method
    """

    def __init__(self, n_levels, sim_factory, step_range, process_options):
        """
        :param n_levels: Number of levels
        :param sim_factory: Object of simulation
        :param step_range: Simulations step range
        :param process_options: Options for processing mlmc samples
                                'output_dir' - directory with sample logs
                                'regen_failed' - bool, if True then failed simulations are generated again
                                'keep_collected' - bool, if True then dirs with finished simulations aren't removed
        """
        # Object of simulation
        self.simulation_factory = sim_factory
        # Array of level objects
        self.levels = []
        self._n_levels = n_levels
        self.step_range = step_range

        self._process_options = process_options
        # Number of simulation steps through whole mlmc
        self.target_time = None
        # Total variance
        self.target_variance = None

        # Create hdf5 file - contains metadata and samples at levels
        self._hdf_object = hdf.HDF5(file_name="mlmc_{}.hdf5".format(n_levels), work_dir=self._process_options['output_dir'])

    def load_from_file(self):
        """
        Run mlmc according to setup parameters, load setup from hdf file {n_levels, step_range} and create levels
        :return: None
        """
        # Load mlmc params from file
        self._hdf_object.load_from_file()

        self._n_levels = self._hdf_object.n_levels
        self.step_range = self._hdf_object.step_range

        # Create mlmc levels
        self.create_levels()

    def create_new_execution(self):
        """
        Save mlmc main attributes {n_levels, step_range} and create levels
        :return: None
        """
        self._hdf_object.clear_groups()
        self._hdf_object.init_header(step_range=self.step_range,
                                     n_levels=self._n_levels)
        self.create_levels()

    def create_levels(self):
        """
        Create level objects, each level has own level logger object
        :return: None
        """
        for i_level in range(self._n_levels):
            previous_level = self.levels[-1] if i_level else None
            if self._n_levels == 1:
                level_param = 1
            else:
                level_param = i_level / (self._n_levels - 1)

            # Create level
            level = Level(self.simulation_factory, previous_level, level_param, i_level,
                          self._hdf_object.add_level_group(str(i_level)),
                          self._process_options['regen_failed'], self._process_options['keep_collected'])
            self.levels.append(level)

    @property
    def n_levels(self):
        """
        Number of levels
        """
        if len(self.levels) > 0:
            return len(self.levels)

    @property
    def n_samples(self):
        """
        Level samples
        """
        return np.array([l.n_samples for l in self.levels])

    @property
    def n_nan_samples(self):
        """
        Level nan samples
        """
        return np.array([len(l.nan_samples) for l in self.levels])

    def estimate_diff_vars(self, moments_fn):
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

        for level in self.levels:
            v, n = level.estimate_diff_var(moments_fn)
            vars.append(v)
            n_samples.append(n)
        return np.array(vars), np.array(n_samples)

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

    def _variance_regression(self, raw_vars, sim_steps):
        """
        Estimate level variance by regression from model:

        log(var_l,r) = A_r + B * log(h_l) + C * log^2(hl),
                                            for l = 0, .. L-1
                                            for r = 1, .., R-1

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

    def estimate_diff_vars_regression(self, moments_fn):
        """
        Estimate variances using linear regression model.
        :param moments_fn: Moment evaluation function
        :return: array of variances L x (R-1)
        """
        vars, n_samples = self.estimate_diff_vars(moments_fn)
        sim_steps = np.array([lvl.fine_simulation.step for lvl in self.levels])
        vars = self._variance_regression(vars, sim_steps)
        return vars

    def sample_range(self, n0, nL):
        """
        Geometric sequence of L elements decreasing from n0 to nL.
        Useful to set number of samples explicitly.
        :param n0: int
        :param nL: int
        :return: np.array of length L = n_levels.
        """
        return np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), self.n_levels))).astype(int)

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

        for i, level in enumerate(self.levels):
            level.set_target_n_samples(int(n_samples[i]))


    # def set_target_time(self, target_time):
    #     """
    #     For each level counts new N according to target_time
    #     :return: array
    #     TODO: Have a linear model to estimate true time per sample as a function of level step.
    #           This needs some test sampling... that is same as with variance estimates.
    #     """
    #     vars =
    #     amount = self._count_sum()
    #     # Loop through levels
    #     # Count new number of simulations for each level
    #     for level in self.levels:
    #         new_num_of_sim = np.round((target_time * np.sqrt(level.variance / level.n_ops_estimate()))
    #                                   / amount).astype(int)
    #
    #         self.num_of_simulations.append(new_num_of_sim)

    # def reset_moment_fn(self, moments_fn):
    #     for level in self.levels:
    #         level.reset_moment_fn(moments_fn)

    def target_var_adding_samples(self, target_var, moments_fn, pbs=None, sleep=20, add_coef=0.1):
        """
        Set level target number of samples according to improving estimates.  
        We assume set_initial_n_samples method was called before.
        :param target_var: float, whole mlmc target variance
        :param moments_fn: Object providing calculating moments
        :param pbs: Pbs script generator object
        :param sleep: Time waiting for samples
        :param add_coef: Coefficient for adding samples
        :return: None
        """
        # Get default scheduled samples
        n_scheduled = np.array(self.l_scheduled_samples())
        # Scheduled samples that are greater than already done samples
        greater_items = np.arange(0, len(n_scheduled))

        # Scheduled samples and wait until at least half of the samples are done
        self.set_scheduled_and_wait(n_scheduled, greater_items, pbs, sleep)

        # New estimation according to already finished samples
        n_estimated = np.ceil(np.max(self.estimate_n_samples_for_target_variance(target_var, moments_fn), axis=1))

        # Loop until number of estimated samples is greater than the number of scheduled samples
        while not np.all(n_estimated[greater_items] == n_scheduled[greater_items]):
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
            self.set_scheduled_and_wait(n_scheduled, greater_items, pbs, sleep)

            # New estimation according to already finished samples
            n_estimated = np.ceil(np.max(self.estimate_n_samples_for_target_variance(target_var, moments_fn), axis=1))

    def set_scheduled_and_wait(self, n_scheduled, greater_items, pbs, sleep, fin_sample_coef=0.5):
        """
        Scheduled samples on each level and wait until at least half of the samples is done
        :param n_scheduled: ndarray, number of scheduled samples on each level
        :param greater_items: Items where n_estimated is greater than n_scheduled
        :param pbs: Pbs script generator object
        :param sleep: Time waiting for samples
        :param done_sample_coef: The proportion of samples to finished for further estimate
        :return: None
        """

        # Set scheduled samples and run simulations
        self.set_level_target_n_samples(n_scheduled)
        self.refill_samples()
        # Use PBS job scheduler
        if pbs is not None:
            pbs.execute()

        # Finished level samples
        n_finished = np.array([level.get_n_finished() for level in self.levels])
        # Wait until at least half of the scheduled samples are done on each level
        while np.any(n_finished[greater_items] < fin_sample_coef * n_scheduled[greater_items]):
            # Wait a while
            time.sleep(sleep)
            n_finished = np.array([level.get_n_finished() for level in self.levels])

    def l_scheduled_samples(self):
        """
        Get all levels target number of samples
        :return: list 
        """
        return [level.target_n_samples for level in self.levels]

    def set_level_target_n_samples(self, n_samples, fraction=1.0):
        """
        Set level number of target samples
        :param n_samples: list, each level target samples
        :param fraction: Use just fraction of total samples
        :return: None
        """
        for level, n in zip(self.levels, n_samples):
            level.set_target_n_samples(int(n * fraction))

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

        return n_samples_estimate_safe

    def set_target_variance(self, target_variance, moments_fn=None, fraction=1.0, prescribe_vars=None):
        """
        Estimate optimal number of samples for individual levels that should provide a target variance of
        resulting moment estimate. Number of samples are directly set to levels.
        This also set given moment functions to be used for further estimates if not specified otherwise.
        TODO: separate target_variance per moment
        :param target_variance: Constrain to achieve this variance.
        :param moments_fn: moment evaluation functions
        :param fraction: Plan only this fraction of computed counts.
        :param prescribe_vars: vars[ L, M] for all levels L and moments M safe the (zeroth) constant moment with zero variance.
        :return: None
        """
        n_samples = self.estimate_n_samples_for_target_variance(target_variance, moments_fn, prescribe_vars)
        n_samples = np.max(n_samples, axis=1)
        self.set_level_target_n_samples(n_samples, fraction)

    def refill_samples(self):
        """
        For each level set fine and coarse simulations and generate samples in reverse order (from the finest sim)
        :return: None
        """
        # Set level coarse sim, it creates also fine simulations if its None
        for level in self.levels:
            level.set_coarse_sim()

        # Generate level's samples in reverse order
        for level in reversed(self.levels):
            level.fill_samples()

    def wait_for_simulations(self, sleep=0, timeout=None):
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
            n_running = 0
            for level in self.levels:
                n_running += level.collect_samples()

            time.sleep(sleep)
            if 0 < timeout < (time.clock() - t0):
                break

        return n_running

    def estimate_domain(self):
        """
        Estimate domain of the density function.
        TODO: compute mean and variance and use quantiles of normal or lognormal distribution (done in Distribution)
        :return:
        """
        ranges = np.array([l.sample_range() for l in self.levels])

        return np.min(ranges[:, 0]), np.max(ranges[:, 1])

    def subsample(self, sub_samples=None):
        """
        :param sub_samples: None - use all generated samples
                    array of ints, shape = n_levels; choose given number of sub samples from computed samples
        :return: None
        """
        if sub_samples is None:
            sub_samples = [None] * self.n_levels
        assert len(sub_samples) == self.n_levels
        for ns, level in zip(sub_samples, self.levels):
            level.subsample(ns)

    def update_moments(self, moments_fn):
        for level in self.levels:
            level.evaluate_moments(moments_fn, force=True)

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

        vars = np.sum(np.array(vars) / n_samples[:, None], axis=0)

        return np.array(means), np.array(vars)

    def estimate_cost(self, level_times=None, n_samples=None):
        """
        Estimate total cost of mlmc
        :param level_times: Number of level executions
        :param n_samples: Number of samples on each level
        :return: total cost
        TODO: Have cost expressed as a time estimate. This requires
        estimate of relation ship to the  task size and complexity.
        """
        if level_times is None:
            level_times = [lvl.n_ops_estimate for lvl in self.levels]
        if n_samples is None:
            n_samples = self.n_samples
        return np.sum(level_times * n_samples)

    def clean_levels(self):
        """
        Reset all levels
        :return: None
        """
        for level in self.levels:
            level.reset()

    def clean_subsamples(self):
        """
        Clean level subsamples
        :return: None
        """
        for level in self.levels:
            level.sample_indices = None

    def get_sample_times(self):
        """
        The total average duration of one sample per level (fine + coarse together)
        :return: list
        """
        return [level.sample_time() for level in self.levels]
